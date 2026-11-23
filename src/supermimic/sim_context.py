"""Per-env sim-context conditioning for the world model."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEGENERATE_HALF_WIDTH = 1e-3
"""Half-widths at or below this are treated as constant. Float32 ULP for
O(1) values is ~1e-7; dividing ``(raw - mean)`` by ``half_width`` amplifies
that noise by ``1/half_width``. At 1e-3 the worst-case amplified noise is
~1e-4 — well below the [-1, 1] signal range. Below ~1e-6 the noise
overwhelms the signal entirely. All default DR ranges
(foot_friction ±0.45, base_com ±0.025–0.05, encoder_bias ±0.01) clear
this threshold."""


def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
  """Range → (mean, scale). Half-widths below the precision floor get
  scale=1.0 so ``(raw - mean) ≈ 0`` instead of amplifying float-ULP noise."""
  assert hi >= lo, f"empty range: lo={lo}, hi={hi}"
  half = (hi - lo) / 2.0
  return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


_ENTITY_IDS_ATTR = {"body": "body_ids", "site": "site_ids", "geom": "geom_ids"}


class SimContextSpec(ABC):
  name: str
  dim: int
  representation = "absolute"

  def schema(self) -> dict:
    """Describe column meaning without scene-dependent global model indices."""
    return {
      "name": self.name,
      "channels": getattr(
        self, "channels", [f"{self.name}/{i}" for i in range(self.dim)]
      ),
      "representation": self.representation,
    }

  @abstractmethod
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """Resolve indices and ``dim``; return each column's DR range ``(lo, hi)``."""
    ...

  @abstractmethod
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return raw per-env values of shape ``(num_envs, dim)``."""
    ...

  @staticmethod
  def _global_entity_id(asset, entity_type: str, local_idx: int) -> int:
    """Asset-local entity index → mjModel global index. DR writes at global
    ids; reading at asset-local silently returns an unrandomized neighbor."""
    return int(getattr(asset.indexing, _ENTITY_IDS_ATTR[entity_type])[local_idx].item())


class FootFrictionSpec(SimContextSpec):
  """Per-env tangential foot friction read from one matching geom.

  mjlab's ``foot_friction`` event uses ``shared_random=True``, so all matching
  geoms in one env carry the same value; we read from the first match.
  """

  name = "foot_friction"
  dim = 1

  def __init__(self) -> None:
    self._geom_idx: int | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"FootFrictionSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]

    asset_cfg = params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    pattern = asset_cfg.geom_names
    if isinstance(pattern, str):
      pat = re.compile(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if pat.match(name)]
    elif isinstance(pattern, (list, tuple)) and pattern:
      name_set = set(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if name in name_set]
    else:
      raise ValueError(f"FootFrictionSpec: '{self.name}' has empty geom_names")
    if not matches:
      raise ValueError(
        f"FootFrictionSpec: no geoms in '{asset_cfg.name}' match {pattern!r}"
      )
    self._geom_idx = self._global_entity_id(asset, "geom", matches[0])
    self.channels = [f"{asset_cfg.name}/foot_friction"]
    return [ranges[0] if isinstance(ranges, dict) else ranges]

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._geom_idx is None:
      raise RuntimeError("FootFrictionSpec.read() called before bind()")
    return env.sim.model.geom_friction[:, self._geom_idx : self._geom_idx + 1, 0]


class CoMOffsetSpec(SimContextSpec):
  name = "base_com"
  dim = 3

  def __init__(self) -> None:
    self._body_id: int | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"CoMOffsetSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]
    asset_cfg = params["asset_cfg"]
    if len(asset_cfg.body_names) != 1:
      raise ValueError(
        f"CoMOffsetSpec: expected exactly 1 body_name, got {asset_cfg.body_names!r}"
      )
    body_name = asset_cfg.body_names[0]
    asset = env.scene[asset_cfg.name]
    self._body_id = self._global_entity_id(
      asset, "body", asset.body_names.index(body_name)
    )
    self.channels = [f"{asset_cfg.name}/{body_name}/{axis}" for axis in "xyz"]
    return [ranges[i] for i in range(self.dim)]

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._body_id is None:
      raise RuntimeError("CoMOffsetSpec.read() called before bind()")
    return env.sim.model.body_ipos[:, self._body_id, :]


class EncoderBiasSpec(SimContextSpec):
  name = "encoder_bias"
  dim = 0  # set in bind() from asset.num_joints

  def __init__(self) -> None:
    self._asset_name: str | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"EncoderBiasSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    asset_cfg = params["asset_cfg"]
    self._asset_name = asset_cfg.name
    asset = env.scene[self._asset_name]
    self.dim = asset.num_joints
    joint_names = getattr(asset, "joint_names", range(self.dim))
    self.channels = [f"{self._asset_name}/{name}" for name in joint_names]
    return [params["bias_range"]] * self.dim

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._asset_name is None:
      raise RuntimeError("EncoderBiasSpec.read() called before bind()")
    return env.scene[self._asset_name].data.encoder_bias


def _event_asset(env: ManagerBasedRlEnv, name: str, operation: str):
  event = env.cfg.events.get(name)
  if event is None:
    raise RuntimeError(f"No event '{name}' on env.cfg.events")
  params = event.params
  if params.get("operation", operation) != operation:
    raise ValueError(f"Sim context '{name}' requires operation={operation!r}")
  return params, env.scene[params["asset_cfg"].name]


def _nominal_field(env: ManagerBasedRlEnv, field: str, ids: torch.Tensor):
  """Read unrandomized compiled defaults, including per-world variant defaults."""
  default = env.sim.get_default_field(field)
  if field in getattr(env.sim, "per_world_default_fields", ()):
    return default[:, ids].clone()
  return default[ids].unsqueeze(0).clone()


class _ModelParameterSpec(SimContextSpec):
  """Scale/offset context based on original model values, never sampled caches."""

  dim = 0
  representation = "scale"

  def __init__(self) -> None:
    self._ids: torch.Tensor | None = None
    self._nominal: torch.Tensor | None = None

  def _transform(self, raw: torch.Tensor) -> torch.Tensor:
    assert self._nominal is not None
    if self.representation == "offset":
      return raw - self._nominal
    # Multiplicative DR cannot change zero-mass links or zero armatures. Their
    # scale is unobservable, so represent them by the neutral scale of one.
    nonzero = self._nominal != 0
    denominator = torch.where(nonzero, self._nominal, 1.0)
    return torch.where(nonzero, raw / denominator, 1.0)

  def _require_bound(self) -> torch.Tensor:
    if self._ids is None:
      raise RuntimeError(f"{type(self).__name__}.read() called before bind()")
    return self._ids

  def schema(self) -> dict:
    result = super().schema()
    if self._nominal is not None:
      nominal = self._nominal.detach().cpu()
      # A shared baseline does not depend on the number of play environments.
      result["nominal"] = (nominal[0] if nominal.shape[0] == 1 else nominal).tolist()
    return result


class PdGainsSpec(_ModelParameterSpec):
  """All joint kp scales followed by all kd scales, both in joint order."""

  name = "pd_gains"

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    from mjlab.actuator import BuiltinPositionActuator
    from mjlab.actuator.actuator import TransmissionType

    params, asset = _event_asset(env, self.name, "scale")
    controls: dict[int, int] = {}
    for actuator in asset.actuators:
      if not isinstance(actuator, BuiltinPositionActuator) or (
        actuator.transmission_type != TransmissionType.JOINT
      ):
        raise ValueError("pd_gains context requires joint BuiltinPositionActuators")
      for joint, control in zip(
        actuator.target_ids.tolist(), actuator.global_ctrl_ids.tolist(), strict=True
      ):
        if joint in controls:
          raise ValueError(f"pd_gains context: joint {joint} has multiple actuators")
        controls[joint] = control
    if set(controls) != set(range(asset.num_joints)):
      raise ValueError("pd_gains context requires one position actuator per joint")
    self._ids = torch.tensor(
      [controls[joint] for joint in range(asset.num_joints)],
      dtype=torch.long,
      device=env.device,
    )
    kp = _nominal_field(env, "actuator_gainprm", self._ids)[..., 0]
    kd = -_nominal_field(env, "actuator_biasprm", self._ids)[..., 2]
    self._nominal = torch.cat((kp, kd), dim=-1)
    self.dim = 2 * asset.num_joints
    asset_name = params["asset_cfg"].name
    self.channels = [
      f"{asset_name}/{gain}/{joint}"
      for gain in ("kp", "kd")
      for joint in asset.joint_names
    ]
    return [params["kp_range"]] * asset.num_joints + [
      params["kd_range"]
    ] * asset.num_joints

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    ids = self._require_bound()
    kp = env.sim.model.actuator_gainprm[:, ids, 0]
    kd = -env.sim.model.actuator_biasprm[:, ids, 2]
    return self._transform(torch.cat((kp, kd), dim=-1))


class JointArmatureSpec(_ModelParameterSpec):
  name = "joint_armature"

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    params, asset = _event_asset(env, self.name, "scale")
    self._ids = asset.indexing.joint_v_adr.to(dtype=torch.long)
    if self._ids.numel() != asset.num_joints:
      raise ValueError("joint_armature context requires single-DoF joints")
    self._nominal = _nominal_field(env, "dof_armature", self._ids)
    self.dim = asset.num_joints
    self.channels = [
      f"{params['asset_cfg'].name}/{joint}" for joint in asset.joint_names
    ]
    return [params["ranges"]] * self.dim

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return self._transform(env.sim.model.dof_armature[:, self._require_bound()])


class BodyMassScaleSpec(_ModelParameterSpec):
  name = "body_mass"

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    params, asset = _event_asset(env, self.name, "scale")
    self._ids = asset.indexing.body_ids.to(dtype=torch.long)
    self._nominal = _nominal_field(env, "body_mass", self._ids)
    self.dim = len(asset.body_names)
    self.channels = [f"{params['asset_cfg'].name}/{body}" for body in asset.body_names]
    return [params["ranges"]] * self.dim

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return self._transform(env.sim.model.body_mass[:, self._require_bound()])


class BodyCoMOffsetSpec(_ModelParameterSpec):
  """All local inertial-position offsets, relative to each body's nominal COM."""

  name = "body_com"
  representation = "offset"

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    params, asset = _event_asset(env, self.name, "add")
    self._ids = asset.indexing.body_ids.to(dtype=torch.long)
    nominal = _nominal_field(env, "body_ipos", self._ids)
    self._nominal = nominal.flatten(start_dim=1)
    self.dim = 3 * len(asset.body_names)
    self.channels = [
      f"{params['asset_cfg'].name}/{body}/{axis}"
      for body in asset.body_names
      for axis in "xyz"
    ]
    ranges = params["ranges"]
    axes = [ranges[i] for i in range(3)] if isinstance(ranges, dict) else [ranges] * 3
    return axes * len(asset.body_names)

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    raw = env.sim.model.body_ipos[:, self._require_bound(), :].flatten(start_dim=1)
    return self._transform(raw)


class SimContext(nn.Module):
  def __init__(self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]):
    super().__init__()
    self.specs = list(specs)
    self._ranges = [list(r) for r in ranges]
    stats = [_mean_scale(lo, hi) for lo, hi in ranges]
    self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
    self.register_buffer("scale", torch.tensor([s for _, s in stats]))

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> SimContext:
    """Discover, build, bind, and validate from the env's active DR events.

    Spec order here is the column layout of the normalized context: the replay
    buffer's ``sim_ctx`` field and the Dynamics input.
    """
    candidates: tuple[type[SimContextSpec], ...] = (
      FootFrictionSpec,
      CoMOffsetSpec,
      EncoderBiasSpec,
      PdGainsSpec,
      JointArmatureSpec,
      BodyMassScaleSpec,
      BodyCoMOffsetSpec,
    )
    specs = [
      candidate() for candidate in candidates if candidate.name in env.cfg.events
    ]
    if not specs:
      raise RuntimeError(
        "No sim_ctx specs derived from env.cfg.events. "
        "Enable at least one supported DR event in the task cfg."
      )
    return cls(specs, [r for s in specs for r in s.bind(env)]).to(env.device)

  def schema(self) -> dict:
    """JSON-compatible layout/baselines for checkpoint compatibility checks."""
    specs = []
    start = 0
    for spec in self.specs:
      entry = spec.schema()
      entry["ranges"] = self._ranges[start : start + spec.dim]
      specs.append(entry)
      start += spec.dim
    return {"version": 1, "specs": specs}

  @property
  def output_dim(self) -> int:
    return sum(spec.dim for spec in self.specs)

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized per-env context ``(num_envs, output_dim)``, in spec order."""
    return (torch.cat([s.read(env) for s in self.specs], -1) - self.mean) / self.scale

  def find(self, name: str) -> slice | None:
    """The columns spec ``name`` occupies in a normalized context."""
    offset = 0
    for spec in self.specs:
      if spec.name == name:
        return slice(offset, offset + spec.dim)
      offset += spec.dim
    return None

  def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
    """Raw values of ``columns`` in the normalized context ``sim_ctx``."""
    return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
