"""Shared training/play domain randomization settings.

Edit ``DR_CONFIG`` below to tune the ranges. Scale ranges of ``(1.0, 1.0)``
and offset ranges of ``(0.0, 0.0)`` disable the corresponding perturbation.
All parameters are resampled on every reset, independently for each environment
and each joint/body (foot friction is shared across one robot's foot geoms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class DomainRandomizationCfg:
  kp_scale: tuple[float, float] = (0.75, 1.25)
  kd_scale: tuple[float, float] = (0.75, 1.25)
  armature_scale: tuple[float, float] = (0.75, 1.25)
  mass_scale: tuple[float, float] = (0.75, 1.25)
  com_x: tuple[float, float] = (-0.03, 0.03)  # Body-local meters.
  com_y: tuple[float, float] = (-0.03, 0.03)
  com_z: tuple[float, float] = (-0.03, 0.03)
  foot_friction: tuple[float, float] = (0.3, 1.2)
  encoder_bias: tuple[float, float] = (-0.01, 0.01)  # Radians.


# Gains/mass/COM follow EMAN's teleop configuration; the armature range follows
# its commented-out omni configuration. Existing SuperMimic friction/bias remain.
DR_CONFIG = DomainRandomizationCfg()


@requires_model_fields("body_mass", "body_inertia")
def scale_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Uniformly scale each link's density, preserving its COM and inertia axes.

  mjlab's ``body_mass`` only changes mass. Scaling principal inertia by the same
  factor keeps density changes physically consistent; the decorator also ensures
  mjlab updates the dependent simulation constants after the event.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  body_ids = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
  scales = torch.empty(
    (len(env_ids), len(body_ids)),
    device=env.device,
    dtype=env.sim.model.body_mass.dtype,
  ).uniform_(*ranges)
  for field in ("body_mass", "body_inertia"):
    defaults = env.sim.get_default_field(field)[body_ids]
    factor = scales if field == "body_mass" else scales.unsqueeze(-1)
    getattr(env.sim.model, field)[env_ids[:, None], body_ids] = defaults * factor


def make_domain_randomization_events(
  cfg: DomainRandomizationCfg | None = None,
) -> dict[str, EventTermCfg]:
  """Build fresh event configs for both training and play, covering every link."""
  cfg = DR_CONFIG if cfg is None else cfg
  return {
    "foot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "ranges": cfg.foot_friction,
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "bias_range": cfg.encoder_bias,
      },
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
        "kp_range": cfg.kp_scale,
        "kd_range": cfg.kd_scale,
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "ranges": cfg.armature_scale,
        "operation": "scale",
      },
    ),
    "body_mass": EventTermCfg(
      func=scale_body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "ranges": cfg.mass_scale,
      },
    ),
    "body_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "ranges": {0: cfg.com_x, 1: cfg.com_y, 2: cfg.com_z},
        "operation": "add",
      },
    ),
  }
