"""Domain randomization coverage and physically consistent link mass scaling."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco_warp as mjwarp
import pytest
import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventManager

from supermimic.tasks import domain_randomization as dr_cfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg, maxcoord_env_cfg


@pytest.mark.parametrize("factory", [gencoord_env_cfg, maxcoord_env_cfg])
@pytest.mark.parametrize("play", [False, True])
def test_train_and_play_use_all_randomizations(factory, play, monkeypatch):
  settings = replace(dr_cfg.DR_CONFIG, kp_scale=(0.9, 1.1), com_z=(-0.02, 0.04))
  monkeypatch.setattr(dr_cfg, "DR_CONFIG", settings)
  events = factory(play=play).events
  assert set(events) == {
    "foot_friction",
    "encoder_bias",
    "pd_gains",
    "joint_armature",
    "body_mass",
    "body_com",
  }
  assert all(event.mode == "reset" for event in events.values())
  assert events["pd_gains"].params["kp_range"] == settings.kp_scale
  assert events["pd_gains"].params["asset_cfg"].actuator_names == ".*"
  assert events["joint_armature"].params["asset_cfg"].joint_names == ".*"
  assert events["encoder_bias"].params["asset_cfg"].joint_names == ".*"
  assert events["body_mass"].params["asset_cfg"].body_names == ".*"
  assert events["body_com"].params["asset_cfg"].body_names == ".*"
  assert events["body_com"].params["ranges"][2] == settings.com_z


def test_all_ranges_are_configurable_and_event_configs_are_independent():
  settings = dr_cfg.DomainRandomizationCfg(
    kp_scale=(0.8, 0.9),
    kd_scale=(1.1, 1.2),
    armature_scale=(0.5, 0.8),
    mass_scale=(1.0, 1.1),
    com_x=(0.0, 0.01),
    com_y=(0.0, 0.02),
    com_z=(0.0, 0.03),
    foot_friction=(0.4, 0.6),
    encoder_bias=(-0.005, 0.005),
  )
  events = dr_cfg.make_domain_randomization_events(settings)
  assert events["pd_gains"].params["kp_range"] == settings.kp_scale
  assert events["pd_gains"].params["kd_range"] == settings.kd_scale
  assert events["joint_armature"].params["ranges"] == settings.armature_scale
  assert events["body_mass"].params["ranges"] == settings.mass_scale
  assert events["body_com"].params["ranges"] == {
    0: settings.com_x,
    1: settings.com_y,
    2: settings.com_z,
  }
  assert events["foot_friction"].params["ranges"] == settings.foot_friction
  assert events["encoder_bias"].params["bias_range"] == settings.encoder_bias

  # EventManager resolves selectors in-place. Separate environments must not
  # share those selectors or the mutable COM range dictionary.
  other = dr_cfg.make_domain_randomization_events(settings)
  events["body_mass"].params["asset_cfg"].body_ids = [0]
  events["body_com"].params["ranges"][0] = (0.0, 0.0)
  assert other["body_mass"].params["asset_cfg"].body_ids == slice(None)
  assert other["body_com"].params["ranges"][0] == settings.com_x


def _mass_env():
  defaults = {
    "body_mass": torch.tensor([0.0, 2.0, 3.0, 4.0, 0.0]),
    "body_inertia": torch.tensor(
      [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.0, 0.0, 0.0],
      ]
    ),
  }
  model = SimpleNamespace(
    **{
      field: values.unsqueeze(0).repeat(4, *([1] * values.ndim))
      for field, values in defaults.items()
    }
  )
  env = SimpleNamespace(
    num_envs=4,
    device="cpu",
    sim=SimpleNamespace(model=model, get_default_field=defaults.__getitem__),
    scene={
      "robot": SimpleNamespace(
        indexing=SimpleNamespace(body_ids=torch.tensor([1, 3, 4])),
      )
    },
  )
  return env, defaults


def test_mass_and_inertia_scale_together_without_accumulation_or_other_body_changes():
  env, defaults = _mass_env()
  asset_cfg = dr_cfg.make_domain_randomization_events()["body_mass"].params["asset_cfg"]
  # Non-contiguous environment/body selections exercise local-to-global indexing.
  asset_cfg.body_ids = [0, 2]
  env_ids = torch.tensor([0, 2])
  for _ in range(2):
    dr_cfg.scale_body_mass(env, env_ids, (1.25, 1.25), asset_cfg)
    for field, baseline in defaults.items():
      expected = baseline.unsqueeze(0).repeat(4, *([1] * baseline.ndim))
      expected[env_ids[:, None], torch.tensor([1, 4])] *= 1.25
      torch.testing.assert_close(getattr(env.sim.model, field), expected)
      torch.testing.assert_close(env.sim.get_default_field(field), baseline)
  # These fields make mjlab expand per-world storage and recompute its derived
  # inverse masses/inertias after changing density.
  assert set(dr_cfg.scale_body_mass.model_fields) == {"body_mass", "body_inertia"}


def test_mass_scale_is_independent_per_environment_and_link():
  env, defaults = _mass_env()
  asset_cfg = dr_cfg.make_domain_randomization_events()["body_mass"].params["asset_cfg"]
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(123)
    dr_cfg.scale_body_mass(env, None, (0.75, 1.25), asset_cfg)
  body_ids = torch.tensor([1, 3])
  scale = env.sim.model.body_mass[:, body_ids] / defaults["body_mass"][body_ids]
  assert torch.all((scale >= 0.75) & (scale <= 1.25))
  assert scale.unique().numel() == scale.numel()
  inertia_scale = (
    env.sim.model.body_inertia[:, body_ids] / defaults["body_inertia"][body_ids]
  )
  torch.testing.assert_close(
    inertia_scale, scale.unsqueeze(-1).expand_as(inertia_scale)
  )
  assert torch.count_nonzero(env.sim.model.body_mass[:, 4]) == 0  # Massless link.


def test_reset_events_resample_only_selected_worlds_and_recompute_constants():
  # Use actual G1 selectors, actuators and nominal MuJoCo parameters, with CPU
  # tensors standing in for Warp storage; no GPU or physics stepping is needed.
  cfg = gencoord_env_cfg()
  robot = Entity(cfg.scene.entities["robot"])
  mj_model = robot.compile()
  robot.indexing = robot._compute_indexing(mj_model, "cpu")
  num_envs = 4
  robot._data = SimpleNamespace(encoder_bias=torch.zeros(num_envs, robot.num_joints))
  for actuator in robot.actuators:
    actuator.initialize(mj_model, None, SimpleNamespace(nworld=num_envs), "cpu")
  fields = (
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_armature",
    "body_mass",
    "body_inertia",
    "body_ipos",
  )
  defaults = {
    field: torch.as_tensor(getattr(mj_model, field).copy(), dtype=torch.float32)
    for field in fields
  }
  model = SimpleNamespace(
    **{
      field: baseline.unsqueeze(0).repeat(num_envs, *([1] * baseline.ndim))
      for field, baseline in defaults.items()
    }
  )
  sim = SimpleNamespace(
    model=model,
    mj_model=mj_model,
    per_world_default_fields=set(),
    get_default_field=defaults.__getitem__,
    recompute_constants=Mock(),
  )
  env = SimpleNamespace(
    sim=sim,
    scene={"robot": robot},
    device="cpu",
    num_envs=num_envs,
  )
  manager = EventManager(cfg.events, env)
  assert robot.num_joints == robot.num_actuators == 29
  assert robot.num_bodies == 30
  assert {"body_subtreemass", "body_invweight0", "dof_invweight0"} <= set(
    manager.domain_randomization_fields
  )
  for name, selector_attr in (
    ("pd_gains", "actuator_ids"),
    ("joint_armature", "joint_ids"),
    ("body_mass", "body_ids"),
    ("body_com", "body_ids"),
  ):
    selector = manager.get_term_cfg(name).params["asset_cfg"]
    assert getattr(selector, selector_attr) == slice(None)

  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(7)
    # First reset initializes every world's randomized parameters.
    manager.apply("reset", env_ids=torch.arange(num_envs), global_env_step_count=0)
    previous = {field: getattr(model, field).clone() for field in fields}
    previous_bias = robot.data.encoder_bias.clone()
    env_ids = torch.tensor([0, 2])
    manager.apply("reset", env_ids=env_ids, global_env_step_count=1)

  for field in fields:
    actual = getattr(model, field)
    torch.testing.assert_close(actual[[1, 3]], previous[field][[1, 3]])
    assert not torch.equal(actual[env_ids], previous[field][env_ids]), field
  torch.testing.assert_close(robot.data.encoder_bias[[1, 3]], previous_bias[[1, 3]])
  assert not torch.equal(robot.data.encoder_bias[env_ids], previous_bias[env_ids])
  assert sim.recompute_constants.call_count == 2
  sim.recompute_constants.assert_called_with(mjwarp.SetConstGroup.ALL)

  # Repeated reset draws remain relative to the original model, not the last draw.
  for field, ids, axis in (
    ("actuator_gainprm", robot.indexing.ctrl_ids, 0),
    ("actuator_biasprm", robot.indexing.ctrl_ids, 2),
    ("dof_armature", robot.indexing.joint_v_adr, None),
    ("body_mass", robot.indexing.body_ids, None),
  ):
    actual, default = getattr(model, field)[:, ids], defaults[field][ids]
    if axis is not None:
      actual, default = actual[..., axis], default[..., axis]
    ratio = actual / default
    assert torch.all((ratio >= 0.75) & (ratio <= 1.25)), field
  offsets = model.body_ipos - defaults["body_ipos"]
  assert torch.all(offsets.abs() <= 0.030001)
  inertia_ratio = model.body_inertia[:, 1:] / defaults["body_inertia"][1:]
  mass_ratio = model.body_mass[:, 1:] / defaults["body_mass"][1:]
  torch.testing.assert_close(
    inertia_ratio, mass_ratio[..., None].expand_as(inertia_ratio)
  )
