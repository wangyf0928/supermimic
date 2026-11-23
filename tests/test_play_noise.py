"""Play sensor noise changes policy inputs without changing clean model state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.representations import create_representation
from supermimic.scripts.viewers import SuperMimicViewerMixin

NUM_JOINTS = 3
NUM_BODIES = 4


def _representation(name="gencoord", mode="rep_state"):
  return create_representation(
    name,
    13 + 2 * NUM_JOINTS if name == "gencoord" else 13 * NUM_BODIES,
    num_actions=NUM_JOINTS,
    dt=0.02,
    decimation=4,
    default_joint_pos=torch.zeros(NUM_JOINTS),
    actor_obs_mode=mode,
  )


def _expected_bounds(name):
  if name == "gencoord":
    # [height, gravity, linear velocity, angular velocity, q, qdot].
    return torch.tensor(
      [0.0] + [0.05] * 3 + [0.0] * 3 + [0.2] * 3 + [0.01] * 3 + [0.5] * 3
    )
  bounds = torch.zeros(15 * NUM_BODIES + 4)
  bounds[12 * NUM_BODIES : 12 * NUM_BODIES + 3] = 0.2
  bounds[-3:] = 0.05
  return bounds


@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_rep_state_noise_has_only_defined_sensor_ranges(name):
  rep = _representation(name)
  clean = torch.zeros(128, rep.rep_state_dim)
  before = clean.clone()
  noisy = rep.add_rep_state_noise(clean)
  bounds = _expected_bounds(name)
  assert torch.equal(rep.rep_state_noise_bounds("cpu"), bounds)
  assert noisy.shape == clean.shape
  assert torch.equal(clean, before)
  assert noisy.data_ptr() != clean.data_ptr()
  assert torch.equal(noisy[:, bounds == 0], clean[:, bounds == 0])
  assert torch.all(noisy.abs() <= bounds)
  assert torch.all(noisy[:, bounds > 0].abs().sum(dim=0) > 0)
  assert not torch.equal(noisy[0], noisy[1])


@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_play_noise_buffer_does_not_change_checkpoint_shape_or_keys(name):
  rep = _representation(name)
  expected_dim = 10 + 2 * NUM_JOINTS if name == "gencoord" else 15 * NUM_BODIES + 4
  assert rep.rep_state_dim == expected_dim
  assert rep._rep_state_noise_bounds.shape == (expected_dim,)
  assert "_rep_state_noise_bounds" in dict(rep.named_buffers())
  assert "_rep_state_noise_bounds" not in rep.state_dict()
  # Device/dtype moves still carry the nonpersistent buffer.
  rep.to(dtype=torch.float64)
  assert rep._rep_state_noise_bounds.dtype == torch.float64


@pytest.mark.parametrize("mode", ["rep_state", "proprio"])
def test_real_sim_viewer_sensor_adds_noise_exactly_once(mode, monkeypatch):
  rep = _representation(mode=mode)
  viewer = SuperMimicViewerMixin()
  viewer.representation = rep
  viewer.sim_ctx = torch.zeros(2, 1)
  state = torch.zeros(2, rep.rep_state_dim)
  kwargs = {}
  if mode == "proprio":
    kwargs["sim_actor_obs"] = torch.zeros(2, rep.proprio_obs_dim(NUM_JOINTS))
  clean = rep.compute_actor_sensor(state, viewer.sim_ctx, **kwargs).clone()
  random_calls = []

  def fixed_random(tensor):
    random_calls.append(tensor.shape)
    return torch.full_like(tensor, 0.75)

  monkeypatch.setattr(torch, "rand_like", fixed_random)
  noisy = viewer._play_actor_sensor(state, **kwargs)
  bounds = (
    _expected_bounds("gencoord")
    if mode == "rep_state"
    else rep.proprio_noise_bounds(NUM_JOINTS, "cpu")
  )
  torch.testing.assert_close(noisy, clean + bounds * 0.5)
  assert random_calls == [clean.shape]
  assert torch.equal(state, torch.zeros_like(state))


@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_existing_world_model_training_rep_state_remains_clean(name):
  rep = _representation(name)
  wm_env = WorldModelEnv.__new__(WorldModelEnv)
  wm_env.representation = rep
  wm_env.rep_state = torch.randn(2, rep.rep_state_dim)
  wm_env.sim_ctx = torch.zeros(2, 1)
  wm_env.aux_state = torch.zeros(2, 7)
  wm_env.joint_state = torch.zeros(2, NUM_JOINTS * 2)
  wm_env.last_actions = torch.zeros(2, NUM_JOINTS)
  # _synth_sensor passes noisy=True; that existing training path must stay clean
  # for rep_state actors. Play adds its noise at a separate policy boundary.
  before = wm_env.rep_state.clone()
  assert torch.equal(wm_env._synth_sensor(), before)
  assert torch.equal(wm_env.rep_state, before)


@pytest.mark.parametrize("mode", ["rep_state", "proprio"])
def test_play_step_noises_real_sim_policy_but_keeps_wm_policy_clean(mode, monkeypatch):
  rep = _representation(mode=mode)
  full_state = torch.zeros(1, rep.full_state_dim)
  full_state[:, 2] = 0.8
  full_state[:, 3] = 1
  clean_state = rep.represent(full_state).clone()
  clean_sensor = rep.compute_actor_sensor(
    clean_state,
    torch.zeros(1, 1),
    aux_state=rep.extract_aux_state(full_state),
    joint_state=rep.joint_state(clean_state),
    last_action=torch.zeros(1, NUM_JOINTS),
  ).clone()
  goal = torch.zeros(1, 100)
  motion_command = SimpleNamespace(time_steps=torch.zeros(1), motion=object())
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda _: motion_command),
      sim=SimpleNamespace(mj_model=SimpleNamespace(nq=7 + NUM_JOINTS)),
    ),
    get_observations=lambda: {
      "full_state": full_state,
      "goal": goal,
      "actor": clean_sensor,
    },
    step=MagicMock(return_value=(None, None, torch.tensor([False]), None)),
  )
  rep.compute_wm_goal = MagicMock(return_value=(goal, goal))
  viewer = SuperMimicViewerMixin()
  viewer.env = env
  viewer.representation = rep
  viewer.policy = MagicMock(return_value=torch.zeros(1, NUM_JOINTS))
  viewer.policy.history_length = 2
  viewer.dynamics = MagicMock(side_effect=lambda rs, action, aux, ctx: (aux, rs))
  viewer.dynamics.num_actions = NUM_JOINTS
  viewer.sim_context = SimpleNamespace(read=lambda _: torch.zeros(1, 1))
  viewer.current_task = "motion"
  viewer._is_paused = False
  viewer._step_count = viewer._stats_steps = 0
  viewer.num_steps_rollout = 10
  viewer.free_joint_q_adr = np.arange(7)
  viewer.joint_q_adr = np.arange(7, 7 + NUM_JOINTS)
  viewer.pause = MagicMock()
  viewer.reset_env_supermimic()
  monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))

  assert viewer.step_and_predict(), getattr(viewer, "_last_error", "play step failed")
  assert viewer.policy.call_count == 2
  bounds = (
    _expected_bounds("gencoord")
    if mode == "rep_state"
    else rep.proprio_noise_bounds(NUM_JOINTS, "cpu")
  )
  gt_history = viewer.policy.call_args_list[0].args[0]["history"]
  wm_history = viewer.policy.call_args_list[1].args[0]["history"]
  noisy_sensor = clean_sensor + bounds * 0.5
  torch.testing.assert_close(gt_history, noisy_sensor[:, None].repeat(1, 2, 1))
  assert torch.equal(wm_history, clean_sensor[:, None].repeat(1, 2, 1))
  # The dynamics receives the original predicted state, and its output remains
  # clean in the viewer. Policy observation noise never mutates either tensor.
  assert torch.equal(viewer.dynamics.call_args.args[0], clean_state)
  assert torch.equal(viewer.curr_pred_rep_state, clean_state)
  assert torch.equal(rep.represent(full_state), clean_state)
  viewer.pause.assert_not_called()
