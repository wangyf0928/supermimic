"""Weight warm starts restore model state without replacing optimizer parameters."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.models import Dynamics
from supermimic.representations import create_representation
from supermimic.runners import supermimic_runner as runner
from supermimic.sim_context import FootFrictionSpec, SimContext


def _models(marker, *, with_critic=True):
  representation = create_representation(
    "gencoord", 17, 2, 0.02, 4, torch.zeros(2), "rep_state"
  )
  dynamics = Dynamics(
    representation, 2, 1, torch.ones(2), torch.zeros(2), hidden_dims=(8,)
  )
  context = SimContext([FootFrictionSpec()], [(0.3, 1.2)])
  wm = runner.WorldModel(representation, dynamics, context, num_goals=1, num_actions=2)
  actor = torch.nn.Linear(4, 2)
  wm_env = SimpleNamespace(
    dynamics=dynamics, representation=representation, max_step_reward=1.0
  )
  if with_critic:
    learner = ActorCriticPolicyLearner(
      actor, torch.nn.Linear(4, 1), wm_env, ActorCriticPolicyLearnerCfg(), device="cpu"
    )
  else:
    learner = BpttPolicyLearner(actor, wm_env, BpttPolicyLearnerCfg(), device="cpu")
  with torch.no_grad():
    modules = [dynamics, actor]
    if with_critic:
      modules.append(learner.critic)
    for module in modules:
      for parameter in module.parameters():
        parameter.fill_(marker)
    for normalizer, width in [
      (dynamics.state_normalizer, 14),
      (dynamics.action_normalizer, 2),
    ]:
      normalizer.update(torch.arange(5 * width).reshape(5, width).float() + marker)
    context.mean.fill_(marker)
    context.scale.fill_(marker + 1.0)
  return wm, learner


def _save(directory, wm, learner, *, omit=None):
  directory.mkdir(parents=True, exist_ok=True)
  contents = {"world_model.pt": wm.bundle(), "actor.pt": learner.actor.state_dict()}
  if hasattr(learner, "critic"):
    contents["critic.pt"] = learner.critic.state_dict()
  for filename, state in contents.items():
    if filename != omit:
      torch.save(state, directory / filename)


def _snapshot(wm, learner):
  modules = {"dynamics": wm.dynamics, "context": wm.sim_context, "actor": learner.actor}
  if hasattr(learner, "critic"):
    modules["critic"] = learner.critic
  return deepcopy({name: module.state_dict() for name, module in modules.items()})


def _assert_state_equal(actual, expected):
  assert actual.keys() == expected.keys()
  for module, state in actual.items():
    assert state.keys() == expected[module].keys()
    for key, tensor in state.items():
      assert torch.equal(tensor, expected[module][key]), f"{module}.{key}"


@pytest.mark.parametrize("remote", [False, True])
def test_resume_restores_all_weights_and_normalization(tmp_path, monkeypatch, remote):
  source_wm, source_learner = _models(0.6)
  _save(tmp_path, source_wm, source_learner)
  wm, learner = _models(0.1)
  before = _snapshot(wm, learner)
  expected = _snapshot(source_wm, source_learner)
  assert not torch.equal(
    before["dynamics"]["state_normalizer._mean"],
    expected["dynamics"]["state_normalizer._mean"],
  )
  api = MagicMock()
  api_factory = MagicMock(return_value=api)
  download = MagicMock(return_value=tmp_path)
  monkeypatch.setattr(runner.wandb, "Api", api_factory)
  monkeypatch.setattr(runner, "download_model_dir", download)
  source = "wandb://team/supermimic/run123" if remote else str(tmp_path)
  original_load = torch.load

  def read_checkpoint(*args, **kwargs):
    # Read every file successfully before applying the first restored tensor.
    _assert_state_equal(_snapshot(wm, learner), before)
    return original_load(*args, **kwargs)

  load = MagicMock(side_effect=read_checkpoint)
  monkeypatch.setattr(runner.torch, "load", load)

  runner.load_training_checkpoint(source, wm, learner, "cpu")

  _assert_state_equal(_snapshot(wm, learner), expected)
  assert load.call_count == 3
  for call in load.call_args_list:
    assert call.kwargs["weights_only"] is True
    assert call.kwargs["map_location"] == "cpu"
  if remote:
    api.run.assert_called_once_with("team/supermimic/run123")
    download.assert_called_once_with(api.run.return_value)
  else:
    api_factory.assert_not_called()
    download.assert_not_called()


def test_bptt_resume_does_not_require_a_critic(tmp_path):
  source_wm, source_learner = _models(0.6, with_critic=False)
  _save(tmp_path, source_wm, source_learner)
  assert not (tmp_path / "critic.pt").exists()
  wm, learner = _models(0.1, with_critic=False)
  runner.load_training_checkpoint(str(tmp_path), wm, learner, "cpu")
  _assert_state_equal(_snapshot(wm, learner), _snapshot(source_wm, source_learner))


@pytest.mark.parametrize("missing", ["world_model.pt", "actor.pt", "critic.pt"])
def test_missing_checkpoint_file_fails_before_loading_any_weights(tmp_path, missing):
  source_wm, source_learner = _models(0.6)
  _save(tmp_path, source_wm, source_learner, omit=missing)
  wm, learner = _models(0.1)
  before = _snapshot(wm, learner)
  with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
    runner.load_training_checkpoint(str(tmp_path), wm, learner, "cpu")
  _assert_state_equal(_snapshot(wm, learner), before)


def test_resume_preserves_parameter_objects_and_existing_optimizers(tmp_path):
  source_wm, source_learner = _models(0.6)
  _save(tmp_path, source_wm, source_learner)
  wm, learner = _models(0.1)
  wm_optimizer = torch.optim.Adam(wm.dynamics.parameters(), lr=0.01)
  optimizers = [wm_optimizer, *learner.actor_optimizers, *learner.critic_optimizers]
  modules = [wm.dynamics, learner.actor, learner.critic]
  parameters = [parameter for module in modules for parameter in module.parameters()]
  parameter_ids = [id(parameter) for parameter in parameters]
  optimizer_ids = [
    [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    for optimizer in optimizers
  ]
  buffer_ptrs = {
    name: buffer.data_ptr() for name, buffer in wm.dynamics.named_buffers()
  }

  runner.load_training_checkpoint(str(tmp_path), wm, learner, "cpu")

  assert [
    id(parameter) for module in modules for parameter in module.parameters()
  ] == parameter_ids
  assert [id(parameter) for parameter in learner.actor_params] == [
    id(parameter) for parameter in learner.actor.parameters()
  ]
  assert buffer_ptrs == {
    name: buffer.data_ptr() for name, buffer in wm.dynamics.named_buffers()
  }
  for optimizer, expected_ids in zip(optimizers, optimizer_ids, strict=True):
    assert [
      id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ] == expected_ids
    optimizer.zero_grad(set_to_none=True)
  loaded = [parameter.detach().clone() for parameter in parameters]
  sum(parameter.square().sum() for parameter in parameters).backward()
  for optimizer in optimizers:
    optimizer.step()
  assert all(
    not torch.equal(parameter, before)
    for parameter, before in zip(parameters, loaded, strict=True)
  )


def test_resume_rejects_incompatible_actor_weights_strictly(tmp_path):
  source_wm, source_learner = _models(0.6)
  _save(tmp_path, source_wm, source_learner)
  torch.save(torch.nn.Linear(5, 2).state_dict(), tmp_path / "actor.pt")
  wm, learner = _models(0.1)
  with pytest.raises(RuntimeError, match="size mismatch"):
    runner.load_training_checkpoint(str(tmp_path), wm, learner, "cpu")


def test_resume_and_world_model_only_warm_start_are_mutually_exclusive():
  with pytest.raises(ValueError, match="resume"):
    runner.SuperMimicRunnerCfg(
      resume="checkpoints/run", wandb_run_path="team/project/run"
    )


def test_runner_restores_after_both_learners_are_initialized(monkeypatch):
  calls = []
  wm = object()
  learner = object()

  def init_common(self):
    self._wm = wm
    calls.append("world_model")

  def init_policy(self):
    self.policy_learner = learner
    calls.append("policy")

  def load(source, supplied_wm, supplied_learner, device):
    assert calls == ["world_model", "policy"]
    assert (source, supplied_wm, supplied_learner, device) == (
      "checkpoints/run",
      wm,
      learner,
      "cpu",
    )
    calls.append("restore")

  monkeypatch.setattr(runner.SuperMimicRunner, "_init_common", init_common)
  monkeypatch.setattr(runner.SuperMimicRunner, "_init_policy", init_policy)
  monkeypatch.setattr(runner, "load_training_checkpoint", load)
  monkeypatch.setattr(runner, "Logger", MagicMock())
  runner.SuperMimicRunner(
    SimpleNamespace(num_envs=2, max_episode_length=100),
    runner.SuperMimicRunnerCfg(resume="checkpoints/run"),
    task_id="SuperMimic-GenCoord-G1",
    device="cpu",
  )
  assert calls == ["world_model", "policy", "restore"]
