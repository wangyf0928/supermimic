"""Checkpoint context semantics and saved normalization must stay aligned."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from supermimic.runners.supermimic_runner import WorldModel
from supermimic.sim_context import SimContext


def _world_model(*, channel="kp/hip", nominal=10.0, ranges=(0.75, 1.25)):
  context = SimContext([], [ranges])
  schema = {
    "version": 1,
    "specs": [
      {
        "name": "pd_gains",
        "channels": [channel],
        "representation": "scale",
        "nominal": [nominal],
        "ranges": [list(ranges)],
      }
    ],
  }
  context.schema = lambda: deepcopy(schema)
  return WorldModel(
    representation=SimpleNamespace(),
    dynamics=torch.nn.Linear(2, 2),
    sim_context=context,
    num_goals=1,
    num_actions=1,
  )


def test_checkpoint_roundtrip_keeps_training_normalizer_with_new_play_ranges(tmp_path):
  trained = _world_model()
  checkpoint = tmp_path / "world_model.pt"
  torch.save(trained.bundle(), checkpoint)
  playing = _world_model(ranges=(0.5, 1.75))
  assert not torch.equal(playing.sim_context.mean, trained.sim_context.mean)
  assert not torch.equal(playing.sim_context.scale, trained.sim_context.scale)
  playing.load_bundle(torch.load(checkpoint, weights_only=True))
  torch.testing.assert_close(playing.sim_context.scale, trained.sim_context.scale)
  torch.testing.assert_close(playing.sim_context.mean, trained.sim_context.mean)
  torch.testing.assert_close(playing.dynamics.weight, trained.dynamics.weight)


@pytest.mark.parametrize("change", [{"channel": "kp/knee"}, {"nominal": 20.0}])
def test_same_dimension_different_physics_or_channel_is_rejected(change):
  trained = _world_model()
  playing = _world_model(**change)
  before = playing.dynamics.weight.detach().clone()
  with pytest.raises(ValueError, match="layout or nominal physics"):
    playing.load_bundle(trained.bundle())
  torch.testing.assert_close(playing.dynamics.weight, before)


def test_checkpoint_without_context_schema_requires_retraining():
  wm = _world_model()
  bundle = wm.bundle()
  del bundle["sim_context_schema"]
  with pytest.raises(ValueError, match="Train a new world model"):
    wm.load_bundle(bundle)
