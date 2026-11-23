"""GUI command resets must invalidate WM rollout state and its cached context."""

import pytest

from supermimic.scripts.viewers import SuperMimicViserViewer, ViserPlayViewer


@pytest.mark.parametrize("all_envs", [False, True])
def test_gui_reset_invalidates_world_model_rollout(monkeypatch, all_envs):
  # Exercise the reset hook without opening a browser or starting the simulation.
  viewer = SuperMimicViserViewer.__new__(SuperMimicViserViewer)
  old_aux = object()
  viewer.curr_wm_aux = old_aux
  viewer.step_in_rollout = 5
  viewer.curr_pred_rep_state = object()
  viewer.last_actions = object()
  viewer.wm_vis_qpos = object()
  calls = []

  def base_reset(self, reset_all):
    assert self is viewer
    assert self.curr_wm_aux is old_aux
    calls.append(reset_all)

  monkeypatch.setattr(ViserPlayViewer, "_handle_gui_reset", base_reset)
  viewer._handle_gui_reset(all_envs)

  assert calls == [all_envs]
  assert viewer.step_in_rollout == 0
  # step_and_predict now initializes from the reset state and rereads sim_context.
  assert viewer.curr_wm_aux is None
  assert viewer.curr_pred_rep_state is None
  assert viewer.last_actions is None
  assert viewer.wm_vis_qpos is None
