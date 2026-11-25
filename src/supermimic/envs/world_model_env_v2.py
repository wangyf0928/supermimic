"""T3 observations for the existing differentiable motion-tracking environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensordict import TensorDict

from supermimic.envs.t3_observations import T3History, T3Observations, raw_self_frame
from supermimic.envs.world_model_env import WorldModelEnv

if TYPE_CHECKING:
  from supermimic.replay_buffer import ReplayBuffer


class WorldModelEnvV2(WorldModelEnv):
  def __init__(self, *args, observations: T3Observations, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.observations = observations

  def sample_start(self, buffer: ReplayBuffer, batch_size: int) -> TensorDict:
    past = self.observations.past_length
    window, _ = buffer.sample_causal(max(self.history_length, past), batch_size)
    # Causal sampling has already repeated the earliest available in-episode frame.
    recent = window[:, -self.history_length :]
    roots = self.representation.extract_aux_state(recent["state"])
    history = self.observations.build_history(
      raw_self_frame(recent["actor_obs"], roots[..., 3:7]),
      window["motion_time"][:, -past:],
    )
    start = window[:, -1]
    start["self_frames"] = history.self_frames
    start["task_poses"] = history.task_poses
    return start

  def _reset_history(self, batch: TensorDict) -> None:
    self._history = T3History(batch["self_frames"], batch["task_poses"])
    self._observe()

  def _advance_history(self) -> None:
    frame = raw_self_frame(self._synth_sensor(noisy=False), self.aux_state[:, 3:7])
    self._history = self.observations.advance(self._history, frame, self.motion_times)
    self._observe()

  def _observe(self) -> None:
    self._obs = self.observations.observe(
      self._history,
      self.aux_state[:, 3:7],
      self.motion_times,
      noisy=True,
      critic=self._critic_obs(),
    )

  def get_observations(self) -> TensorDict:
    return self._obs
