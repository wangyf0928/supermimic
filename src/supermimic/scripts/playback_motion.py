"""Per-robot motion assignments for parallel playback."""

from bisect import bisect_right

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


class PlaybackMotionCommand(MotionCommand):
  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    ends = self.motion.motion_ends.tolist()
    self._clip_starts = [0, *ends[:-1]]
    # Single-robot playback uses the task's reset behavior until a clip is pinned.
    self.clip_indices = (
      [i % len(ends) for i in range(self.num_envs)] if self.num_envs > 1 else []
    )

  def randomize_clips(self) -> bool:
    """Assign each robot a uniformly sampled clip other than its current one."""
    num_clips = len(self._clip_starts)
    if num_clips < 2:
      return False
    current = torch.searchsorted(self.motion.motion_ends, self.time_steps, right=True)
    draws = torch.randint_like(current, high=num_clips - 1)
    # Skip the current index in each robot's draw from the remaining clips.
    self.clip_indices[:] = (draws + (draws >= current)).tolist()
    return True

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self.clip_indices:
      return super()._resample_command(env_ids)
    # Includes automatic resets after falls and the end of each clip.
    for row, env_id in enumerate(env_ids.tolist()):
      frame = self._clip_starts[self.clip_indices[env_id]]
      self.reset_to_frame(env_ids[row : row + 1], frame)
    self._pending_forward = True

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not super().apply_gui_reset(env_ids):
      return False
    # "Start Here" explicitly assigns the scrubbed motion to the target robots.
    if self.clip_indices:
      frame = int(self.time_steps[env_ids[0]].item())
      clip = bisect_right(self._clip_starts, frame) - 1
      for env_id in env_ids.tolist():
        self.clip_indices[env_id] = clip
    return True


class PlaybackMotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> PlaybackMotionCommand:
    return PlaybackMotionCommand(self, env)
