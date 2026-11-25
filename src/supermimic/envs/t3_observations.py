"""Temporal observation windows shared by the simulator and world model."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import matrix_from_quat, yaw_quat
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader

  from supermimic.t3_model import T3ModelCfg


# EMAN's visible G1 bodies. mjlab calls the knee/elbow pitch links knee/elbow.
MOTION_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def append_frame(
  history: torch.Tensor | None,
  frame: torch.Tensor,
  length: int,
  reset: torch.Tensor | None = None,
) -> torch.Tensor:
  """Append one frame per row. New episodes repeat their first frame as padding."""
  if length < 1:
    raise ValueError("history length must be positive")
  newest = frame.unsqueeze(1)
  if history is None or length == 1:
    # mjlab updates command and sensor buffers in place on the following step.
    return newest.clone().expand(-1, length, *frame.shape[1:])
  shifted = torch.cat([history[:, 1:], newest], dim=1)
  if reset is None:
    return shifted
  reset = reset.bool().reshape(frame.shape[0], *([1] * frame.ndim))
  return torch.where(reset, newest, shifted)


def task_frame_indices(
  motion: MotionLoader, past_times: torch.Tensor, next_length: int
) -> torch.Tensor:
  """Append consecutive reference frames, stopping at the current clip's final frame."""
  if next_length < 0:
    raise ValueError("next_length must be nonnegative")
  if next_length == 0:
    return past_times
  current = past_times[:, -1].contiguous()
  clip = torch.searchsorted(motion.motion_ends, current, right=True)
  last = motion.motion_ends[clip] - 1
  offsets = torch.arange(1, next_length + 1, device=current.device, dtype=current.dtype)
  future = torch.minimum(current[:, None] + offsets, last[:, None])
  return torch.cat([past_times, future], dim=1)


def _tan_norm(quat: torch.Tensor) -> torch.Tensor:
  """World directions of the body's x and z axes, in EMAN's feature order."""
  matrix = matrix_from_quat(quat)
  return torch.cat([matrix[..., :, 0], matrix[..., :, 2]], dim=-1)


def raw_self_frame(proprio: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  """Replace projected gravity with the world-space root tangent and normal."""
  return torch.cat([_tan_norm(root_quat), proprio[..., 3:]], dim=-1)


class T3History(NamedTuple):
  self_frames: torch.Tensor  # (B, H, D), unscaled, before observation noise
  task_poses: torch.Tensor  # (B, P, 16, 7), world position and wxyz quaternion


class T3Observations:
  """EMAN actor windows in the current robot heading, using mjlab's G1 kinematics."""

  task_dim = len(MOTION_BODY_NAMES) * 9

  def __init__(self, env: ManagerBasedRlEnv, cfg: T3ModelCfg):
    # Share reference tensors and the loader's retargeting corrections. Only
    # the link selection changes; rewards still use the command's original FK.
    motion_command = cast("MotionCommand", env.command_manager.get_term("motion"))
    self.motion = copy(motion_command.motion)
    self.self_length = cfg.self_length
    self.next_length = cfg.next_length
    self.past_length = cfg.task_length - cfg.next_length
    if self.self_length < 1 or self.past_length < 1 or self.next_length < 0:
      raise ValueError("self and task histories must include the current frame")
    robot = motion_command.robot
    rows, names = robot.find_bodies(MOTION_BODY_NAMES, preserve_order=True)
    if tuple(names) != MOTION_BODY_NAMES:
      raise ValueError("T3 requires all 16 visible G1 reference bodies")
    device = self.motion.joint_pos.device
    self.motion.fk = ForwardKinematics(
      env.sim.mj_model,
      robot.indexing.body_ids,
      robot.indexing.joint_q_adr,
      torch.tensor(rows),
      device=str(device),
    )
    if device.type == "cuda":
      # Reuse mjlab's batched FK compilation settings for collection and replay.
      self._reference_poses = torch.compile(
        self._reference_poses, dynamic=True, options={"max_fusion_size": 512}
      )
    joints = self.motion.joint_pos.shape[-1]
    self.scale = torch.tensor(
      [1.0] * 6 + [0.1] * 3 + [1.0] * joints + [0.1] * joints + [1.0] * joints,
      device=device,
    )
    self.noise = torch.tensor(
      [0.05] * 6 + [0.2] * 3 + [0.02] * joints + [1.5] * joints + [0.0] * joints,
      device=device,
    )

  def _reference_poses(self, times: torch.Tensor) -> torch.Tensor:
    pos, quat, _, _ = self.motion.body_state(times)
    return torch.cat([pos, quat], dim=-1)

  def build_history(
    self, self_frames: torch.Tensor, past_times: torch.Tensor
  ) -> T3History:
    """Restore the actual causal history at a sampled replay frame."""
    flat_times = past_times.flatten()
    # FK follows the articulated chains. Bound its temporary tensors when
    # warm-starting thousands of 30-frame windows in one batch.
    chunks = [self._reference_poses(chunk) for chunk in flat_times.split(8192)]
    poses = torch.cat(chunks).reshape(*past_times.shape, len(MOTION_BODY_NAMES), 7)
    return T3History(self_frames, poses)

  def advance(
    self,
    history: T3History | None,
    self_frame: torch.Tensor,
    motion_times: torch.Tensor,
    reset: torch.Tensor | None = None,
  ) -> T3History:
    """Append the current self frame and reference pose once per control step."""
    current = self._reference_poses(motion_times)
    return T3History(
      append_frame(
        history.self_frames if history else None, self_frame, self.self_length, reset
      ),
      append_frame(
        history.task_poses if history else None, current, self.past_length, reset
      ),
    )

  def observe(
    self,
    history: T3History,
    root_quat: torch.Tensor,
    motion_times: torch.Tensor,
    noisy: bool,
    critic: torch.Tensor | None = None,
  ) -> TensorDict:
    """Anchor every past/future frame to the current reference pelvis and robot yaw."""
    batch = motion_times.shape[0]
    heading = matrix_from_quat(yaw_quat(root_quat))

    def heading_frame(vectors):
      # Row vectors times the yaw matrix are inverse-rotated into robot heading.
      return torch.bmm(vectors.reshape(batch, -1, 3), heading).reshape_as(vectors)

    self_frames, poses = history
    self_obs = torch.cat(
      [heading_frame(self_frames[..., :6]), self_frames[..., 6:]], dim=-1
    )
    origin = poses[:, -1:, :1, :3]
    if self.next_length:
      times = task_frame_indices(self.motion, motion_times[:, None], self.next_length)
      poses = torch.cat([poses, self._reference_poses(times[:, 1:])], dim=1)
    positions = heading_frame(poses[..., :3] - origin)
    rotations = heading_frame(_tan_norm(poses[..., 3:]))
    if noisy:
      self_obs = self_obs + (2 * torch.rand_like(self_obs) - 1) * self.noise
      rotations = rotations + (2 * torch.rand_like(rotations) - 1) * 0.05
    task = torch.cat([positions.flatten(-2), rotations.flatten(-2)], dim=-1)
    obs = {"self_obs": self_obs * self.scale, "task_obs": task}
    if critic is not None:
      obs["critic"] = critic
    return TensorDict(obs, batch_size=[batch], device=self_frames.device)
