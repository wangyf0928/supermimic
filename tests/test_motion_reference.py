"""Reference-root corrections preserve every non-root world-space target."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

from supermimic.representations import motion_anchor_window


@pytest.fixture(scope="module")
def fk():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  return ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.from_numpy(model.jnt_qposadr[1:]),
    torch.tensor([0, 6, 12, 15, 23]),
  )


@pytest.fixture
def motion(tmp_path, fk):
  kinematics = fk
  generator = np.random.default_rng(71)
  frames = 8
  root_pos = generator.normal(size=(frames, 3)).astype(np.float32) * 0.05
  root_pos[:, 2] += 0.7
  # Tilted roots distinguish a world-Z adjustment from a body-local offset.
  quat = np.array([0.89, 0.2, -0.3, 0.12], dtype=np.float32)
  quat /= np.linalg.norm(quat)
  path = tmp_path / "reference.npz"
  arrays = dict(
    root_pos_w=root_pos,
    root_quat_w=np.repeat(quat[None], frames, axis=0),
    root_lin_vel_w=generator.normal(size=(frames, 3)).astype(np.float32),
    root_ang_vel_w=generator.normal(size=(frames, 3)).astype(np.float32),
    joint_pos=(generator.normal(size=(frames, 29)) * 0.1).astype(np.float32),
    joint_vel=generator.normal(size=(frames, 29)).astype(np.float32),
    joint_names=np.asarray(kinematics.joint_names),
    fps=np.array([50.0]),
  )
  np.savez(path, **arrays)
  raw = SimpleNamespace(
    **{
      key: torch.from_numpy(value.copy())
      for key, value in arrays.items()
      if key not in {"joint_names", "fps"}
    },
    motion_ends=torch.tensor([frames]),
  )
  return SimpleNamespace(
    loaded=MotionLoader(str(path), kinematics),
    raw=raw,
    fk=kinematics,
    path=path,
    bytes=path.read_bytes(),
  )


def assert_root_only(motion, frames):
  raw = motion.raw
  before = motion.fk(
    raw.root_pos_w[frames],
    raw.root_quat_w[frames],
    raw.joint_pos[frames],
    raw.root_lin_vel_w[frames],
    raw.root_ang_vel_w[frames],
    raw.joint_vel[frames],
  )
  after = motion.loaded.body_state(frames)
  expected_pos = before[0].clone()
  expected_pos[..., 0, 2] += 0.1
  torch.testing.assert_close(after[0], expected_pos, rtol=0, atol=0)
  for old, new in zip(before[1:], after[1:], strict=True):
    torch.testing.assert_close(new, old, rtol=0, atol=0)
  assert torch.equal(after[0][..., 1:, :], before[0][..., 1:, :])
  assert torch.equal(after[0][..., 0, :2], before[0][..., 0, :2])
  torch.testing.assert_close(
    after[0][..., 0, :], motion.loaded.root_pos_w[frames], rtol=0, atol=0
  )


@pytest.mark.parametrize(
  "frames", [torch.tensor([0, 3, 7]), torch.tensor([[0, 3], [4, 7]])]
)
def test_only_root_world_height_changes_with_tilted_pose(motion, frames):
  loaded = motion.loaded
  assert_root_only(motion, frames)
  expected_root = motion.raw.root_pos_w.clone()
  expected_root[:, 2] += 0.1
  assert torch.equal(loaded.root_pos_w, expected_root)
  assert loaded.fk is motion.fk
  for field in (
    "joint_pos",
    "joint_vel",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
  ):
    assert torch.equal(getattr(loaded, field), getattr(motion.raw, field))
  assert motion.path.read_bytes() == motion.bytes


def test_anchor_goals_follow_corrected_root_without_changing_physical_fk(motion):
  raw, loaded = motion.raw, motion.loaded
  times = torch.tensor([0, 6, 7])
  original_pos, original_quat = motion_anchor_window(raw, times)
  positions, quats = motion_anchor_window(loaded, times)
  expected = original_pos.clone()
  expected[..., 2] += 0.1
  assert torch.equal(positions, expected)
  assert torch.equal(quats, original_quat)

  # Physical G1 FK still translates every link when its actual root is moved.
  # Correcting reference targets must not replace or patch that function.
  physical = loaded.fk(
    loaded.root_pos_w[times],
    raw.root_quat_w[times],
    raw.joint_pos[times],
    raw.root_lin_vel_w[times],
    raw.root_ang_vel_w[times],
    raw.joint_vel[times],
  )[0]
  reference = loaded.body_state(times)[0]
  torch.testing.assert_close(
    physical[:, 1:, 2] - reference[:, 1:, 2],
    torch.full((len(times), 4), 0.1),
    rtol=0,
    atol=1e-7,
  )


def test_reference_body_state_is_compileable_for_dynamic_frame_batches(motion):
  loaded = motion.loaded
  compiled = torch.compile(
    loaded.body_state, backend="eager", fullgraph=True, dynamic=True
  )
  for frames in (torch.tensor([0, 2]), torch.tensor([1, 3, 4, 7])):
    for actual, expected in zip(
      compiled(frames), loaded.body_state(frames), strict=True
    ):
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_repeated_loading_and_sampling_never_accumulate_or_modify_file(motion):
  frames = torch.tensor([0, 7])
  expected = motion.loaded.body_state(frames)
  for _ in range(2):
    loaded = MotionLoader(str(motion.path), motion.fk)
    torch.testing.assert_close(
      loaded.root_pos_w, motion.loaded.root_pos_w, rtol=0, atol=0
    )
    for _ in range(2):
      torch.testing.assert_close(
        loaded.body_state(frames)[0], expected[0], rtol=0, atol=0
      )
  assert motion.path.read_bytes() == motion.bytes
