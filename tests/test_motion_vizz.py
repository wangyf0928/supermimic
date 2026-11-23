"""Headless data, pose, and playback checks for the MuJoCo NPZ viewer.

Run without importing the simulator fixtures in conftest.py:
  PYTHONPATH=src python -m unittest discover -s tests -p test_motion_vizz.py
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

try:
  import mujoco
  import numpy as np
except ModuleNotFoundError as exc:
  raise unittest.SkipTest("Motion viewer tests require mujoco and numpy") from exc

from supermimic.scripts.motion_vizz import (
  RobotVisualizer,
  default_xml_path,
  load_model,
  load_motion,
  resolve_motion_files,
)


def motion_arrays(names: tuple[str, ...], frames: int = 6) -> dict:
  times = np.arange(frames, dtype=np.float64) / 50.0
  root_pos = np.column_stack((1 + times, 2 - times, 0.8 + 0.1 * times))
  # A scaled wxyz rotation around z detects both order errors and normalization.
  root_quat = np.tile([2 * np.cos(0.3), 0, 0, 2 * np.sin(0.3)], (frames, 1))
  joints = np.linspace(-0.2, 0.2, len(names))[None, :] + times[:, None] * 0.1
  return {
    "root_pos_w": root_pos,
    "root_quat_w": root_quat,
    "joint_pos": joints,
    "joint_names": np.asarray(names),
    "fps": np.array([50.0]),
  }


class MotionDataTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.path = self.root / "walk.npz"
    self.names = tuple(f"joint_{i}" for i in range(29))

  def save(self, arrays):
    np.savez(self.path, **arrays)
    return self.path

  def test_loads_root_wxyz_and_joint_names_without_requiring_velocities(self):
    arrays = motion_arrays(self.names)
    self.save(arrays)
    original = self.path.read_bytes()
    motion = load_motion(self.path)
    self.assertEqual(motion.num_frames, 6)
    self.assertEqual(motion.joint_names, self.names)
    self.assertEqual(motion.fps, 50.0)
    np.testing.assert_array_equal(motion.root_pos_w, arrays["root_pos_w"])
    np.testing.assert_array_equal(motion.joint_pos, arrays["joint_pos"])
    np.testing.assert_allclose(motion.root_quat_w, arrays["root_quat_w"] / 2)
    self.assertEqual(self.path.read_bytes(), original)

  def test_frame_rate_file_default_and_override(self):
    arrays = motion_arrays(self.names)
    arrays["fps"] = np.array([60.0])
    self.save(arrays)
    self.assertEqual(load_motion(self.path).fps, 60.0)
    self.assertEqual(load_motion(self.path, fps=25.0).fps, 25.0)
    arrays.pop("fps")
    self.save(arrays)
    self.assertEqual(load_motion(self.path).fps, 50.0)
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
      with self.subTest(fps=invalid), self.assertRaises(ValueError):
        load_motion(self.path, fps=invalid)

  def test_rejects_invalid_shapes_values_and_duplicate_names(self):
    invalid = []
    arrays = motion_arrays(self.names)
    arrays["joint_pos"] = arrays["joint_pos"][:, :-1]
    invalid.append(("joint width", arrays))
    arrays = motion_arrays(self.names)
    arrays["root_pos_w"] = arrays["root_pos_w"][:-1]
    invalid.append(("root frame count", arrays))
    arrays = motion_arrays(self.names)
    arrays["root_pos_w"][1, 2] = np.nan
    invalid.append(("nonfinite position", arrays))
    arrays = motion_arrays(self.names)
    arrays["root_quat_w"][1] = 0
    invalid.append(("zero quaternion", arrays))
    arrays = motion_arrays(self.names)
    arrays["root_quat_w"][1] = 1e200
    invalid.append(("overflowed quaternion norm", arrays))
    arrays = motion_arrays(self.names)
    arrays["joint_names"] = np.asarray(["duplicate"] * 29)
    invalid.append(("duplicate joint names", arrays))
    arrays = motion_arrays(self.names)
    arrays.pop("root_quat_w")
    invalid.append(("missing root rotation", arrays))
    arrays = motion_arrays(self.names)
    arrays["fps"] = np.array([50.0, 60.0])
    invalid.append(("multiple frame rates", arrays))
    invalid.append(("no frames", motion_arrays(self.names, frames=0)))
    for label, arrays in invalid:
      with self.subTest(case=label), self.assertRaises(ValueError):
        load_motion(self.save(arrays))

  def test_resolves_file_recursive_directory_and_glob(self):
    self.save(motion_arrays(self.names))
    child = self.root / "nested"
    child.mkdir()
    other = child / "other.npz"
    np.savez(other, **motion_arrays(self.names))
    (self.root / "conversion_report.json").write_text("{}")
    self.assertEqual(resolve_motion_files(str(self.path)), [self.path])
    self.assertEqual(
      set(resolve_motion_files(str(self.root))), {self.path, other}
    )
    self.assertEqual(resolve_motion_files(str(self.root / "*.npz")), [self.path])
    with self.assertRaises(ValueError):
      resolve_motion_files(str(self.root / "missing*.npz"))


class MuJoCoPlaybackTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.xml = default_xml_path()
    if not cls.xml.is_file():
      raise unittest.SkipTest("G1 XML is not available")
    cls.model = load_model(cls.xml)
    cls.names = tuple(
      cls.model.joint(i).name
      for i in range(cls.model.njnt)
      if cls.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    if len(cls.names) != 29:
      raise AssertionError("Expected the G1 model's 29 hinge joints")

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.first = self.root / "first.npz"
    self.second = self.root / "second.npz"
    np.savez(self.first, **motion_arrays(self.names, frames=6))
    arrays = motion_arrays(self.names, frames=4)
    arrays["root_pos_w"][:, 0] += 5.0
    arrays["fps"] = np.array([25.0])
    np.savez(self.second, **arrays)

  def visualizer(self, files=None, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
      return RobotVisualizer(files or [self.first, self.second], self.xml, **kwargs)

  def press(self, visualizer, key):
    visualizer.key_callback(ord(key) if isinstance(key, str) else key)
    with contextlib.redirect_stdout(io.StringIO()):
      return visualizer.handle_keys()

  def test_model_adds_floor_without_rewriting_xml(self):
    original = self.xml.read_bytes()
    model = load_model(self.xml)
    floors = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    self.assertGreater(len(floors), 0)
    self.assertTrue(np.any(model.geom_group[floors] == 0))
    self.assertEqual(self.xml.read_bytes(), original)

  def test_joint_names_map_permuted_columns_and_root_pose_correctly(self):
    arrays = motion_arrays(self.names)
    permutation = np.roll(np.arange(29)[::-1], 5)
    expected_joints = arrays["joint_pos"].copy()
    arrays["joint_pos"] = expected_joints[:, permutation]
    arrays["joint_names"] = arrays["joint_names"][permutation]
    np.savez(self.first, **arrays)
    visualizer = self.visualizer([self.first])
    visualizer.current_frame = 3
    visualizer.update_pose()
    model, data = visualizer.model, visualizer.data
    for column, name in enumerate(self.names):
      address = int(model.joint(name).qposadr[0])
      self.assertAlmostEqual(data.qpos[address], expected_joints[3, column])
    free_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    root_address = int(model.jnt_qposadr[free_joint])
    root_body = int(model.jnt_bodyid[free_joint])
    np.testing.assert_allclose(
      data.qpos[root_address : root_address + 3], arrays["root_pos_w"][3]
    )
    np.testing.assert_allclose(data.xpos[root_body], arrays["root_pos_w"][3])
    np.testing.assert_allclose(data.xquat[root_body], arrays["root_quat_w"][3] / 2)
    np.testing.assert_array_equal(data.qvel, 0.0)
    self.assertAlmostEqual(data.time, 3 / 50.0)

  def test_unknown_joint_name_is_rejected_before_playback(self):
    arrays = motion_arrays(self.names)
    names = list(self.names)
    names[0] = "not_a_g1_joint"
    arrays["joint_names"] = np.asarray(names)
    np.savez(self.first, **arrays)
    with self.assertRaises(ValueError):
      self.visualizer([self.first])

  def test_pause_single_step_and_clip_switching(self):
    visualizer = self.visualizer()
    visualizer.key_callback(ord(" "))
    self.assertFalse(visualizer.paused, "Callback must only queue the event")
    with contextlib.redirect_stdout(io.StringIO()):
      self.assertTrue(visualizer.handle_keys())
    self.assertTrue(visualizer.paused)
    self.press(visualizer, 262)
    self.assertEqual(visualizer.current_frame, 1)
    self.press(visualizer, 263)
    self.assertEqual(visualizer.current_frame, 0)
    self.press(visualizer, 263)
    self.assertEqual(visualizer.current_frame, 5)
    self.press(visualizer, "R")
    self.assertEqual(visualizer.current_frame, 0)
    self.press(visualizer, "N")
    self.assertEqual(visualizer.motion_index, 1)
    self.assertEqual(visualizer.current_frame, 0)
    self.assertEqual(visualizer.motion.fps, 25.0)
    self.press(visualizer, "P")
    self.assertEqual(visualizer.motion_index, 0)
    self.press(visualizer, "O")
    self.assertEqual(visualizer.motion_index, 1)
    self.press(visualizer, "B")
    self.assertTrue(visualizer.show_bodies)
    self.press(visualizer, "F")
    self.assertFalse(visualizer.follow)

  def test_fractional_clock_speed_and_pause(self):
    visualizer = self.visualizer()
    for expected in (0, 0, 0, 1):
      visualizer.advance(0.005)
      self.assertEqual(visualizer.current_frame, expected)
    visualizer.paused = True
    visualizer.advance(10.0)
    self.assertEqual(visualizer.current_frame, 1)
    visualizer.paused = False
    self.press(visualizer, "+")
    self.assertGreater(visualizer.speed, 1.0)
    self.press(visualizer, "-")
    self.assertAlmostEqual(visualizer.speed, 1.0)
    visualizer.speed = 2.0
    visualizer.advance(0.02)
    self.assertEqual(visualizer.current_frame, 3)
    with contextlib.redirect_stdout(io.StringIO()):
      visualizer.select_motion(1)
    visualizer.advance(0.02)
    self.assertEqual(visualizer.current_frame, 1)


if __name__ == "__main__":
  unittest.main()
