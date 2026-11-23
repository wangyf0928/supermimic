"""Numerical and file-handling checks for O1 motion conversion.

Run without importing the simulator fixtures in conftest.py:
  PYTHONPATH=src python -m unittest discover -s tests -p test_convert_motions.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

try:
  import joblib
  import numpy as np
  from scipy.spatial.transform import Rotation
except ModuleNotFoundError as exc:
  # The converter has its own script dependencies, separate from the simulator.
  raise unittest.SkipTest("Converter tests require joblib, numpy, and scipy") from exc

from supermimic.scripts.convert_motions import (
  G1_JOINT_NAMES,
  O1_JOINT_NAMES,
  convert_motion,
  main,
)


def make_motion(frames: int = 11, fps: float = 50.0) -> dict:
  """A translating root with a constant world-z spin and tilted local frame."""
  times = np.arange(frames, dtype=np.float64) / fps
  root_velocity = np.array([0.7, -0.2, 0.3])
  root_pos = np.array([1.0, 2.0, 0.8]) + times[:, None] * root_velocity
  root_rotation = Rotation.from_rotvec(
    times[:, None] * np.array([0.0, 0.0, 0.8])
  ) * Rotation.from_euler("x", 0.65)
  root_quat = np.roll(root_rotation.as_quat(), 1, axis=-1)
  slopes = np.linspace(-0.4, 0.6, 29)
  joints = np.linspace(-0.2, 0.2, 29) + times[:, None] * slopes
  body_pos = np.zeros((frames, 30, 3), dtype=np.float64)
  body_pos[:, 0] = root_pos
  body_quat = np.zeros((frames, 30, 4), dtype=np.float64)
  body_quat[..., 0] = 1.0
  body_quat[:, 0] = root_quat
  # Deliberately distinct from pose derivatives: default conversion must preserve
  # the supplied velocities; explicit recomputation must use the poses instead.
  body_lin_vel = np.full((frames, 30, 3), 1.25)
  body_ang_vel = np.full((frames, 30, 3), -2.5)
  joint_vel = np.broadcast_to(np.arange(29) / 10, (frames, 29)).copy()
  return {
    "robot": "westlake.o1_29dof",
    "reset_body_pos": body_pos,
    "reset_body_rot": body_quat,
    "reset_body_lin_vel": body_lin_vel,
    "reset_body_ang_vel": body_ang_vel,
    "reset_joint_pos": joints,
    "reset_joint_vel": joint_vel,
  }


class ConvertMotionTests(unittest.TestCase):
  def test_preserves_reset_state_and_velocities(self):
    data = make_motion()
    data["body_pos"] = np.full((11, 23, 3), 999.0)
    data["omni_body_pos"] = np.full((11, 23, 3), -999.0)
    before = copy.deepcopy(data)
    converted = convert_motion(data)
    source_fields = {
      "root_pos_w": data["reset_body_pos"][:, 0],
      "root_quat_w": data["reset_body_rot"][:, 0],
      "root_lin_vel_w": data["reset_body_lin_vel"][:, 0],
      "root_ang_vel_w": data["reset_body_ang_vel"][:, 0],
      "joint_pos": data["reset_joint_pos"],
      "joint_vel": data["reset_joint_vel"],
    }
    self.assertEqual(set(converted), {*source_fields, "joint_names", "fps"})
    for key, expected in source_fields.items():
      with self.subTest(field=key):
        self.assertEqual(converted[key].dtype, np.float32)
        np.testing.assert_allclose(converted[key], expected, atol=1e-7)
    self.assertEqual(converted["joint_names"].tolist(), list(G1_JOINT_NAMES))
    self.assertTrue(np.issubdtype(converted["joint_names"].dtype, np.str_))
    np.testing.assert_array_equal(converted["fps"], [50.0])
    for key, value in before.items():
      if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(data[key], value)

  def test_maps_named_o1_joints_and_velocities_to_g1_order(self):
    data = make_motion()
    permutation = np.roll(np.arange(29)[::-1], 7)
    expected_pos = data["reset_joint_pos"].copy()
    expected_vel = data["reset_joint_vel"].copy()
    data["joint_names"] = np.asarray(O1_JOINT_NAMES)[permutation]
    data["reset_joint_pos"] = expected_pos[:, permutation]
    data["reset_joint_vel"] = expected_vel[:, permutation]
    converted = convert_motion(data)
    np.testing.assert_allclose(converted["joint_pos"], expected_pos, atol=1e-7)
    np.testing.assert_allclose(converted["joint_vel"], expected_vel, atol=1e-7)
    self.assertIn("left_knee_joint", converted["joint_names"])
    self.assertIn("right_elbow_joint", converted["joint_names"])

  def test_resampling_uses_world_angular_velocity_and_shortest_quat_path(self):
    data = make_motion()
    data["reset_body_rot"][::2, 0] *= -2.0
    converted = convert_motion(data, input_fps=50.0, output_fps=100.0)
    times = np.arange(21) / 100.0
    expected_pos = np.array([1.0, 2.0, 0.8]) + times[:, None] * [0.7, -0.2, 0.3]
    expected_joints = np.linspace(-0.2, 0.2, 29) + times[:, None] * np.linspace(
      -0.4, 0.6, 29
    )
    expected_rotation = Rotation.from_rotvec(
      times[:, None] * [0.0, 0.0, 0.8]
    ) * Rotation.from_euler("x", 0.65)
    actual_rotation = Rotation.from_quat(np.roll(converted["root_quat_w"], -1, -1))
    np.testing.assert_allclose(converted["root_pos_w"], expected_pos, atol=2e-7)
    np.testing.assert_allclose(converted["joint_pos"], expected_joints, atol=2e-7)
    np.testing.assert_allclose(
      (actual_rotation * expected_rotation.inv()).magnitude(), 0.0, atol=1e-6
    )
    self.assert_constant_velocities(converted)
    quats = converted["root_quat_w"]
    np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-7)
    self.assertTrue(np.all(np.sum(quats[1:] * quats[:-1], axis=-1) >= 0))
    np.testing.assert_array_equal(converted["fps"], [100.0])

  def test_explicit_recompute_overrides_stored_velocities(self):
    converted = convert_motion(make_motion(), recompute_velocities=True)
    self.assert_constant_velocities(converted)

  def assert_constant_velocities(self, converted):
    for field, vector in (
      ("root_lin_vel_w", [0.7, -0.2, 0.3]),
      ("root_ang_vel_w", [0.0, 0.0, 0.8]),
      ("joint_vel", np.linspace(-0.4, 0.6, 29)),
    ):
      with self.subTest(velocity=field):
        np.testing.assert_allclose(
          converted[field],
          np.broadcast_to(vector, converted[field].shape),
          atol=2e-6,
        )

  def test_non_grid_endpoint_is_not_extrapolated(self):
    data = make_motion(frames=8, fps=60.0)
    converted = convert_motion(data, input_fps=60.0, output_fps=50.0)
    # Input covers 7/60 s; the last valid 50 Hz frame is 0.10 s.
    self.assertEqual(converted["joint_pos"].shape[0], 6)
    np.testing.assert_allclose(converted["root_pos_w"][-1], [1.07, 1.98, 0.83])

  def test_rejects_invalid_source(self):
    cases = []
    data = make_motion()
    data["robot"] = "unknown.robot"
    cases.append(("robot", data))
    data = make_motion()
    data["reset_joint_pos"] = data["reset_joint_pos"][:, :-1]
    cases.append(("joint width", data))
    data = make_motion()
    data["reset_body_pos"] = data["reset_body_pos"][:-1]
    cases.append(("frame count", data))
    data = make_motion()
    data["reset_body_rot"][3, 0] = 0
    cases.append(("zero quaternion", data))
    data = make_motion()
    data["reset_joint_vel"][2, 7] = np.nan
    cases.append(("nonfinite", data))
    data = make_motion()
    data["joint_names"] = [O1_JOINT_NAMES[0]] * 29
    cases.append(("duplicate names", data))
    data = make_motion()
    data["joint_names"] = list(O1_JOINT_NAMES)
    data["joint_names"][0] = "nonexistent_joint"
    cases.append(("unknown name", data))
    cases.append(("single frame", make_motion(frames=1)))
    for label, data in cases:
      with self.subTest(case=label), self.assertRaises(ValueError):
        convert_motion(data)

  def test_rejects_invalid_frame_rates(self):
    for field in ("input_fps", "output_fps"):
      for value in (0.0, -10.0, float("nan"), float("inf")):
        with self.subTest(field=field, value=value), self.assertRaises(ValueError):
          convert_motion(make_motion(), **{field: value})

  def test_rejects_quaternion_norm_overflow(self):
    data = make_motion()
    data["reset_body_rot"][3, 0] = [1e200, 0.0, 0.0, 0.0]
    # Finite components are insufficient: an overflowing norm must not silently
    # turn the root orientation into an invalid all-zero quaternion.
    with np.errstate(over="ignore"), self.assertRaises(ValueError):
      convert_motion(data)

  def test_rejects_frame_rates_unrepresentable_in_float32(self):
    smallest_positive = float(np.nextafter(np.float32(0), np.float32(1)))
    for fps in (float(np.finfo(np.float32).max) * 2, smallest_positive / 4):
      with self.subTest(fps=fps), self.assertRaises(ValueError):
        # Equal rates isolate writing the fps metadata from resampling. Neither
        # infinity nor zero in the saved array is a usable output frame rate.
        convert_motion(make_motion(), input_fps=fps, output_fps=fps)


class ConversionCliTests(unittest.TestCase):
  def run_cli(self, *args):
    with contextlib.redirect_stdout(io.StringIO()):
      with contextlib.redirect_stderr(io.StringIO()):
        return main(list(args))

  def test_joblib_to_npz_roundtrip_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      inputs = root / "motions"
      inputs.mkdir()
      source = inputs / "walk.pkl"
      joblib.dump(make_motion(), source)
      source_bytes = source.read_bytes()
      output = root / "converted"
      args = ("--input", str(inputs), "--output", str(output))
      self.assertEqual(self.run_cli(*args), 0)
      self.assertEqual(source.read_bytes(), source_bytes)
      destination = output / "walk.npz"
      with np.load(destination, allow_pickle=False) as result:
        self.assertEqual(result["joint_pos"].shape, (11, 29))
        self.assertEqual(result["joint_names"].tolist(), list(G1_JOINT_NAMES))
        np.testing.assert_allclose(result["root_pos_w"][0], [1.0, 2.0, 0.8])
      old_output = destination.read_bytes()
      changed = make_motion()
      changed["reset_body_pos"][:, 0, 0] += 10.0
      joblib.dump(changed, source)
      changed_bytes = source.read_bytes()
      self.assertEqual(self.run_cli(*args), 1)
      self.assertEqual(destination.read_bytes(), old_output)
      self.assertEqual(self.run_cli(*args, "--overwrite"), 0)
      with np.load(destination, allow_pickle=False) as result:
        np.testing.assert_allclose(result["root_pos_w"][0], [11.0, 2.0, 0.8])
      # The converter must never mutate its input, including during overwrite.
      self.assertEqual(source.read_bytes(), changed_bytes)
      np.testing.assert_array_equal(
        joblib.load(source)["reset_body_pos"], changed["reset_body_pos"]
      )

  def test_empty_input_is_an_error(self):
    with tempfile.TemporaryDirectory() as tmp:
      self.assertEqual(
        self.run_cli("--input", str(Path(tmp) / "*.pkl"), "--output", tmp), 1
      )

  def test_recursive_directory_preserves_distinct_same_named_clips(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      inputs = root / "motions"
      for name, root_x in (("left", 4.0), ("right", -3.0)):
        folder = inputs / name
        folder.mkdir(parents=True)
        data = make_motion()
        data["reset_body_pos"][:, 0, 0] = root_x
        joblib.dump(data, folder / "walk.pkl")
      output = root / "converted"
      self.assertEqual(
        self.run_cli("--input", str(inputs), "--output", str(output)), 0
      )
      for name, root_x in (("left", 4.0), ("right", -3.0)):
        with np.load(output / name / "walk.npz", allow_pickle=False) as result:
          np.testing.assert_array_equal(result["root_pos_w"][:, 0], root_x)
      self.assertEqual(len(list(output.rglob("*.npz"))), 2)

  def test_glob_filename_collision_is_rejected_before_writing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      inputs = root / "motions"
      for name in ("left", "right"):
        folder = inputs / name
        folder.mkdir(parents=True)
        joblib.dump(make_motion(), folder / "walk.pkl")
      output = root / "converted"
      self.assertEqual(
        self.run_cli(
          "--input", str(inputs / "*" / "*.pkl"), "--output", str(output)
        ),
        1,
      )
      self.assertEqual(list(output.rglob("*.npz")), [])


if __name__ == "__main__":
  unittest.main()
