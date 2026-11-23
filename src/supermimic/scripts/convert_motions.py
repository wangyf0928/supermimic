# /// script
# requires-python = ">=3.10"
# dependencies = ["joblib>=1.3", "numpy>=1.24", "scipy>=1.10"]
# ///
"""Convert Westlake O1 reset trajectories to SuperMimic's G1 motion NPZ format.

Run independently of the simulator environment:
  uv run --script src/supermimic/scripts/convert_motions.py \
    --input motions --output motions_npz

Inputs are trusted Joblib/pickle dictionaries for ``westlake.o1_29dof``.
The O1 export uses 50 Hz, world-frame root link states, wxyz quaternions,
and MuJoCo depth-first joint order. The corresponding G1 hinges have the
same order and axis signs; only knee/elbow names differ. This converter
preserves the root trajectory and joint angles, without changing robot
geometry or correcting ground contact. G1 body states are computed by the
training loader's forward kinematics, not copied from the O1 skeleton.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Column order in mjlab's G1 model and csv_to_npz.JOINT_NAMES. Keep the
# standalone converter independent of mjlab's simulator/GPU dependencies.
G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
O1_JOINT_NAMES = tuple(
  name.replace("_knee_joint", "_knee_pitch_joint").replace(
    "_elbow_joint", "_elbow_pitch_joint"
  )
  for name in G1_JOINT_NAMES
)


def _array(data: Mapping, name: str) -> np.ndarray:
  if name not in data:
    raise ValueError(f"Missing required field: {name}")
  array = np.asarray(data[name])
  if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
    raise ValueError(f"{name} must contain finite real numbers")
  return array.astype(np.float64)


def _joint_order(data: Mapping) -> list[int]:
  raw_names = data.get("joint_names", O1_JOINT_NAMES)
  names = np.asarray(raw_names)
  if names.shape != (29,) or names.dtype.kind not in "US":
    raise ValueError("joint_names must contain 29 strings in source column order")
  mapped = [
    name.replace("_knee_pitch_joint", "_knee_joint").replace(
      "_elbow_pitch_joint", "_elbow_joint"
    )
    for name in names.astype(str).tolist()
  ]
  if len(set(mapped)) != 29 or set(mapped) != set(G1_JOINT_NAMES):
    raise ValueError("joint_names must map uniquely to all 29 G1 joints")
  return [mapped.index(name) for name in G1_JOINT_NAMES]


def _continuous_quaternions(quat: np.ndarray) -> np.ndarray:
  with np.errstate(over="ignore"):
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
  if not np.isfinite(norms).all():
    raise ValueError("reset_body_rot root quaternion norm is not finite")
  if np.any(norms < 1e-8):
    raise ValueError("reset_body_rot contains a zero root quaternion")
  quat = quat / norms
  # q and -q describe the same rotation; choose a continuous representative.
  flips = np.sum(quat[1:] * quat[:-1], axis=-1) < 0
  signs = np.concatenate(([1], np.cumprod(np.where(flips, -1, 1))))
  return quat * signs[:, None]


def _velocities(arrays: dict[str, np.ndarray], fps: float) -> None:
  dt = 1.0 / fps
  for pos, vel in (("root_pos_w", "root_lin_vel_w"), ("joint_pos", "joint_vel")):
    arrays[vel] = np.gradient(arrays[pos], dt, axis=0)
  rotation = Rotation.from_quat(arrays["root_quat_w"][:, [1, 2, 3, 0]])
  # Left relative rotations give angular velocity in the WORLD frame.
  omega = np.empty((len(rotation), 3), dtype=np.float64)
  omega[0] = (rotation[1] * rotation[0].inv()).as_rotvec() / dt
  omega[-1] = (rotation[-1] * rotation[-2].inv()).as_rotvec() / dt
  if len(rotation) > 2:
    omega[1:-1] = (rotation[2:] * rotation[:-2].inv()).as_rotvec() / (2 * dt)
  arrays["root_ang_vel_w"] = omega


def convert_motion(
  data: Mapping,
  *,
  input_fps: float = 50.0,
  output_fps: float = 50.0,
  recompute_velocities: bool = False,
) -> dict[str, np.ndarray]:
  """Convert one motion; preserve exported velocities unless resampling/requested.

  All source arrays share T frames. ``reset_body_*`` have shape (T, 30, C),
  with the root link in slot zero; ``reset_joint_*`` have shape (T, 29).
  Optional ``joint_names`` override the known O1 export column order.
  """
  if not isinstance(data, Mapping):
    raise ValueError("Expected a motion dictionary")
  if data.get("robot") != "westlake.o1_29dof":
    raise ValueError(f"Unsupported robot: {data.get('robot')!r}")
  for label, fps in (("input_fps", input_fps), ("output_fps", output_fps)):
    if not math.isfinite(fps) or fps <= 0:
      raise ValueError(f"{label} must be finite and positive")
    bounds = np.finfo(np.float32)
    if fps < float(bounds.smallest_subnormal) or fps > float(bounds.max):
      raise ValueError(f"{label} must be representable as a positive finite float32")

  joint_pos = _array(data, "reset_joint_pos")
  if joint_pos.ndim != 2 or joint_pos.shape[1] != 29 or len(joint_pos) < 2:
    raise ValueError("reset_joint_pos must have shape (T, 29) with at least 2 frames")
  frame_count = len(joint_pos)
  joint_vel = _array(data, "reset_joint_vel")
  if joint_vel.shape != joint_pos.shape:
    raise ValueError("reset_joint_vel must have the same shape as reset_joint_pos")
  order = _joint_order(data)
  arrays = {"joint_pos": joint_pos[:, order], "joint_vel": joint_vel[:, order]}
  for source, target, width in (
    ("reset_body_pos", "root_pos_w", 3),
    ("reset_body_rot", "root_quat_w", 4),
    ("reset_body_lin_vel", "root_lin_vel_w", 3),
    ("reset_body_ang_vel", "root_ang_vel_w", 3),
  ):
    values = _array(data, source)
    if values.shape != (frame_count, 30, width):
      raise ValueError(f"{source} must have shape ({frame_count}, 30, {width})")
    arrays[target] = values[:, 0, :].copy()
  arrays["root_quat_w"] = _continuous_quaternions(arrays["root_quat_w"])

  if input_fps != output_fps:
    duration = (frame_count - 1) / input_fps
    count = math.floor(duration * output_fps + 1e-6) + 1
    if count < 2:
      raise ValueError("Output frame rate leaves fewer than 2 frames")
    source_times = np.arange(frame_count) / input_fps
    times = np.minimum(np.arange(count) / output_fps, duration)
    for field in ("root_pos_w", "joint_pos"):
      arrays[field] = np.stack(
        [np.interp(times, source_times, col) for col in arrays[field].T], axis=-1
      )
    rotation = Rotation.from_quat(arrays["root_quat_w"][:, [1, 2, 3, 0]])
    quat = Slerp(source_times, rotation)(times).as_quat()[:, [3, 0, 1, 2]]
    arrays["root_quat_w"] = _continuous_quaternions(quat)
    recompute_velocities = True
  if recompute_velocities:
    _velocities(arrays, output_fps)

  for field, array in arrays.items():
    if not np.isfinite(array).all() or np.any(abs(array) > np.finfo(np.float32).max):
      raise ValueError(f"{field} cannot be represented as finite float32")
  return {
    **{field: array.astype(np.float32) for field, array in arrays.items()},
    "joint_names": np.asarray(G1_JOINT_NAMES),
    "fps": np.array([output_fps], dtype=np.float32),
  }


def _input_files(source: str) -> tuple[list[Path], Path | None]:
  expanded = os.path.expanduser(source)
  path = Path(expanded)
  if path.is_dir():
    return sorted(path.rglob("*.pkl")), path
  if path.is_file():
    return [path], None
  files = sorted(
    Path(p) for p in glob.glob(expanded, recursive=True) if Path(p).is_file()
  )
  return files, None


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
      temporary = Path(f.name)
      np.savez_compressed(f, **arrays)
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input", default="motions", help="PKL file, directory, or quoted glob"
  )
  parser.add_argument("--output", type=Path, default=Path("motions_npz"))
  parser.add_argument("--input-fps", type=float, default=50.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--recompute-velocities", action="store_true")
  parser.add_argument(
    "--overwrite", action="store_true", help="Replace existing outputs"
  )
  args = parser.parse_args(argv)
  try:
    files, base = _input_files(args.input)
    if not files:
      raise ValueError(f"No motion files matched {args.input!r}")
    output = args.output.expanduser()
    relative_paths = [
      p.relative_to(base) if base is not None else Path(p.name) for p in files
    ]
    destinations = [output / p.with_suffix(".npz") for p in relative_paths]
    if len({p.resolve() for p in destinations}) != len(destinations):
      raise ValueError(
        "Input filenames collide in the output directory; use a directory input"
      )
    source_paths = {p.resolve() for p in files}
    report_path = output / "conversion_report.json"
    for path in [*destinations, report_path]:
      if path.resolve() in source_paths:
        raise ValueError(f"Output would overwrite an input: {path}")
      if path.exists() and not args.overwrite:
        raise FileExistsError(
          f"Output already exists: {path}; use --overwrite to replace it"
        )

    records = []
    for i, (source, destination) in enumerate(zip(files, destinations, strict=True), 1):
      try:
        data = joblib.load(source)
        arrays = convert_motion(
          data,
          input_fps=args.input_fps,
          output_fps=args.output_fps,
          recompute_velocities=args.recompute_velocities,
        )
        _save_npz(destination, arrays)
      except Exception as exc:
        raise ValueError(f"{source}: {exc}") from exc
      records.append(
        {
          "source": str(source),
          "output": str(destination),
          "input_frames": int(len(data["reset_joint_pos"])),
          "output_frames": int(len(arrays["joint_pos"])),
        }
      )
      if i == 1 or i % 25 == 0 or i == len(files):
        print(f"[{i}/{len(files)}] {source.name} -> {destination}")
    total_frames = sum(r["output_frames"] for r in records)
    recomputed = args.recompute_velocities or args.input_fps != args.output_fps
    report = {
      "source_robot": "westlake.o1_29dof",
      "target_robot": "unitree_g1",
      "input_fps": args.input_fps,
      "output_fps": args.output_fps,
      "velocities": "recomputed" if recomputed else "preserved",
      "joint_names": list(G1_JOINT_NAMES),
      "motion_count": len(records),
      "total_frames": total_frames,
      "total_duration_seconds": (total_frames - len(records)) / args.output_fps,
      "motions": records,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
      f"Converted {len(files)} motions, {total_frames} frames "
      f"at {args.output_fps:g} Hz."
    )
    print(f"Report: {report_path}")
    return 0
  except (ValueError, OSError) as exc:
    print(f"Conversion failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
