# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.3", "numpy>=1.24"]
# ///
"""Play converted SuperMimic NPZ motions on the G1 model in MuJoCo.

  uv run --script src/supermimic/scripts/motion_vizz.py
  python src/supermimic/scripts/motion_vizz.py --motion-file motions_npz --speed 0.5

Space: pause/resume; R: restart; Left/Right: step and pause.
N/P: next clip; O: previous clip; +/-: speed; B: G1 skeleton; F: camera follow.
Playback sets the recorded poses and calls mj_forward, without physics stepping.
Based on the playback controls in tmp/motion_vizz*.py.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
G1_XML_RELATIVE = Path("asset_zoo/robots/unitree_g1/xmls/g1.xml")
CONTROLS = (
  "Space: pause | R: restart | Left/Right: step | N/P: next | O: previous | "
  "+/-: speed | B: G1 skeleton | F: follow"
)


def default_xml_path() -> Path:
  """Find mjlab's actual G1 model without importing the training environment."""
  candidate = REPO_ROOT.parent / "mjlab_research/src/mjlab" / G1_XML_RELATIVE
  if candidate.is_file():
    return candidate
  spec = importlib.util.find_spec("mjlab")
  if spec is not None:
    for directory in spec.submodule_search_locations or []:
      candidate = Path(directory) / G1_XML_RELATIVE
      if candidate.is_file():
        return candidate
  raise FileNotFoundError("Cannot find mjlab's G1 XML; pass --xml /path/to/g1.xml")


def resolve_motion_files(source: str) -> list[Path]:
  path = Path(source).expanduser()
  if path.is_dir():
    files = sorted(path.rglob("*.npz"))
  elif path.is_file():
    files = [path]
  else:
    files = sorted(
      Path(p)
      for p in glob.glob(os.path.expanduser(source), recursive=True)
      if Path(p).is_file() and Path(p).suffix == ".npz"
    )
  if not files:
    raise ValueError(f"No NPZ motions found: {source}")
  return [p.resolve() for p in files]


@dataclass
class MotionClip:
  path: Path
  fps: float
  joint_names: tuple[str, ...]
  joint_pos: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray

  @property
  def num_frames(self) -> int:
    return len(self.joint_pos)


def load_motion(path: Path, fps: float | None = None) -> MotionClip:
  path = Path(path).expanduser().resolve()
  required = ("root_pos_w", "root_quat_w", "joint_pos", "joint_names")
  with np.load(path, allow_pickle=False) as archive:
    missing = set(required) - set(archive.files)
    if missing:
      raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
    arrays = {key: archive[key] for key in required}
    rate = np.asarray(fps if fps is not None else archive.get("fps", 50.0))
  if rate.size != 1 or rate.dtype.kind not in "fiu":
    raise ValueError("fps must contain one finite, positive number")
  rate = float(rate.item())
  if not math.isfinite(rate) or rate <= 0:
    raise ValueError("fps must contain one finite, positive number")

  names = arrays["joint_names"]
  if names.shape != (29,) or names.dtype.kind not in "US":
    raise ValueError("joint_names must contain 29 strings in column order")
  names = tuple(names.astype(str).tolist())
  if len(set(names)) != 29 or any(not name for name in names):
    raise ValueError("joint_names must be unique and nonempty")
  joints = arrays["joint_pos"]
  if joints.ndim != 2 or joints.shape[1] != 29 or len(joints) == 0:
    raise ValueError("joint_pos must have shape (T, 29) with at least one frame")
  for field, shape in (
    ("joint_pos", (len(joints), 29)),
    ("root_pos_w", (len(joints), 3)),
    ("root_quat_w", (len(joints), 4)),
  ):
    values = arrays[field]
    if (
      values.shape != shape
      or values.dtype.kind not in "fiu"
      or not np.isfinite(values).all()
    ):
      raise ValueError(f"{field} must have shape {shape} and finite real values")
    arrays[field] = values.astype(np.float64)
  with np.errstate(over="ignore"):
    norms = np.linalg.norm(arrays["root_quat_w"], axis=-1, keepdims=True)
  if not np.isfinite(norms).all() or np.any(norms < 1e-8):
    raise ValueError("Root quaternion norm must be finite and nonzero")
  return MotionClip(
    path=path,
    fps=rate,
    joint_names=names,
    joint_pos=arrays["joint_pos"],
    root_pos_w=arrays["root_pos_w"],
    root_quat_w=arrays["root_quat_w"] / norms,
  )


def load_model(xml_path: Path) -> mujoco.MjModel:
  """Keep asset paths intact and add the preview floor only in memory."""
  spec = mujoco.MjSpec.from_file(str(Path(xml_path).expanduser().resolve()))
  spec.worldbody.add_geom(
    name="motion_preview_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.05],
    rgba=[0.22, 0.25, 0.28, 1],
    group=0,
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
  return spec.compile()


class RobotVisualizer:
  def __init__(
    self,
    files: list[Path],
    xml_path: Path | None = None,
    *,
    speed: float = 1.0,
    show_bodies: bool = False,
    paused: bool = False,
    follow: bool = True,
    fps: float | None = None,
  ):
    if not files:
      raise ValueError("No motions to play")
    if not math.isfinite(speed) or speed <= 0:
      raise ValueError("Playback speed must be finite and positive")
    self.files = files
    self.speed = speed
    self.show_bodies = show_bodies
    self.paused = paused
    self.follow = follow
    self.fps_override = fps
    self.key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    self.model = load_model(xml_path or default_xml_path())
    self.data = mujoco.MjData(self.model)
    free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or self.model.njnt != 30:
      raise ValueError("Expected a G1 model with one free root and 29 joints")
    self.root_qpos_adr = int(self.model.jnt_qposadr[free[0]])
    self.root_body_id = int(self.model.jnt_bodyid[free[0]])
    self._joint_ids = [j for j in range(self.model.njnt) if j != free[0]]
    if any(
      self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in self._joint_ids
    ):
      raise ValueError("Expected 29 hinge joints")
    self.select_motion(0)

  def select_motion(self, index: int) -> None:
    index %= len(self.files)
    motion = load_motion(self.files[index], fps=self.fps_override)
    model_names = {self.model.joint(j).name for j in self._joint_ids}
    if set(motion.joint_names) != model_names:
      missing = sorted(model_names - set(motion.joint_names))
      extra = sorted(set(motion.joint_names) - model_names)
      raise ValueError(
        f"{motion.path.name}: joint mismatch; missing={missing}, extra={extra}"
      )
    # File columns need not be in MuJoCo order: map names to qpos addresses.
    self.joint_qpos_adr = np.array(
      [int(self.model.joint(name).qposadr[0]) for name in motion.joint_names]
    )
    self.motion = motion
    self.motion_index = index
    self.current_frame = 0
    self._frame_fraction = 0.0
    self._recenter = True
    self.model.opt.timestep = 1.0 / motion.fps
    self.update_pose()

  def update_pose(self) -> None:
    frame = self.current_frame
    root = self.root_qpos_adr
    self.data.qpos[:] = self.model.qpos0
    self.data.qpos[root : root + 3] = self.motion.root_pos_w[frame]
    self.data.qpos[root + 3 : root + 7] = self.motion.root_quat_w[frame]
    self.data.qpos[self.joint_qpos_adr] = self.motion.joint_pos[frame]
    self.data.qvel[:] = 0
    self.data.ctrl[:] = 0
    self.data.qfrc_applied[:] = 0
    self.data.xfrc_applied[:] = 0
    self.data.time = frame / self.motion.fps
    mujoco.mj_forward(self.model, self.data)

  def advance(self, elapsed: float) -> None:
    if not self.paused:
      self._frame_fraction += elapsed * self.motion.fps * self.speed
      steps = int(self._frame_fraction)
      self.current_frame = (self.current_frame + steps) % self.motion.num_frames
      self._frame_fraction -= steps

  def key_callback(self, keycode: int) -> None:
    # Viewer callbacks run on another thread; only the main thread changes state.
    self.key_events.put(keycode)

  def status(self) -> str:
    state = "Paused" if self.paused else "Playing"
    return (
      f"[{self.motion_index + 1}/{len(self.files)}] {self.motion.path.name} | "
      f"frame {self.current_frame + 1}/{self.motion.num_frames} | "
      f"{self.motion.fps:g} Hz | {self.speed:g}x | {state}"
    )

  def handle_keys(self) -> bool:
    changed = False
    while True:
      try:
        key = self.key_events.get_nowait()
      except queue.Empty:
        break
      if key == ord(" "):
        self.paused = not self.paused
      elif key == ord("R"):
        self.current_frame = 0
      elif key in (263, 262):  # GLFW left/right
        self.paused = True
        step = -1 if key == 263 else 1
        self.current_frame = (self.current_frame + step) % self.motion.num_frames
      elif key in (ord("N"), ord("P"), 267):  # Page Down
        self.select_motion(self.motion_index + 1)
      elif key in (ord("O"), 266):  # Page Up
        self.select_motion(self.motion_index - 1)
      elif key in (ord("+"), ord("="), 334):
        self.speed = min(self.speed * 2, 16.0)
      elif key in (ord("-"), 333):
        self.speed = max(self.speed / 2, 0.0625)
      elif key == ord("B"):
        self.show_bodies = not self.show_bodies
        continue
      elif key == ord("F"):
        self.follow = not self.follow
        continue
      else:
        continue
      changed = True
      self._frame_fraction = 0.0
      print(self.status())
    return changed

  def update_viewer(self, viewer) -> None:
    self.update_pose()
    if self.follow or self._recenter:
      viewer.cam.lookat[:] = self.data.xpos[self.root_body_id]
      self._recenter = False
    scene = viewer.user_scn
    scene.ngeom = 0
    if not self.show_bodies:
      return
    color = np.array([1.0, 0.25, 0.1, 0.85], dtype=np.float32)
    for body in range(1, self.model.nbody):
      if scene.ngeom >= scene.maxgeom:
        break
      position = self.data.xpos[body]
      mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.015, 0, 0]),
        position,
        np.eye(3).ravel(),
        color,
      )
      scene.ngeom += 1
      parent = self.model.body_parentid[body]
      if parent > 0 and scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
          geom,
          mujoco.mjtGeom.mjGEOM_CAPSULE,
          np.zeros(3),
          np.zeros(3),
          np.eye(3).ravel(),
          color,
        )
        mujoco.mjv_connector(
          geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, self.data.xpos[parent], position
        )
        scene.ngeom += 1

  def run(self) -> None:
    import mujoco.viewer

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
      raise RuntimeError(
        "GUI playback needs a desktop DISPLAY; use --headless for FK checks"
      )
    print(CONTROLS)
    print(self.status())
    with mujoco.viewer.launch_passive(
      self.model,
      self.data,
      key_callback=self.key_callback,
      show_left_ui=False,
      show_right_ui=False,
    ) as viewer:
      with viewer.lock():
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.opt.geomgroup[:] = 1
        viewer.opt.geomgroup[3] = 0  # mjlab G1: visual=2, collision=3, floor=0.
        self.update_viewer(viewer)
      viewer.sync()
      previous = time.perf_counter()
      while viewer.is_running():
        start = time.perf_counter()
        elapsed, previous = start - previous, start
        with viewer.lock():
          changed = self.handle_keys()
          if not changed:
            self.advance(elapsed)
          self.update_viewer(viewer)
        if hasattr(viewer, "set_texts"):
          state = "Paused" if self.paused else "Playing"
          text = (
            f"{self.motion.path.name}\n"
            f"Clip {self.motion_index + 1}/{len(self.files)} | "
            f"Frame {self.current_frame + 1}/{self.motion.num_frames}\n"
            f"{self.motion.fps:g} Hz | {self.speed:g}x | {state}\n"
            "Space: pause | R: restart | Left/Right: step\n"
            "N/P: next | O: previous | +/-: speed\n"
            "B: G1 skeleton | F: camera follow"
          )
          viewer.set_texts(
            (
              mujoco.mjtFontScale.mjFONTSCALE_100,
              mujoco.mjtGridPos.mjGRID_TOPLEFT,
              text,
              "",
            )
          )
        viewer.sync()
        if changed:
          # Do not count file I/O during a clip switch as playback time.
          previous = time.perf_counter()
        time.sleep(max(0.0, 1.0 / 120 - (time.perf_counter() - start)))

  def check_all_frames(self) -> None:
    total = 0
    for index in range(len(self.files)):
      self.select_motion(index)
      for frame in range(self.motion.num_frames):
        self.current_frame = frame
        self.update_pose()
        if (
          not np.isfinite(self.data.xpos).all()
          or not np.isfinite(self.data.xquat).all()
        ):
          raise ValueError(f"{self.motion.path}: non-finite FK at frame {frame}")
      total += self.motion.num_frames
      if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(self.files):
        print(f"Checked {index + 1}/{len(self.files)}: {self.motion.path.name}")
    print(f"Passed MuJoCo FK: {len(self.files)} motions, {total} frames.")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion-file",
    "--input",
    default=str(REPO_ROOT / "motions_npz"),
    help="NPZ file, directory, or quoted glob (default: repository motions_npz)",
  )
  parser.add_argument("--xml", type=Path, help="Override the mjlab G1 XML path")
  parser.add_argument("--fps", type=float, help="Override file fps (fallback: 50)")
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--paused", action="store_true")
  parser.add_argument(
    "--show-bodies", action="store_true", help="Overlay G1 FK skeleton"
  )
  parser.add_argument("--no-follow", action="store_true")
  parser.add_argument(
    "--headless", action="store_true", help="Check all frames without a window"
  )
  args = parser.parse_args(argv)
  try:
    files = resolve_motion_files(args.motion_file)
    player = RobotVisualizer(
      files,
      args.xml,
      speed=args.speed,
      show_bodies=args.show_bodies,
      paused=args.paused,
      follow=not args.no_follow,
      fps=args.fps,
    )
    if args.headless:
      player.check_all_frames()
    else:
      player.run()
    return 0
  except (ValueError, OSError, RuntimeError) as exc:
    print(f"Motion preview failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
