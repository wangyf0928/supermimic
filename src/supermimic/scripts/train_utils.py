"""Shared utilities for training and playback scripts."""

from __future__ import annotations

import glob
import os

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from supermimic.utils.wandb_artifacts import download_artifact_cached


def resolve_motion_source(source: str) -> str | list[str]:
  """Resolve local motion paths and comma-separated globs for MotionLoader."""
  paths = [os.path.expanduser(part.strip()) for part in source.split(",")]
  if len(paths) == 1 and "**" not in paths[0]:
    # MotionLoader handles a single file, packed directory, or ordinary glob.
    return paths[0]

  files: list[str] = []
  for path in paths:
    if glob.has_magic(path):
      matches = sorted(set(glob.glob(path, recursive=True)))
      if not matches:
        raise ValueError(f"No motion files matched: {path}")
      files.extend(matches)
    else:
      files.append(path)
  return files


def download_registry_motions(registry_name: str | None) -> str | list[str] | None:
  """Download motion files from W&B registry artifacts.

  Supports comma-separated registry names for multi-motion.
  Returns a single path string if one motion, a list if multiple, or None.
  """
  if not registry_name:
    return None

  names = [n.strip() for n in registry_name.split(",")]
  api = wandb.Api()

  paths = []
  for name in names:
    if ":" not in name:
      name = name + ":latest"
    artifact = api.artifact(name)
    paths.append(str(download_artifact_cached(artifact) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(
  env_cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str
) -> RslRlVecEnvWrapper:
  """Set ``env_cfg.scene.num_envs`` and build a wrapped RL environment."""
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
