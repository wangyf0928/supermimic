"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import faulthandler
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
import torch.distributed as dist
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

import supermimic.tasks  # noqa: F401
from supermimic.distributed import is_main_process, world_size
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  resolve_motion_source,
  setup_device,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 4096
  """Number of simulation environments on each GPU."""
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  """W&B motion artifact to download when --motion-file is omitted."""
  motion_file: str | None = None
  """Local NPZ file, packed directory, or quoted glob of motion clips."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


@contextmanager
def training_device(requested: str | None) -> Iterator[str]:
  """Bind each torchrun worker to its local CUDA device for the whole run."""
  if int(os.environ.get("WORLD_SIZE", "1")) == 1:
    yield setup_device(requested)
    return

  if requested not in (None, "cuda"):
    raise ValueError(
      "With torchrun, omit --device and select GPUs via CUDA_VISIBLE_DEVICES"
    )
  local_rank = int(os.environ["LOCAL_RANK"])
  if not 0 <= local_rank < torch.cuda.device_count():
    raise ValueError(
      f"LOCAL_RANK={local_rank} has no visible GPU. Check --nproc-per-node and CUDA_VISIBLE_DEVICES."
    )
  torch.cuda.set_device(local_rank)
  device = setup_device(f"cuda:{local_rank}")
  dist.init_process_group(backend="nccl", device_id=local_rank)
  try:
    yield device
  finally:
    dist.destroy_process_group()


def run_training(chosen_task: str, args: TrainConfig, device: str) -> None:
  if args.runner.version == "v2":
    # Triton fails to compile the fused kinematics kernels in v2 reset and step.
    torch.compiler.set_stance("force_eager")
    print("[INFO] v2 training: torch.compile disabled")

  if world_size() > 1 and args.env.seed is not None and args.env.seed >= 0:
    args.env.seed = (args.env.seed + dist.get_rank()) % 2**32
  print(
    f"[INFO] Loading {chosen_task} on {device} with {args.num_envs} environments "
    f"({args.num_envs * world_size()} total)"
  )

  motion_cfg = cast(MotionCommandCfg, args.env.commands["motion"])
  if args.motion_file:
    motion_cfg.motion_file = resolve_motion_source(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  elif args.registry_name:
    motion_file = _shared_registry_motions(args.registry_name)
    if motion_file is not None:
      motion_cfg.motion_file = motion_file

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  motion_cfg.sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


def _shared_registry_motions(registry_name: str) -> str | list[str] | None:
  """Share the download result or the failure with every local worker."""
  result: list[str | list[str] | None] = [None, None]
  cause = None
  if is_main_process():
    try:
      result[0] = download_registry_motions(registry_name)
    except Exception as exc:
      cause = exc
      result[1] = str(exc)
  if world_size() > 1:
    dist.broadcast_object_list(result, src=0)
  motion_file, error = result
  if error is not None:
    raise RuntimeError(
      f"Could not load W&B motion artifact {registry_name!r}: {error}. "
      "Pass --motion-file with a local NPZ file or quoted glob, "
      "or set --registry-name to an artifact your W&B account can read."
    ) from cause
  return motion_file


def main() -> None:
  faulthandler.enable()
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )
  with training_device(args.device) as device:
    run_training(chosen_task, args, device)


if __name__ == "__main__":
  main()
