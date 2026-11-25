"""Process-group queries shared by training and single-process callers."""

import torch.distributed as dist


def world_size() -> int:
  return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
  return not dist.is_initialized() or dist.get_rank() == 0
