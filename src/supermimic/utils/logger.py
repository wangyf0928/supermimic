from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import wandb

from supermimic.distributed import is_main_process, world_size

if TYPE_CHECKING:
  from torch.utils.tensorboard import SummaryWriter


class Logger:
  """Aggregate training metrics for W&B, TensorBoard and the console."""

  def __init__(
    self,
    run_name: str | None,
    num_envs: int,
    max_episode_length: int,
    device: str,
  ) -> None:
    self.run_name = run_name
    self.num_envs = num_envs
    self.max_episode_length = max_episode_length
    self.device = device

    self.tot_timesteps = 0
    self.tot_time = 0.0

    # Create buffers
    self.ep_extras = []
    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    self.cur_reward_sum = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )
    self.cur_episode_length = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )

  def process_env_step(
    self,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    extras: dict,
  ) -> None:
    """Add metrics from the environment step to the buffers."""
    if "episode" in extras:
      self.ep_extras.append(extras["episode"])
    elif "log" in extras:
      self.ep_extras.append(extras["log"])

    self.cur_reward_sum += rewards
    self.cur_episode_length += 1

    flush_mask = terminated | (self.cur_episode_length >= self.max_episode_length)
    new_ids = flush_mask.nonzero(as_tuple=False)
    if len(new_ids) > 0:
      self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
      self.lenbuffer.extend(
        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
      )
      self.cur_reward_sum[new_ids] = 0
      self.cur_episode_length[new_ids] = 0

  def _metrics(self, loss_dict: dict[str, float | torch.Tensor]) -> dict[str, float]:
    """Combine scalar sums and counts across ranks, including rank-specific extras."""
    # Transfer the latest learner metrics together. Each rank uses equal batch sizes.
    device_losses = {k: v for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
    device_values = (
      torch.stack(list(device_losses.values())).tolist() if device_losses else []
    )
    losses = loss_dict | dict(zip(device_losses, device_values, strict=True))
    stats = {k: (float(v), 1) for k, v in losses.items()}
    if self.rewbuffer:
      stats["Train/mean_reward"] = (sum(self.rewbuffer), len(self.rewbuffer))
      stats["Train/mean_episode_length"] = (sum(self.lenbuffer), len(self.lenbuffer))

    extras = defaultdict(list)
    for ep_info in self.ep_extras:
      for key, value in ep_info.items():
        key = key if "/" in key else f"Episode/{key}"
        extras[key].append(torch.as_tensor(value, device=self.device).reshape(-1))
    for key, values in extras.items():
      samples = torch.cat(values)
      if samples.numel():
        stats[key] = (samples.sum(dtype=torch.float64).item(), samples.numel())
    self.ep_extras.clear()

    if world_size() == 1:
      return {key: total / count for key, (total, count) in stats.items()}

    # A rank with no completed episodes still participates. Metric names may
    # differ across workers, so a fixed positional tensor reduction would mix them.
    gathered = [{} for _ in range(world_size())] if is_main_process() else None
    dist.gather_object(stats, gathered, dst=0)
    totals = defaultdict(float)
    counts = defaultdict(int)
    for worker_stats in gathered or ():
      for key, (total, count) in worker_stats.items():
        totals[key] += total
        counts[key] += count
    return {key: total / counts[key] for key, total in totals.items()}

  def _build_console_log(
    self,
    it: int,
    total_it: int,
    fps: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    metrics: dict[str, float],
    iteration_time: float,
    width: int,
    pad: int,
  ) -> str:
    """Build the formatted console log string."""
    log_string = f"""{"#" * width}\n"""
    log_string += (
      f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
    )

    log_string += f"""{"Run name:":>{pad}} {self.run_name}\n"""

    log_string += (
      f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
      f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
      f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
      f"""{"WM Learning time:":>{pad}} {wm_learn_time:.3f}s \n"""
      f"""{"Policy Learning time:":>{pad}} {policy_learn_time:.3f}s \n"""
    )

    for key, value in metrics.items():
      log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

    done_it = it + 1
    remaining_it = total_it - done_it
    eta = self.tot_time / done_it * remaining_it if done_it > 0 else 0.0
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
    )

    return log_string

  def log(
    self,
    it: int,
    total_it: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: dict[str, float | torch.Tensor],
    num_collected_steps: int,
    width: int = 80,
    pad: int = 40,
    *,
    tensorboard_writer: SummaryWriter | None = None,
  ) -> None:
    metrics = self._metrics(loss_dict)
    if not is_main_process():
      return
    num_collected_steps *= world_size()
    iteration_time = collect_time + wm_learn_time + policy_learn_time
    self.tot_timesteps += num_collected_steps
    self.tot_time += iteration_time
    fps = int(num_collected_steps / (iteration_time + 1e-6))

    row = metrics | {
      "Perf/total_steps": self.tot_timesteps,
      "Perf/total_fps": fps,
      "Perf/collection_time": collect_time,
      "Perf/wm_learning_time": wm_learn_time,
      "Perf/policy_learning_time": policy_learn_time,
    }
    if tensorboard_writer is not None:
      for tag, value in row.items():
        tensorboard_writer.add_scalar(tag, value, global_step=it)
      tensorboard_writer.flush()
    wandb.log(row, step=it)
    print(
      self._build_console_log(
        it,
        total_it,
        fps,
        collect_time,
        wm_learn_time,
        policy_learn_time,
        metrics,
        iteration_time,
        width,
        pad,
      )
    )
