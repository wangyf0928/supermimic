"""T3 motion-tracking policy, with separate self and reference token encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

if TYPE_CHECKING:
  from supermimic.tasks.task_spec import TaskSpec


@dataclass
class T3ModelCfg:
  """Window lengths count frames. The last next_length task frames are future references."""

  self_length: int = 32
  task_length: int = 32
  next_length: int = 2
  embedding_dims: tuple[int, ...] = (256, 256)
  hidden_dim: int = 128
  ff_dim: int = 512
  num_heads: int = 4
  num_blocks: int = 1
  chunk_size: int = 4


class _RMSNorm(nn.Module):
  def __init__(self, width: int):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(width))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class _WindowEncoder(nn.Module):
  """Add time to each frame, then embed contiguous oldest-to-newest chunks."""

  times: torch.Tensor

  def __init__(
    self, cfg: T3ModelCfg, frame_dim: int, length: int, future: int, dt: float
  ):
    super().__init__()
    if not 0 <= future < length:
      raise ValueError(
        "each window needs a current frame and a nonnegative future length"
      )
    if cfg.chunk_size <= 0 or length % cfg.chunk_size:
      raise ValueError("chunk_size must be positive and divide the window length")
    self.num_tokens = length // cfg.chunk_size
    offsets = torch.arange(length, dtype=torch.float32) - (length - 1 - future)
    self.register_buffer("times", offsets.view(1, length, 1) * dt, persistent=False)
    dims = ((frame_dim + 1) * cfg.chunk_size, *cfg.embedding_dims, cfg.hidden_dim)
    layers = []
    for index, (width, next_width) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      if index:
        layers.append(nn.SiLU())
      layers.append(nn.Linear(width, next_width))
    self.embedding = nn.Sequential(*layers, _RMSNorm(cfg.hidden_dim))

  def forward(self, frames: torch.Tensor) -> torch.Tensor:
    times = self.times.to(frames.dtype).expand(frames.size(0), -1, -1)
    chunks = torch.cat([frames, times], dim=-1).reshape(
      frames.size(0), self.num_tokens, -1
    )
    return self.embedding(chunks).flip(1)


class _TransformerBlock(nn.Module):
  """Pre-normalized, bidirectional self-attention and a gated SiLU MLP."""

  def __init__(self, cfg: T3ModelCfg):
    super().__init__()
    width = cfg.hidden_dim
    if cfg.num_heads <= 0 or width % cfg.num_heads:
      raise ValueError("num_heads must be positive and divide hidden_dim")
    self.num_heads = cfg.num_heads
    self.attn_norm = _RMSNorm(width)
    self.qkv = nn.Linear(width, 3 * width, bias=False)
    self.out = nn.Linear(width, width, bias=False)
    self.ff_norm = _RMSNorm(width)
    self.gate_value = nn.Linear(width, 2 * cfg.ff_dim, bias=False)
    self.ff_out = nn.Linear(cfg.ff_dim, width, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, tokens, width = x.shape
    qkv = self.qkv(self.attn_norm(x)).reshape(batch, tokens, 3, self.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    attended = F.scaled_dot_product_attention(q, k, v)
    x = x + self.out(attended.transpose(1, 2).reshape(batch, tokens, width))
    gate, value = self.gate_value(self.ff_norm(x)).chunk(2, dim=-1)
    return x + self.ff_out(F.silu(gate) * value)


class T3Policy(nn.Module):
  """Read the newest self chunk after it has attended to both observation streams."""

  def __init__(
    self, self_dim: int, task_dim: int, action_dim: int, dt: float, cfg: T3ModelCfg
  ):
    super().__init__()
    self.cfg = cfg
    self.self_dim = self_dim
    self.task_dim = task_dim
    self.action_dim = action_dim
    if min(cfg.hidden_dim, cfg.ff_dim, cfg.num_blocks, *cfg.embedding_dims) < 1:
      raise ValueError("T3 network widths and num_blocks must be positive")
    self.self_encoder = _WindowEncoder(cfg, self_dim, cfg.self_length, 0, dt)
    self.task_encoder = _WindowEncoder(
      cfg, task_dim, cfg.task_length, cfg.next_length, dt
    )
    self.transformer = nn.Sequential(
      *[_TransformerBlock(cfg) for _ in range(cfg.num_blocks)], _RMSNorm(cfg.hidden_dim)
    )
    head_in = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
    head_out = nn.Linear(cfg.hidden_dim, action_dim)
    self.head = nn.Sequential(head_in, nn.SiLU(), head_out)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
    nn.init.orthogonal_(head_in.weight)
    nn.init.orthogonal_(head_out.weight, gain=0.05)

  def forward(
    self, self_history: torch.Tensor, task_history: torch.Tensor
  ) -> torch.Tensor:
    tokens = torch.cat(
      [self.self_encoder(self_history), self.task_encoder(task_history)], dim=1
    )
    return self.head(self.transformer(tokens)[:, 0]).float()


class T3Actor(nn.Module):
  """Motion-only T3 actor for simulator collection and differentiable rollouts."""

  def __init__(
    self,
    self_dim: int,
    task_dim: int,
    action_dim: int,
    dt: float,
    cfg: T3ModelCfg,
    task_specs: list[TaskSpec],
    slot_mix: dict[str, float],
  ):
    super().__init__()
    if [task.name for task in task_specs] != ["motion"]:
      raise ValueError("T3Actor requires one motion task spec")
    self.task_specs = task_specs
    self.set_slot_mix(slot_mix)
    self.cfg = cfg
    self.history_length = cfg.self_length
    self.network = T3Policy(self_dim, task_dim, action_dim, dt, cfg)

  def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
    if (
      slot_mix.get("motion") != 1.0
      or slot_mix.get("velocity", 0.0) != 0.0
      or slot_mix.keys() - {"motion", "velocity"}
    ):
      raise ValueError("supermimic-v2 requires all slots to use the motion task")

  def forward(self, obs_td: TensorDict) -> torch.Tensor:
    return self.network(obs_td["self_obs"], obs_td["task_obs"])
