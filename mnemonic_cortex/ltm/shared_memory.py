"""Shared value store used by Spatial LTM and MANN geometry-key banks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SharedWriteTrace:
    slots: torch.Tensor
    mode: str
    lr: float
    scratch_ttl: Optional[int] = None


class SharedValueStore(nn.Module):
    """Shared Euclidean values, geometry-specific keys elsewhere.

    This implements the DOCX idea: content values live once, while HG/CGMN/Spatial
    and MANN banks maintain separate geometry-specific address spaces.
    """

    def __init__(self, slots: int, value_dim: int, scratch_ttl_default: int = 4):
        super().__init__()
        if slots <= 0 or value_dim <= 0:
            raise ValueError("slots and value_dim must be positive")
        self.slots = int(slots)
        self.value_dim = int(value_dim)
        self.values = nn.Parameter(torch.randn(slots, value_dim) * 0.02)
        self.register_buffer("scratch_mask", torch.zeros(slots, dtype=torch.bool))
        self.register_buffer("scratch_ttl", torch.zeros(slots, dtype=torch.long))
        self.scratch_ttl_default = int(scratch_ttl_default)

    @torch.no_grad()
    def consolidate_write(self, slot_idx: torch.Tensor, new_values: torch.Tensor, lr: float = 0.02) -> SharedWriteTrace:
        slot_idx = slot_idx.reshape(-1).long().to(self.values.device)
        new_values = new_values.reshape(slot_idx.numel(), self.value_dim).to(self.values.device, self.values.dtype)
        lr = float(max(0.0, min(1.0, lr)))
        for b, s_t in enumerate(slot_idx):
            s = int(s_t.item()) % self.slots
            self.values[s].mul_(1.0 - lr).add_(lr * new_values[b])
        return SharedWriteTrace(slots=slot_idx.detach().cpu(), mode="consolidate", lr=lr)

    @torch.no_grad()
    def scratch_write(self, slot_idx: torch.Tensor, new_values: torch.Tensor, lr: float = 0.05, ttl: Optional[int] = None) -> SharedWriteTrace:
        trace = self.consolidate_write(slot_idx, new_values, lr=lr)
        slots = slot_idx.reshape(-1).long().to(self.values.device) % self.slots
        self.scratch_mask[slots] = True
        self.scratch_ttl[slots] = int(ttl if ttl is not None else self.scratch_ttl_default)
        return SharedWriteTrace(slots=trace.slots, mode="scratch", lr=trace.lr, scratch_ttl=int(ttl or self.scratch_ttl_default))

    @torch.no_grad()
    def decay_ttl(self) -> Dict[str, int]:
        alive = self.scratch_mask & (self.scratch_ttl > 0)
        self.scratch_ttl[alive] -= 1
        expired = self.scratch_mask & (self.scratch_ttl <= 0)
        expired_count = int(expired.sum().item())
        self.scratch_mask[expired] = False
        self.scratch_ttl[expired] = 0
        return {"expired": expired_count, "alive": int(self.scratch_mask.sum().item())}

    def snapshot(self) -> Dict[str, int]:
        return {"slots": self.slots, "value_dim": self.value_dim, "scratch_alive": int(self.scratch_mask.sum().item())}
