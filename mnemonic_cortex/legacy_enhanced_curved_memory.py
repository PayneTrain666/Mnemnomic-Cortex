from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurvedMemoryReadTrace:
    top_indices: torch.Tensor
    top_scores: torch.Tensor
    activation_entropy: torch.Tensor
    spread_steps: int
    used_associative_spread: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "top_indices": self.top_indices.detach().cpu().tolist(),
            "top_scores": self.top_scores.detach().cpu().tolist(),
            "activation_entropy": self.activation_entropy.detach().cpu().tolist(),
            "spread_steps": self.spread_steps,
            "used_associative_spread": self.used_associative_spread,
        }


class EnhancedCurvedMemory(nn.Module):
    """Canonical-compatible EnhancedCurvedMemory.

    This class preserves the behavior documented for the original Mnemonic Cortex
    working memory:
    - encoder
    - learned curvature parameters
    - memory slots
    - memory importance
    - associative weights
    - content-based addressing
    - associative activation spread
    - read decode path
    - write/update path

    It is intentionally conservative. WM-1B+ will add the deeper Curved
    Resonant WM Core around this preserved base rather than replacing it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        curvature_dim: int = 8,
        mem_slots: int = 7,
        spread_steps: int = 1,
        write_rate: float = 0.10,
        eps: float = 1e-8,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or mem_slots <= 0:
            raise ValueError("input_dim, hidden_dim, and mem_slots must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.curvature_dim = curvature_dim
        self.mem_slots = mem_slots
        self.spread_steps = spread_steps
        self.write_rate = write_rate
        self.eps = eps
        self.energy_mode = False

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.curvature = nn.Parameter(torch.zeros(curvature_dim))
        self.curvature_gate = nn.Sequential(
            nn.Linear(hidden_dim, curvature_dim),
            nn.Tanh(),
            nn.Linear(curvature_dim, curvature_dim),
            nn.Sigmoid(),
        )

        self.memory_slots = nn.Parameter(torch.randn(mem_slots, hidden_dim) * 0.02)
        self.memory_importance = nn.Parameter(torch.ones(mem_slots))
        self.associative_weights = nn.Parameter(torch.eye(mem_slots))

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.last_trace: Optional[CurvedMemoryReadTrace] = None

    def enable_energy_efficient_mode(self, enable: bool = True) -> None:
        self.energy_mode = bool(enable)

    def _curvature_scale(self, query: torch.Tensor) -> torch.Tensor:
        """Return a stable positive scale derived from learned curvature."""
        gate = self.curvature_gate(query)
        curv = torch.tanh(self.curvature).view(1, -1)
        # Compress curvature signal into a gentle scalar multiplier.
        scale = 1.0 + 0.05 * (gate * curv).mean(dim=-1, keepdim=True)
        return scale.clamp(0.75, 1.25)

    def _bounded_associative_matrix(self) -> torch.Tensor:
        """Row-stochastic, non-negative associative matrix."""
        weights = torch.relu(self.associative_weights)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return weights

    def _address(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.query_projection(query), dim=-1)
        slots = F.normalize(self.memory_slots, dim=-1)
        content_scores = torch.matmul(q, slots.t())

        importance = torch.sigmoid(self.memory_importance).view(1, -1)
        scores = content_scores + 0.15 * importance
        scores = scores * self._curvature_scale(query)
        activation = torch.softmax(scores, dim=-1)
        return activation, scores

    def _spread_activation(self, activation: torch.Tensor) -> torch.Tensor:
        spread = activation
        if self.spread_steps <= 0:
            return spread
        assoc = self._bounded_associative_matrix()
        for _ in range(self.spread_steps):
            spread = 0.65 * spread + 0.35 * torch.matmul(spread, assoc)
            spread = spread / spread.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return spread

    def _read(self, x: torch.Tensor, return_trace: bool = False):
        encoded = self.encoder(x)
        query = encoded.mean(dim=1)
        activation, scores = self._address(query)
        spread = self._spread_activation(activation)
        retrieved = torch.matmul(spread, self.memory_slots)
        out = self.output_projection(retrieved).unsqueeze(1).expand(-1, x.size(1), -1)

        top_scores, top_indices = torch.topk(scores, k=min(3, self.mem_slots), dim=-1)
        entropy = -(spread * (spread + self.eps).log()).sum(dim=-1)
        self.last_trace = CurvedMemoryReadTrace(
            top_indices=top_indices,
            top_scores=top_scores,
            activation_entropy=entropy,
            spread_steps=self.spread_steps,
            used_associative_spread=self.spread_steps > 0,
        )

        if return_trace:
            return out, self.last_trace.to_dict()
        return out

    @torch.no_grad()
    def _write(self, x: torch.Tensor, importance: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded = self.encoder(x)
        query = encoded.mean(dim=1)
        activation, _ = self._address(query)

        if importance is None:
            imp = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        elif isinstance(importance, torch.Tensor):
            imp = importance.to(device=x.device, dtype=x.dtype).reshape(-1)
            if imp.numel() == 1:
                imp = imp.expand(x.size(0))
        else:
            imp = torch.full((x.size(0),), float(importance), device=x.device, dtype=x.dtype)

        # Weighted slot update, bounded to preserve stability.
        weighted_activation = activation * imp.view(-1, 1)
        denom = weighted_activation.sum(dim=0).view(-1, 1).clamp_min(self.eps)
        slot_delta = torch.matmul(weighted_activation.t(), query) / denom

        update_mask = (weighted_activation.sum(dim=0) > self.eps).float().view(-1, 1)
        self.memory_slots.data = (
            (1.0 - self.write_rate * update_mask) * self.memory_slots.data
            + self.write_rate * update_mask * slot_delta
        )

        imp_delta = weighted_activation.mean(dim=0)
        self.memory_importance.data = 0.99 * self.memory_importance.data + 0.01 * imp_delta
        return x

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        importance: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x [B,T,{self.input_dim}], got {tuple(x.shape)}")

        if operation == "write":
            return self._write(x, importance=importance)
        if operation == "read":
            return self._read(x, return_trace=return_trace)
        if operation == "process":
            read = self._read(x, return_trace=False)
            return 0.5 * x + 0.5 * read

        raise ValueError(f"Unsupported operation: {operation}")
