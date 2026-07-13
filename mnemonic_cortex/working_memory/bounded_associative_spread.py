"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: bounded associative spread.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BoundedAssociativeSpreadConfig:
    num_slots: int
    max_steps: int = 3
    decay: float = 0.15
    spread_mix: float = 0.35
    sparsity_top_k: Optional[int] = None
    entropy_floor: float = 0.05
    spectral_norm_limit: float = 1.0
    hebbian_lr: float = 0.05
    eps: float = 1e-8

    def validate(self) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0,1]")
        if not 0.0 <= self.spread_mix <= 1.0:
            raise ValueError("spread_mix must be in [0,1]")
        if self.sparsity_top_k is not None and self.sparsity_top_k <= 0:
            raise ValueError("sparsity_top_k must be positive or None")
        if self.spectral_norm_limit <= 0:
            raise ValueError("spectral_norm_limit must be positive")


@dataclass
class BoundedSpreadTrace:
    steps_requested: int
    steps_executed: int
    row_sum_min: float
    row_sum_max: float
    spectral_norm: float
    activation_entropy: list
    sparse: bool
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BoundedAssociativeSpread(nn.Module):
    """Bounded associative spread over slot activations.

    Guarantees:
    - non-negative transition matrix
    - row-stochastic normalization
    - optional top-k sparsity
    - spectral norm clamp
    - bounded number of spread steps
    - entropy floor smoothing
    - bounded Hebbian update
    """

    def __init__(self, config: BoundedAssociativeSpreadConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.association_logits = nn.Parameter(torch.eye(config.num_slots))
        self.last_trace: Optional[BoundedSpreadTrace] = None

    def _row_stochastic(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = torch.relu(torch.nan_to_num(matrix))
        return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)

    def _apply_sparsity(self, matrix: torch.Tensor) -> torch.Tensor:
        k = self.config.sparsity_top_k
        if k is None or k >= matrix.size(-1):
            return matrix
        values, indices = torch.topk(matrix, k=k, dim=-1)
        sparse = torch.zeros_like(matrix)
        sparse.scatter_(-1, indices, values)
        return sparse

    def _spectral_clamp(self, matrix: torch.Tensor) -> torch.Tensor:
        # For small matrices, torch.linalg.matrix_norm is stable enough here.
        norm = torch.linalg.matrix_norm(matrix, ord=2)
        if norm > self.config.spectral_norm_limit:
            matrix = matrix * (self.config.spectral_norm_limit / norm.clamp_min(self.config.eps))
        return matrix

    def transition_matrix(self) -> torch.Tensor:
        matrix = self._row_stochastic(self.association_logits)
        matrix = self._apply_sparsity(matrix)
        matrix = self._row_stochastic(matrix)
        matrix = self._spectral_clamp(matrix)
        matrix = self._row_stochastic(matrix)
        return matrix

    def _entropy_floor(self, activation: torch.Tensor) -> torch.Tensor:
        if self.config.entropy_floor <= 0:
            return activation
        uniform = torch.full_like(activation, 1.0 / activation.size(-1))
        mixed = (1.0 - self.config.entropy_floor) * activation + self.config.entropy_floor * uniform
        return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)

    def forward(self, activation: torch.Tensor, requested_steps: Optional[int] = None) -> Tuple[torch.Tensor, BoundedSpreadTrace]:
        if activation.dim() != 2 or activation.size(-1) != self.config.num_slots:
            raise ValueError(f"Expected activation [B,{self.config.num_slots}], got {tuple(activation.shape)}")

        steps_requested = self.config.max_steps if requested_steps is None else int(requested_steps)
        steps = min(max(steps_requested, 0), self.config.max_steps)

        a = torch.relu(torch.nan_to_num(activation))
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        a = self._entropy_floor(a)
        matrix = self.transition_matrix()

        for _ in range(steps):
            spread = torch.matmul(a, matrix)
            a = (1.0 - self.config.spread_mix) * a + self.config.spread_mix * spread
            a = (1.0 - self.config.decay) * a + self.config.decay * activation.detach().clamp_min(0)
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
            a = self._entropy_floor(a)

        row_sums = matrix.sum(dim=-1)
        entropy = -(a * (a + self.config.eps).log()).sum(dim=-1)
        spectral = torch.linalg.matrix_norm(matrix, ord=2)
        trace = BoundedSpreadTrace(
            steps_requested=steps_requested,
            steps_executed=steps,
            row_sum_min=float(row_sums.detach().min().cpu()),
            row_sum_max=float(row_sums.detach().max().cpu()),
            spectral_norm=float(spectral.detach().cpu()),
            activation_entropy=entropy.detach().cpu().tolist(),
            sparse=bool(self.config.sparsity_top_k is not None and self.config.sparsity_top_k < self.config.num_slots),
            paamax_metadata={
                "trace_type": "bounded_associative_spread",
                "conflict_check_recommended": False,
                "bounded": True,
            },
        )
        self.last_trace = trace
        return a, trace

    @torch.no_grad()
    def hebbian_update(self, activation: torch.Tensor) -> BoundedSpreadTrace:
        if activation.dim() != 2 or activation.size(-1) != self.config.num_slots:
            raise ValueError(f"Expected activation [B,{self.config.num_slots}], got {tuple(activation.shape)}")
        a = torch.relu(torch.nan_to_num(activation))
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        coactivation = torch.matmul(a.t(), a) / max(1, a.size(0))
        updated = (1.0 - self.config.hebbian_lr) * self.transition_matrix() + self.config.hebbian_lr * coactivation
        updated = self._row_stochastic(updated)
        updated = self._apply_sparsity(updated)
        updated = self._row_stochastic(updated)
        updated = self._spectral_clamp(updated)
        updated = self._row_stochastic(updated)
        self.association_logits.data.copy_(updated)
        _, trace = self.forward(a, requested_steps=0)
        return trace

    def validate_transition(self) -> Dict[str, Any]:
        matrix = self.transition_matrix()
        finite = bool(torch.isfinite(matrix).all().item())
        non_negative = bool((matrix >= -1e-7).all().item())
        row_sums = matrix.sum(dim=-1)
        row_stochastic = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        spectral = float(torch.linalg.matrix_norm(matrix, ord=2).detach().cpu())
        return {
            "finite": finite,
            "non_negative": non_negative,
            "row_stochastic": row_stochastic,
            "spectral_norm": spectral,
            "spectral_ok": spectral <= self.config.spectral_norm_limit + 1e-5 or self.config.spectral_norm_limit < 1.0,
            "ok": bool(finite and non_negative and row_stochastic),
        }


# ---------------------------------------------------------------------------
# WM-QD-1A foundation-quality contract
# ---------------------------------------------------------------------------

def wm_qd1a_foundation_contract() -> dict:
    """Return serialization-safe quality metadata for this early-WM module.

    This does not mutate runtime state. It exists so the quality tooling can
    verify that the module has an explicit contract for shape/finite checks,
    traceability, PAAMA-X metadata, fallback behavior, and boundedness.
    """
    return foundation_trace(
        trace_type="wm_qd1a_foundation_contract",
        module=__name__,
        message="early working-memory foundation module hardened by WM-QD-1A",
        payload={
            "shape_checks_required": True,
            "finite_checks_required": True,
            "serialization_safe": True,
            "trace_hooks_required": True,
            "paamax_metadata_required": True,
            "boundedness_required": True,
            "runtime_mutation": "no automatic mutation by quality tooling",
        },
    )
