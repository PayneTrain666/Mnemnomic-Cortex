"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: curved slot state.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurvedSlotStateConfig:
    """Configuration for the curved slot-state bank.

    Tensor conventions:
    - slot tensors: [S,D]
    - selected slot tensors: [K,D]
    """

    num_slots: int
    dim: int
    position_radius: float = 0.995
    curvature_min: float = -5.0
    curvature_max: float = 5.0
    importance_min: float = 0.0
    importance_max: float = 1.0
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    eps: float = 1e-8

    def validate(self) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not (0.0 < self.position_radius < 1.0):
            raise ValueError("position_radius must be in (0,1)")
        if self.curvature_min >= self.curvature_max:
            raise ValueError("curvature_min must be < curvature_max")


@dataclass
class CurvedSlotSnapshot:
    slot_id: List[str]
    content: torch.Tensor
    position: torch.Tensor
    tangent: torch.Tensor
    phase: torch.Tensor
    curvature: torch.Tensor
    importance: torch.Tensor
    confidence: torch.Tensor
    last_updated: torch.Tensor
    trace_links: List[List[str]]

    def shape_summary(self) -> Dict[str, Any]:
        return {
            "slot_count": len(self.slot_id),
            "content": list(self.content.shape),
            "position": list(self.position.shape),
            "tangent": list(self.tangent.shape),
            "phase": list(self.phase.shape),
            "curvature": list(self.curvature.shape),
            "importance": list(self.importance.shape),
            "confidence": list(self.confidence.shape),
            "last_updated": list(self.last_updated.shape),
            "trace_link_lists": len(self.trace_links),
        }


@dataclass
class CurvedSlotStateTrace:
    operation: str
    selected_slots: List[str] = field(default_factory=list)
    ok: bool = True
    repaired: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CurvedSlotStateBank(nn.Module):
    """Trainable curved slot state.

    Each slot has:
    - slot_id
    - content
    - position
    - tangent
    - phase
    - curvature
    - importance
    - confidence
    - last_updated
    - trace_links

    WM-1C intentionally creates the state bank and stability utilities. WM-1D
    will use this state for geometry-aware addressing and bounded spread.
    """

    def __init__(
        self,
        config: CurvedSlotStateConfig,
        slot_ids: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.slot_ids = list(slot_ids) if slot_ids is not None else [f"slot_{i:04d}" for i in range(config.num_slots)]
        if len(self.slot_ids) != config.num_slots:
            raise ValueError("slot_ids length must equal num_slots")

        scale = 0.02
        self.content = nn.Parameter(torch.randn(config.num_slots, config.dim) * scale)
        self.position = nn.Parameter(torch.randn(config.num_slots, config.dim) * scale)
        self.tangent = nn.Parameter(torch.randn(config.num_slots, config.dim) * scale)
        self.phase = nn.Parameter(torch.randn(config.num_slots, config.dim) * scale)
        self.curvature = nn.Parameter(torch.zeros(config.num_slots))
        self.importance = nn.Parameter(torch.full((config.num_slots,), 0.5))
        self.confidence = nn.Parameter(torch.full((config.num_slots,), 0.5))

        self.register_buffer("last_updated", torch.zeros(config.num_slots))
        self.trace_links: List[List[str]] = [[] for _ in range(config.num_slots)]
        self.last_trace: Optional[CurvedSlotStateTrace] = None

    def _clamp_position(self, position: torch.Tensor) -> torch.Tensor:
        radius = position.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
        factor = torch.clamp(self.config.position_radius / radius, max=1.0)
        return position * factor

    def stable_tensors(self) -> Dict[str, torch.Tensor]:
        """Return repaired/clamped views without mutating parameters."""
        return {
            "content": torch.nan_to_num(self.content),
            "position": self._clamp_position(torch.nan_to_num(self.position)),
            "tangent": torch.nan_to_num(self.tangent),
            "phase": F.normalize(torch.nan_to_num(self.phase), dim=-1, eps=self.config.eps),
            "curvature": torch.clamp(torch.nan_to_num(self.curvature), self.config.curvature_min, self.config.curvature_max),
            "importance": torch.clamp(torch.nan_to_num(self.importance), self.config.importance_min, self.config.importance_max),
            "confidence": torch.clamp(torch.nan_to_num(self.confidence), self.config.confidence_min, self.config.confidence_max),
            "last_updated": torch.nan_to_num(self.last_updated),
        }

    @torch.no_grad()
    def repair_in_place(self) -> CurvedSlotStateTrace:
        stable = self.stable_tensors()
        self.content.data.copy_(stable["content"])
        self.position.data.copy_(stable["position"])
        self.tangent.data.copy_(stable["tangent"])
        self.phase.data.copy_(stable["phase"])
        self.curvature.data.copy_(stable["curvature"])
        self.importance.data.copy_(stable["importance"])
        self.confidence.data.copy_(stable["confidence"])
        trace = CurvedSlotStateTrace(operation="repair", repaired=True, metadata=self.validate_state())
        self.last_trace = trace
        return trace

    def validate_state(self) -> Dict[str, Any]:
        stable = self.stable_tensors()
        finite = all(torch.isfinite(v).all().item() for v in stable.values())
        position_ok = bool((stable["position"].norm(dim=-1) <= self.config.position_radius + 1e-6).all().item())
        curvature_ok = bool(((stable["curvature"] >= self.config.curvature_min) & (stable["curvature"] <= self.config.curvature_max)).all().item())
        importance_ok = bool(((stable["importance"] >= self.config.importance_min) & (stable["importance"] <= self.config.importance_max)).all().item())
        confidence_ok = bool(((stable["confidence"] >= self.config.confidence_min) & (stable["confidence"] <= self.config.confidence_max)).all().item())
        return {
            "finite": bool(finite),
            "position_ok": position_ok,
            "curvature_ok": curvature_ok,
            "importance_ok": importance_ok,
            "confidence_ok": confidence_ok,
            "ok": bool(finite and position_ok and curvature_ok and importance_ok and confidence_ok),
        }

    def snapshot(self, indices: Optional[Iterable[int]] = None) -> CurvedSlotSnapshot:
        stable = self.stable_tensors()
        if indices is None:
            idx = torch.arange(self.config.num_slots, device=self.content.device)
        else:
            idx = torch.tensor(list(indices), device=self.content.device, dtype=torch.long)
        ids = [self.slot_ids[int(i)] for i in idx.detach().cpu().tolist()]
        traces = [self.trace_links[int(i)] for i in idx.detach().cpu().tolist()]
        return CurvedSlotSnapshot(
            slot_id=ids,
            content=stable["content"].index_select(0, idx),
            position=stable["position"].index_select(0, idx),
            tangent=stable["tangent"].index_select(0, idx),
            phase=stable["phase"].index_select(0, idx),
            curvature=stable["curvature"].index_select(0, idx),
            importance=stable["importance"].index_select(0, idx),
            confidence=stable["confidence"].index_select(0, idx),
            last_updated=stable["last_updated"].index_select(0, idx),
            trace_links=traces,
        )

    @torch.no_grad()
    def update_slots(
        self,
        indices: torch.Tensor,
        content: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        tangent: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
        curvature: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
        trace_link: Optional[str] = None,
    ) -> CurvedSlotStateTrace:
        if indices.dim() != 1:
            raise ValueError("indices must be 1D")
        if (indices < 0).any() or (indices >= self.config.num_slots).any():
            raise IndexError("slot index out of range")

        def assign(param: torch.nn.Parameter, value: Optional[torch.Tensor], name: str) -> None:
            if value is None:
                return
            if value.shape != param.data.index_select(0, indices).shape:
                raise ValueError(f"{name} update shape mismatch")
            param.data.index_copy_(0, indices, value.to(device=param.device, dtype=param.dtype))

        assign(self.content, content, "content")
        assign(self.position, position, "position")
        assign(self.tangent, tangent, "tangent")
        assign(self.phase, phase, "phase")
        if curvature is not None:
            self.curvature.data.index_copy_(0, indices, curvature.to(device=self.curvature.device, dtype=self.curvature.dtype))
        if importance is not None:
            self.importance.data.index_copy_(0, indices, importance.to(device=self.importance.device, dtype=self.importance.dtype))
        if confidence is not None:
            self.confidence.data.index_copy_(0, indices, confidence.to(device=self.confidence.device, dtype=self.confidence.dtype))

        ts = float(timestamp) if timestamp is not None else float(self.last_updated.max().item() + 1.0)
        self.last_updated.index_fill_(0, indices.to(self.last_updated.device), ts)

        selected = []
        for i in indices.detach().cpu().tolist():
            selected.append(self.slot_ids[int(i)])
            if trace_link is not None:
                self.trace_links[int(i)].append(trace_link)

        repair_trace = self.repair_in_place()
        trace = CurvedSlotStateTrace(
            operation="update_slots",
            selected_slots=selected,
            ok=repair_trace.metadata.get("ok", False),
            repaired=repair_trace.repaired,
            metadata=self.validate_state(),
        )
        self.last_trace = trace
        return trace

    def forward(self, indices: Optional[torch.Tensor] = None) -> CurvedSlotSnapshot:
        if indices is None:
            return self.snapshot()
        return self.snapshot(indices.detach().cpu().tolist())


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
