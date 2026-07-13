"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: curved resonant wm core.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wm_curved_core import WMCurvedAssociativeCore
from .curved_slot_state import CurvedSlotStateBank, CurvedSlotStateConfig
from .curvature_metric_policy import CurvatureMetricPolicy, CurvatureMetricPolicyConfig
from .geometry_aware_addressing import GeometryAwareAddressing
from .bounded_associative_spread import BoundedAssociativeSpread
from .curved_local_trace import CurvedLocalTraceBuilder
from .curved_shadow_write import CurvedShadowWriteBuffer


@dataclass
class ResonanceStepTrace:
    step: int
    top_indices: List[List[int]]
    top_scores: List[List[float]]
    activation_entropy: List[float]
    delta_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurvedResonanceTrace:
    operation: str
    resonance_steps_requested: int
    resonance_steps_executed: int
    bounded: bool
    novelty_score: float
    confidence_score: float
    lightbulb: bool
    inner_trace: Optional[Dict[str, Any]] = None
    step_traces: List[ResonanceStepTrace] = field(default_factory=list)
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["step_traces"] = [s.to_dict() for s in self.step_traces]
        return out


@dataclass
class CurvedResonanceConfig:
    input_dim: int
    hidden_dim: int = 128
    resonance_slots: int = 16
    max_resonance_steps: int = 3
    requested_resonance_steps: int = 2
    resonance_update_rate: float = 0.35
    activation_temperature: float = 1.0
    lightbulb_threshold: float = 0.15
    dropout: float = 0.0
    trace_top_k: int = 3
    eps: float = 1e-8

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.resonance_slots <= 0:
            raise ValueError("resonance_slots must be positive")
        if self.max_resonance_steps < 0:
            raise ValueError("max_resonance_steps must be non-negative")
        if self.requested_resonance_steps < 0:
            raise ValueError("requested_resonance_steps must be non-negative")
        if self.activation_temperature <= 0:
            raise ValueError("activation_temperature must be positive")


class CurvedResonantWMCore(nn.Module):
    """Curved Resonant Working Memory Core.

    WM-1B purpose:
    - keep WMCurvedAssociativeCore as the preserved inner curved WM.
    - add a bounded resonance/lightbulb loop around it.
    - emit local resonance traces.
    - expose PAAMA-X-compatible metadata hooks.
    - do not replace the original curved memory identity.

    Input:
    - x: [B,T,D]

    Output:
    - y: [B,T,D]
    - optional trace dict
    """

    def __init__(
        self,
        config: CurvedResonanceConfig,
        inner_core: Optional[WMCurvedAssociativeCore] = None,
        geometry_aware_addressing: Optional[GeometryAwareAddressing] = None,
        bounded_spread: Optional[BoundedAssociativeSpread] = None,
        shadow_write_buffer: Optional[CurvedShadowWriteBuffer] = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.geometry_aware_addressing = geometry_aware_addressing
        self.bounded_spread = bounded_spread
        self.shadow_write_buffer = shadow_write_buffer
        self.inner_core = inner_core or WMCurvedAssociativeCore(
            input_dim=config.input_dim,
            hidden_dim=max(config.hidden_dim, config.input_dim),
            mem_slots=7,
        )

        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.inner_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        self.resonance_slots = nn.Parameter(torch.randn(config.resonance_slots, config.hidden_dim) * 0.02)
        self.associative_resonance = nn.Parameter(torch.eye(config.resonance_slots))
        self.resonance_update = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.last_trace: Optional[CurvedResonanceTrace] = None
        # WM-1C integration note: CurvedSlotStateBank and CurvatureMetricPolicy
        # are implemented as standalone modules and will be deeply wired into
        # addressing/spread in WM-1D.

    def _bounded_assoc(self) -> torch.Tensor:
        assoc = torch.relu(self.associative_resonance)
        assoc = assoc / assoc.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        return assoc

    def _slot_activation(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return activation and raw scores over resonance slots.

        state: [B,H]
        activation: [B,S]
        """
        q = F.normalize(state, dim=-1)
        slots = F.normalize(self.resonance_slots, dim=-1)
        scores = torch.matmul(q, slots.t()) / self.config.activation_temperature
        activation = torch.softmax(scores, dim=-1)
        return activation, scores

    def _spread(self, activation: torch.Tensor) -> torch.Tensor:
        assoc = self._bounded_assoc()
        spread = 0.70 * activation + 0.30 * torch.matmul(activation, assoc)
        spread = spread / spread.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        return spread

    def _step_trace(self, step: int, scores: torch.Tensor, activation: torch.Tensor, delta: torch.Tensor) -> ResonanceStepTrace:
        k = min(self.config.trace_top_k, scores.size(-1))
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        entropy = -(activation * (activation + self.config.eps).log()).sum(dim=-1)
        return ResonanceStepTrace(
            step=step,
            top_indices=top_indices.detach().cpu().tolist(),
            top_scores=top_scores.detach().cpu().tolist(),
            activation_entropy=entropy.detach().cpu().tolist(),
            delta_norm=float(delta.detach().norm().cpu()),
        )

    def _run_resonance(self, seed: torch.Tensor) -> Tuple[torch.Tensor, List[ResonanceStepTrace]]:
        steps = min(self.config.requested_resonance_steps, self.config.max_resonance_steps)
        state = seed
        traces: List[ResonanceStepTrace] = []

        for step in range(steps):
            activation, scores = self._slot_activation(state)
            spread = self._spread(activation)
            read = torch.matmul(spread, self.resonance_slots)
            updated = self.resonance_update(read, state)
            delta = updated - state
            state = (1.0 - self.config.resonance_update_rate) * state + self.config.resonance_update_rate * updated
            state = self.dropout(state)
            traces.append(self._step_trace(step, scores, spread, delta))

        return state, traces

    def _make_trace(
        self,
        operation: str,
        seed: torch.Tensor,
        final: torch.Tensor,
        inner_trace: Optional[Dict[str, Any]],
        step_traces: List[ResonanceStepTrace],
    ) -> CurvedResonanceTrace:
        novelty = torch.norm(final - seed, dim=-1)
        novelty_score = float(novelty.detach().mean().cpu())
        confidence_score = float((1.0 / (1.0 + novelty)).detach().mean().cpu())
        lightbulb = novelty_score >= self.config.lightbulb_threshold

        return CurvedResonanceTrace(
            operation=operation,
            resonance_steps_requested=self.config.requested_resonance_steps,
            resonance_steps_executed=len(step_traces),
            bounded=self.config.requested_resonance_steps <= self.config.max_resonance_steps,
            novelty_score=novelty_score,
            confidence_score=confidence_score,
            lightbulb=bool(lightbulb),
            inner_trace=inner_trace,
            step_traces=step_traces,
            paamax_metadata={
                "trace_type": "curved_resonance",
                "write_permission_required": operation == "write",
                "conflict_check_recommended": bool(lightbulb),
                "confidence": confidence_score,
                "novelty": novelty_score,
            },
        )

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        importance: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        if x.dim() != 3 or x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected x [B,T,{self.config.input_dim}], got {tuple(x.shape)}")

        # WM-1E: writes are staged through curved shadow write buffer when available.
        if operation == "write":
            if self.shadow_write_buffer is not None:
                write_summary = x.mean(dim=1)
                # Map batch summaries to the first K slots as a conservative starter policy.
                k = min(write_summary.size(0), getattr(self.shadow_write_buffer.config, "max_pending", write_summary.size(0)))
                slot_count = None
                if getattr(self.shadow_write_buffer, "slot_bank", None) is not None:
                    slot_count = self.shadow_write_buffer.slot_bank.config.num_slots
                if slot_count is None:
                    slot_count = k
                slot_indices = torch.arange(k, device=x.device, dtype=torch.long) % max(1, slot_count)
                proposal = self.shadow_write_buffer.stage(
                    slot_indices=slot_indices,
                    content_delta=write_summary[:k].detach(),
                    confidence=1.0 if importance is None else float(torch.as_tensor(importance).float().mean().item()),
                    source="curved_resonant_wm",
                    paamax_permission=True,
                    metadata={"operation": "write", "shape": list(x.shape)},
                )
                local_trace = CurvedLocalTraceBuilder("write").build()
                local_trace.set_write_decision("staged", proposal.proposal_id, source="curved_resonant_wm")
                decision = self.shadow_write_buffer.commit(proposal.proposal_id, trace=local_trace)

                trace = CurvedResonanceTrace(
                    operation="write",
                    resonance_steps_requested=0,
                    resonance_steps_executed=0,
                    bounded=True,
                    novelty_score=0.0,
                    confidence_score=1.0,
                    lightbulb=False,
                    inner_trace=self.inner_core.last_trace,
                    paamax_metadata={
                        "trace_type": "curved_resonance_shadow_write",
                        "write_permission_required": True,
                        "shadow_write": decision.to_dict(),
                        "local_trace": local_trace.to_dict(),
                    },
                )
                self.last_trace = trace
                if return_trace:
                    return x, trace.to_dict()
                return x

            out = self.inner_core(x, operation="write", importance=importance)
            local_trace = CurvedLocalTraceBuilder("write").build()
            local_trace.set_write_decision("delegated", None, reason="no_shadow_buffer")
            trace = CurvedResonanceTrace(
                operation="write",
                resonance_steps_requested=0,
                resonance_steps_executed=0,
                bounded=True,
                novelty_score=0.0,
                confidence_score=1.0,
                lightbulb=False,
                inner_trace=self.inner_core.last_trace,
                paamax_metadata={
                    "trace_type": "curved_resonance_write_delegate",
                    "write_permission_required": True,
                    "local_trace": local_trace.to_dict(),
                },
            )
            self.last_trace = trace
            if return_trace:
                return out, trace.to_dict()
            return out

        inner_result = self.inner_core(x, operation="read", return_trace=True)
        if isinstance(inner_result, tuple):
            inner_out, inner_trace = inner_result
        else:
            inner_out, inner_trace = inner_result, self.inner_core.last_trace

        seed_tokens = self.input_projection(x) + self.inner_projection(inner_out)
        seed = seed_tokens.mean(dim=1)

        wm1d_addressing_trace = None
        wm1d_spread_trace = None
        if self.geometry_aware_addressing is not None:
            # WM-1D integration bridge:
            # CurvedResonantWMCore uses hidden_dim seed state, while addressing
            # may operate on input_dim slot content. Use the compatible query.
            addr_dim = getattr(getattr(self.geometry_aware_addressing, "config", None), "dim", seed.size(-1))
            if seed.size(-1) == addr_dim:
                addr_query = seed
            elif x.size(-1) == addr_dim:
                addr_query = x.mean(dim=1)
            elif inner_out.size(-1) == addr_dim:
                addr_query = inner_out.mean(dim=1)
            else:
                addr_query = None

            if addr_query is not None:
                addr_out = self.geometry_aware_addressing(addr_query)
                wm1d_addressing_trace = addr_out.trace.to_dict()
                if self.bounded_spread is not None:
                    spread_activation, spread_trace = self.bounded_spread(addr_out.activation)
                    wm1d_spread_trace = spread_trace.to_dict()
                # Nudge seed only when read_content lives in the same space.
                if addr_out.read_content.size(-1) == seed.size(-1):
                    seed = 0.80 * seed + 0.20 * addr_out.read_content
                else:
                    wm1d_addressing_trace["seed_nudge_skipped"] = "dimension_mismatch"

        resonant, step_traces = self._run_resonance(seed)

        resonant_tokens = resonant.unsqueeze(1).expand(-1, x.size(1), -1)
        delta_tokens = self.output_projection(resonant_tokens)
        out = 0.60 * inner_out + 0.40 * delta_tokens

        if operation == "process":
            out = 0.50 * x + 0.50 * out
        elif operation != "read":
            raise ValueError(f"Unsupported operation: {operation}")

        trace = self._make_trace(operation, seed, resonant, inner_trace, step_traces)
        local_trace = CurvedLocalTraceBuilder(operation).from_resonance_trace(trace.to_dict()).build()
        trace.paamax_metadata["curved_local_trace"] = local_trace.to_dict()
        if wm1d_addressing_trace is not None:
            trace.paamax_metadata["geometry_aware_addressing"] = wm1d_addressing_trace
        if wm1d_spread_trace is not None:
            trace.paamax_metadata["bounded_associative_spread"] = wm1d_spread_trace
        self.last_trace = trace

        if return_trace:
            return out, trace.to_dict()
        return out


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
