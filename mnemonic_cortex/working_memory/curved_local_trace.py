from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class CurvedTraceEvent:
    """Single local WM trace event."""

    event_type: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurvedLocalTrace:
    """Local trace schema for Curved Resonant WM Core.

    Captures:
    - selected slots
    - activation route
    - curvature state
    - geometry map
    - depth contribution
    - confidence
    - novelty
    - disagreement
    - write decision
    - PAAMA-X metadata
    """

    trace_id: str = field(default_factory=lambda: f"wmtrace-{uuid.uuid4().hex}")
    operation: str = "unknown"
    selected_slots: List[str] = field(default_factory=list)
    activation_route: List[Dict[str, Any]] = field(default_factory=list)
    curvature_state: Dict[str, Any] = field(default_factory=dict)
    geometry_map: Optional[str] = None
    depth_contribution: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    novelty: float = 0.0
    disagreement: float = 0.0
    write_decision: str = "not_applicable"
    write_proposal_id: Optional[str] = None
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[CurvedTraceEvent] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())

    def add_event(self, event_type: str, message: str, **metadata: Any) -> None:
        self.events.append(CurvedTraceEvent(event_type=event_type, message=message, metadata=metadata))

    def set_write_decision(self, decision: str, proposal_id: Optional[str] = None, **metadata: Any) -> None:
        self.write_decision = decision
        self.write_proposal_id = proposal_id
        self.add_event("write_decision", decision, proposal_id=proposal_id, **metadata)

    def merge_paamax(self, **metadata: Any) -> None:
        self.paamax_metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["events"] = [e.to_dict() for e in self.events]
        return out


class CurvedLocalTraceBuilder:
    """Small helper for constructing serializable local WM traces."""

    def __init__(self, operation: str):
        self.trace = CurvedLocalTrace(operation=operation)

    def from_resonance_trace(self, resonance_trace: Optional[Dict[str, Any]]) -> "CurvedLocalTraceBuilder":
        if not resonance_trace:
            return self

        self.trace.confidence = float(resonance_trace.get("confidence_score", self.trace.confidence) or 0.0)
        self.trace.novelty = float(resonance_trace.get("novelty_score", self.trace.novelty) or 0.0)
        self.trace.merge_paamax(**resonance_trace.get("paamax_metadata", {}))

        step_traces = resonance_trace.get("step_traces", []) or []
        for step in step_traces:
            self.trace.activation_route.append({
                "step": step.get("step"),
                "top_indices": step.get("top_indices"),
                "top_scores": step.get("top_scores"),
                "activation_entropy": step.get("activation_entropy"),
                "delta_norm": step.get("delta_norm"),
            })

        inner_trace = resonance_trace.get("inner_trace") or {}
        selected = inner_trace.get("top_indices")
        if selected and isinstance(selected, list):
            self.trace.selected_slots = [f"inner_slot_{idx}" for row in selected for idx in (row if isinstance(row, list) else [row])]

        self.trace.add_event("resonance_trace_merged", "Merged resonance trace into local WM trace")
        return self

    def with_geometry_map(self, geometry_map: Optional[str]) -> "CurvedLocalTraceBuilder":
        self.trace.geometry_map = geometry_map
        return self

    def with_curvature_state(self, curvature_state: Dict[str, Any]) -> "CurvedLocalTraceBuilder":
        self.trace.curvature_state = dict(curvature_state)
        return self

    def with_depth_contribution(self, depth_contribution: Dict[str, float]) -> "CurvedLocalTraceBuilder":
        self.trace.depth_contribution = dict(depth_contribution)
        return self

    def with_disagreement(self, disagreement: float) -> "CurvedLocalTraceBuilder":
        self.trace.disagreement = float(disagreement)
        return self

    def build(self) -> CurvedLocalTrace:
        if "trace_type" not in self.trace.paamax_metadata:
            self.trace.paamax_metadata["trace_type"] = "curved_local_trace"
        return self.trace
