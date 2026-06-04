from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _safe_jsonable(value.to_dict())
    return repr(value)


@dataclass
class ReasoningTraceEvent:
    """One bounded event in the reasoning orchestration trace."""

    stage: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"reason_evt_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stage": self.stage,
            "message": self.message,
            "payload": _safe_jsonable(self.payload),
            "created_at": self.created_at,
        }


@dataclass
class ReasoningOrchestrationTrace:
    """Serialization-safe trace for a WM→MANN→LTM reasoning pass."""

    trace_id: str = field(default_factory=lambda: f"reason_trace_{uuid.uuid4().hex[:16]}")
    stage: str = "REASON-2A"
    events: List[ReasoningTraceEvent] = field(default_factory=list)
    max_events: int = 64
    metadata: Dict[str, Any] = field(default_factory=dict)
    _budget_event_added: bool = False

    def add_event(self, stage: str, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if len(self.events) >= self.max_events:
            if not self._budget_event_added and self.events:
                self.events[-1] = ReasoningTraceEvent(
                    stage="trace_budget",
                    message="trace event budget reached; additional events suppressed",
                    payload={"max_events": self.max_events},
                )
                self._budget_event_added = True
            return
        self.events.append(ReasoningTraceEvent(stage=stage, message=message, payload=payload or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": self.stage,
            "events": [event.to_dict() for event in self.events],
            "max_events": self.max_events,
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "trace_governance": True,
                "confidence_disagreement_hooks": True,
                "write_permission_required_for_commit": True,
                "conflict_quarantine_hooks": True,
                "audit_metadata": True,
            },
            "safety": {
                "bounded_trace": True,
                "permanent_memory_store_mutation": False,
                "destructive_replacement": False,
            },
        }


def reasoning_orchestration_trace_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_orchestration_trace",
        "stage": "REASON-2A",
        "bounded_trace": True,
        "serializable": True,
        "paamax_metadata": True,
        "default_memory_mutation": False,
    }
