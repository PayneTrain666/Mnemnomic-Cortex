from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional
import time
import uuid


@dataclass
class TraceItem:
    """Single QDT-WM trace event."""

    stage: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WMTrace:
    """Trace container for QDTWorkingMemory.

    Designed to carry:
    - curved-core traces
    - quaternion-depth traces
    - intra/cross-depth transformer traces
    - depth-addressing traces
    - fusion traces
    - shadow-write/local trace metadata
    - PAAMA-X governance metadata
    """

    trace_id: str = field(default_factory=lambda: f"qdtwm-{uuid.uuid4().hex}")
    operation: str = "unknown"
    items: List[TraceItem] = field(default_factory=list)
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    disagreement: float = 0.0
    created_at: float = field(default_factory=lambda: time.time())

    def add(self, stage: str, message: str, severity: str = "info", **metadata: Any) -> None:
        self.items.append(TraceItem(stage=stage, message=message, severity=severity, metadata=metadata))

    def merge_dict(self, stage: str, payload: Optional[Dict[str, Any]], message: str = "merged_trace") -> None:
        if payload is None:
            self.add(stage, "trace_missing", severity="warning")
            return
        self.add(stage, message, payload=payload)
        meta = payload.get("paamax_metadata") if isinstance(payload, dict) else None
        if isinstance(meta, dict):
            self.paamax_metadata.update(meta)

    def set_paamax(self, **metadata: Any) -> None:
        self.paamax_metadata.update(metadata)

    def update_scores(self, confidence: Optional[float] = None, disagreement: Optional[float] = None) -> None:
        if confidence is not None:
            self.confidence = float(confidence)
        if disagreement is not None:
            self.disagreement = float(disagreement)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["items"] = [i.to_dict() for i in self.items]
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "item_count": len(self.items),
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "paamax_keys": sorted(self.paamax_metadata.keys()),
        }


class WMTraceEmitter:
    """Small trace-emitter utility used by assembled WM modules."""

    def start(self, operation: str, **paamax_metadata: Any) -> WMTrace:
        trace = WMTrace(operation=operation)
        trace.set_paamax(trace_type="qdt_working_memory", **paamax_metadata)
        trace.add("trace", "trace_started", operation=operation)
        return trace

    def finish(self, trace: WMTrace, confidence: float = 1.0, disagreement: float = 0.0) -> WMTrace:
        trace.update_scores(confidence=confidence, disagreement=disagreement)
        trace.add("trace", "trace_finished", confidence=confidence, disagreement=disagreement)
        return trace
