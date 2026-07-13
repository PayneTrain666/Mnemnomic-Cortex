"""
Plain-language summary
----------------------
What this file is for: Runtime diagnostics snapshot helpers for the model.
How it fits in the system: Developer visibility into health and internal stats.
Status: WORKING (utility)
Important notes for non-coders: Does not change model behavior by itself.
"""

import json
import os
import time
from collections import deque
from typing import Any, Dict, Optional


class ModelDiagnostics:
    """
    Lightweight runtime diagnostics collector.
    Tracks feature usage and key health metrics, with optional JSONL logging.
    """

    def __init__(self, enabled: bool = False, max_events: int = 2000):
        self.enabled = bool(enabled)
        self.max_events = int(max_events)
        self.events = deque(maxlen=self.max_events)
        self.counts: Dict[str, int] = {}
        self.ema: Dict[str, float] = {}
        self.log_path: Optional[str] = None
        self.flush_every = 100
        self._pending = []

    def configure(self, enabled: Optional[bool] = None, log_path: Optional[str] = None, flush_every: int = 100):
        if enabled is not None:
            self.enabled = bool(enabled)
        self.flush_every = int(max(1, flush_every))
        if log_path:
            self.log_path = log_path
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        return self

    def _inc(self, key: str, n: int = 1):
        self.counts[key] = int(self.counts.get(key, 0) + n)

    def _ema(self, key: str, value: float, momentum: float = 0.98):
        prev = self.ema.get(key, value)
        self.ema[key] = float(momentum * prev + (1.0 - momentum) * value)

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        rec = {"ts": time.time(), "event": event, "payload": payload or {}}
        self.events.append(rec)
        self._inc(f"event:{event}")
        self._pending.append(rec)
        if self.log_path and len(self._pending) >= self.flush_every:
            self.flush()

    def record_scalar(self, name: str, value: float, momentum: float = 0.98):
        if not self.enabled:
            return
        self._ema(name, float(value), momentum=momentum)
        self._inc(f"scalar:{name}")

    def flush(self):
        if not self.log_path or not self._pending:
            return None
        with open(self.log_path, "a", encoding="utf-8") as f:
            for rec in self._pending:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        n = len(self._pending)
        self._pending = []
        return n

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "counts": dict(self.counts),
            "ema": dict(self.ema),
            "pending": len(self._pending),
            "events_buffered": len(self.events),
            "log_path": self.log_path,
        }

