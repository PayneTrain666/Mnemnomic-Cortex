"""
Plain-language summary
----------------------
What this file is for: Records when lightbulb / recall events fire.
How it fits in the system: Diagnostics and analysis aid for recall behavior.
Status: WORKING (utility)
Important notes for non-coders: Does not change memory content by itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class LightbulbEvent:
    source: str
    intensity: float
    step: Optional[int] = None


class LightbulbEventLogger:
    """Lightweight in-memory event log for cross-memory lightbulb moments."""

    def __init__(self, max_events: int = 256):
        self.max_events = int(max_events)
        self._events: Deque[LightbulbEvent] = deque(maxlen=self.max_events)

    def log(
        self,
        *,
        source: str,
        intensity: float,
        step: Optional[int] = None,
    ) -> None:
        self._events.append(
            LightbulbEvent(source=str(source), intensity=float(intensity), step=step)
        )

    def recent(self, n: int = 16) -> List[Dict]:
        out = []
        for ev in list(self._events)[-int(n) :]:
            out.append(
                {
                    "source": ev.source,
                    "intensity": ev.intensity,
                    "step": ev.step,
                }
            )
        return out

    def last_intensity(self, source: Optional[str] = None) -> float:
        for ev in reversed(self._events):
            if source is None or ev.source == source:
                return float(ev.intensity)
        return 0.0
