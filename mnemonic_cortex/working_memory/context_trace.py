"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context trace.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class ContextMountTrace:
    selected_map: str
    selection_reason: str
    geometry_by_depth: List[str]
    depth_weights: List[float]
    mount_strategy: str
    paamax_policy_tags: List[str] = field(default_factory=list)
    stability: Dict[str, Any] = field(default_factory=dict)
    selector_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
