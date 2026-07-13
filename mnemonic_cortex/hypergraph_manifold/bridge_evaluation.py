"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: bridge evaluation.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-5 bridge evaluation harness wrappers.
"""

from .quality_metrics import (
    evaluate_bridge_payload_quality,
    evaluate_execution_preview_quality,
    evaluate_slot_hook_quality,
)

__all__ = [
    "evaluate_bridge_payload_quality",
    "evaluate_execution_preview_quality",
    "evaluate_slot_hook_quality",
]
