"""Backward-compatible import alias for the misspelled module path.

Prefer importing from `shared_slot_arbitrator`.
"""

from .shared_slot_arbitrator import ArbitrationDecision, SharedSlotArbitrator, SlotReadRequest

__all__ = ["ArbitrationDecision", "SharedSlotArbitrator", "SlotReadRequest"]