"""Compatibility shim for old import path.

Prefer importing from `shared_slot_arbitrator`.
"""

from .shared_slot_arbitrator import (  # noqa: F401
    ArbitrationDecision,
    SharedSlotArbitrator,
    SlotReadRequest,
)