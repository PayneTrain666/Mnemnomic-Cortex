"""
Plain-language summary
----------------------
What this file is for: Shared-slot memory subsystem module: shared slot arbitrater.
How it fits in the system: Manages shared memory slots that multiple systems can read/write under rules.
Status: OPT-IN
Important notes for non-coders: Not always enabled in standard capacity profiles.

Technical notes (original):
Compatibility shim for old import path.

Prefer importing from `shared_slot_arbitrator`.
"""

from .shared_slot_arbitrator import (  # noqa: F401
    ArbitrationDecision,
    SharedSlotArbitrator,
    SlotReadRequest,
)