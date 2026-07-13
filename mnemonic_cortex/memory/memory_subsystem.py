"""
Plain-language summary
----------------------
What this file is for: Shared-slot memory subsystem module: memory subsystem.
How it fits in the system: Manages shared memory slots that multiple systems can read/write under rules.
Status: OPT-IN
Important notes for non-coders: Not always enabled in standard capacity profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_lifecycle_manager import MemoryLifecycleManager
from .memory_read_engine import MemoryReadEngine
from .memory_update_engine import MemoryUpdateEngine
from .memory_write_engine import MemoryWriteEngine
from .shared_slot_allocator import SharedSlotAllocator
from .shared_slot_arbitrator import SharedSlotArbitrator
from .shared_slot_retention import SharedSlotRetention
from .shared_slot_store import SharedSlotStore


@dataclass
class SharedMemorySubsystem:
    store: SharedSlotStore
    allocator: SharedSlotAllocator
    arbitrator: SharedSlotArbitrator
    retention: SharedSlotRetention
    read_engine: MemoryReadEngine
    write_engine: MemoryWriteEngine
    update_engine: MemoryUpdateEngine
    lifecycle_manager: MemoryLifecycleManager


def build_shared_memory_subsystem(
    *,
    store: SharedSlotStore,
    geometry_runtime: Any = None,
    reranker: Any = None,
    truth_runtime: Any = None,
    overwrite_threshold: float | None = None,
    merge_threshold: float | None = None,
    quarantine_interference_threshold: float | None = None,
    contradiction_split_threshold: int | None = None,
) -> SharedMemorySubsystem:
    allocator = SharedSlotAllocator(store=store)
    arbitrator_kwargs: dict[str, float] = {}
    if overwrite_threshold is not None:
        arbitrator_kwargs["overwrite_threshold"] = float(overwrite_threshold)
    if merge_threshold is not None:
        arbitrator_kwargs["merge_threshold"] = float(merge_threshold)
    if quarantine_interference_threshold is not None:
        arbitrator_kwargs["quarantine_interference_threshold"] = float(quarantine_interference_threshold)
    arbitrator = SharedSlotArbitrator(store=store, **arbitrator_kwargs)
    retention = SharedSlotRetention(store=store)
    read_engine = MemoryReadEngine(store=store, geometry_runtime=geometry_runtime, reranker=reranker)
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    update_engine_kwargs: dict[str, int] = {}
    if contradiction_split_threshold is not None:
        update_engine_kwargs["contradiction_split_threshold"] = int(contradiction_split_threshold)
    update_engine = MemoryUpdateEngine(
        store=store,
        write_engine=write_engine,
        arbitrator=arbitrator,
        **update_engine_kwargs,
    )
    lifecycle_manager = MemoryLifecycleManager(
        store=store,
        retention=retention,
        truth_runtime=truth_runtime,
    )
    return SharedMemorySubsystem(
        store=store,
        allocator=allocator,
        arbitrator=arbitrator,
        retention=retention,
        read_engine=read_engine,
        write_engine=write_engine,
        update_engine=update_engine,
        lifecycle_manager=lifecycle_manager,
    )
