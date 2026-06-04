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
    overwrite_threshold: float = 0.35,
    merge_threshold: float = 0.65,
    quarantine_interference_threshold: float = 0.85,
    contradiction_split_threshold: int = 3,
) -> SharedMemorySubsystem:
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(
        store=store,
        overwrite_threshold=overwrite_threshold,
        merge_threshold=merge_threshold,
        quarantine_interference_threshold=quarantine_interference_threshold,
    )
    retention = SharedSlotRetention(store=store)
    read_engine = MemoryReadEngine(
        store=store,
        geometry_runtime=geometry_runtime,
        reranker=reranker,
        arbitrator=arbitrator,
    )
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    update_engine = MemoryUpdateEngine(
        store=store,
        write_engine=write_engine,
        arbitrator=arbitrator,
        contradiction_split_threshold=contradiction_split_threshold,
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
