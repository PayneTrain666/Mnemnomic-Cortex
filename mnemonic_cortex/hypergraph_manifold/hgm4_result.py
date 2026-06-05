"""HGM-4 result dataclasses for trace-safe QDT/WM bridge planning.

HGM-4 is intentionally dry-run/read-only by default. It produces bridge
contracts, shared slot-lattice hook records, and execution previews without
mutating QDT/WM internals or executing robotics actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import DepthLayer, GeometryType
from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class QDTWMBridgeOptions:
    """Options controlling HGM-4 bridge planning.

    Defaults are deliberately safe: dry-run enabled, write intent disabled,
    and write-preview blocking enabled unless explicitly overridden.
    """

    dry_run: bool = True
    write_intent: bool = False
    allow_write_preview: bool = False
    max_hook_count: int = 128
    max_payload_count: int = 256
    max_payload_summary_length: int = 512
    default_depth_layer: DepthLayer = DepthLayer.D5_PROCEDURAL
    default_geometry_type: GeometryType = GeometryType.PRODUCT
    expected_module_paths: Tuple[str, ...] = (
        "mnemonic_cortex.working_memory",
        "mnemonic_cortex.working_memory.qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack",
    )
    expected_filesystem_paths: Tuple[str, ...] = (
        "mnemonic_cortex/working_memory",
    )
    qspin_placeholder_prefix: str = "qspin_placeholder"

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_depth_layer", DepthLayer.coerce(self.default_depth_layer))
        object.__setattr__(self, "default_geometry_type", GeometryType.coerce(self.default_geometry_type))
        object.__setattr__(self, "expected_module_paths", tuple(self.expected_module_paths or tuple()))
        object.__setattr__(self, "expected_filesystem_paths", tuple(self.expected_filesystem_paths or tuple()))
        if int(self.max_hook_count) <= 0:
            raise ValueError("max_hook_count must be positive")
        if int(self.max_payload_count) <= 0:
            raise ValueError("max_payload_count must be positive")
        if int(self.max_payload_summary_length) <= 0:
            raise ValueError("max_payload_summary_length must be positive")


@dataclass(frozen=True)
class BridgeAdapterStatus:
    adapter_id: str
    available: bool
    adapter_type: str
    reason: str
    detected_paths: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detected_paths", tuple(self.detected_paths or tuple()))


@dataclass(frozen=True)
class HGMBridgePayload:
    payload_id: str
    source_type: str
    source_id: str
    depth_layer: DepthLayer
    geometry_type: GeometryType
    content_summary: str
    qspin_signature_id: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))
        object.__setattr__(self, "geometry_type", GeometryType.coerce(self.geometry_type))


@dataclass(frozen=True)
class BridgePayloadBuildResult:
    payload: Optional[HGMBridgePayload]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SharedSlotLatticeHook:
    hook_id: str
    source_record_id: str
    target_slot_id: str
    depth_layer: DepthLayer
    geometry_type: GeometryType
    qspin_signature_id: str
    dry_run: bool
    write_intent: bool
    confidence: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))
        object.__setattr__(self, "geometry_type", GeometryType.coerce(self.geometry_type))


@dataclass(frozen=True)
class SharedSlotLatticeHookResult:
    hooks: Tuple[SharedSlotLatticeHook, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hooks", tuple(self.hooks or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class TraceSafeMemoryPlan:
    plan_id: str
    bridge_payloads: Tuple[HGMBridgePayload, ...]
    slot_hooks: Tuple[SharedSlotLatticeHook, ...]
    adapter_status: BridgeAdapterStatus
    dry_run: bool
    write_intent: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bridge_payloads", tuple(self.bridge_payloads or tuple()))
        object.__setattr__(self, "slot_hooks", tuple(self.slot_hooks or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class BridgeExecutionPreview:
    preview_id: str
    plan_id: str
    allowed: bool
    blocked_reason: str
    planned_operations: Tuple[str, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_operations", tuple(self.planned_operations or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM4BridgeResult:
    adapter_status: BridgeAdapterStatus
    memory_plan: TraceSafeMemoryPlan
    execution_preview: BridgeExecutionPreview
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
