from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


SlotState = Literal["free", "volatile", "provisional", "durable", "deprecated", "quarantined"]

MemorySystemID = Literal[
    "hg_ep_ltm",
    "cgmn_semantic_ltm",
    "spatial_ltm",
    "procedural_ltm",
    "hg_mann",
    "semantic_mann",
    "spatial_mann",
    "spcp_mann",
]

MEMORY_SYSTEM_IDS: List[MemorySystemID] = [
    "hg_ep_ltm",
    "cgmn_semantic_ltm",
    "spatial_ltm",
    "procedural_ltm",
    "hg_mann",
    "semantic_mann",
    "spatial_mann",
    "spcp_mann",
]

SLOT_STATES: List[SlotState] = ["free", "volatile", "provisional", "durable", "deprecated", "quarantined"]

SLOT_STATE_TO_CODE: Dict[SlotState, int] = {
    "free": 0,
    "volatile": 1,
    "provisional": 2,
    "durable": 3,
    "deprecated": 4,
    "quarantined": 5,
}
CODE_TO_SLOT_STATE: Dict[int, SlotState] = {v: k for k, v in SLOT_STATE_TO_CODE.items()}

# Compatibility aliases retained for older call sites.
STATE_TO_CODE: Dict[SlotState, int] = SLOT_STATE_TO_CODE
CODE_TO_STATE: Dict[int, SlotState] = CODE_TO_SLOT_STATE
SYSTEM_TO_CODE: Dict[str, int] = {name: idx for idx, name in enumerate(MEMORY_SYSTEM_IDS)}
CODE_TO_SYSTEM: Dict[int, str] = {v: k for k, v in SYSTEM_TO_CODE.items()}


def validate_slot_state(state: str) -> SlotState:
    if state not in SLOT_STATE_TO_CODE:
        raise ValueError(f"Unknown slot state: {state}")
    return state  # type: ignore[return-value]


def validate_system_id(system_id: str) -> str:
    if system_id not in SYSTEM_TO_CODE:
        raise ValueError(f"Unknown memory system id: {system_id}")
    return system_id


def slot_state_to_code(state: SlotState) -> int:
    return SLOT_STATE_TO_CODE[validate_slot_state(state)]


def code_to_slot_state(code: int) -> SlotState:
    return CODE_TO_SLOT_STATE.get(int(code), "free")


@dataclass
class SlotProvenance:
    source_system: str
    created_step: int
    last_update_step: int
    promotion_count: int = 0
    contradiction_count: int = 0
    checkpoint_version: int = 0
    source_trace_ids: List[str] = field(default_factory=list)


@dataclass
class SlotMetadata:
    slot_id: int
    state: SlotState = "free"
    confidence: float = 0.0
    usage_score: float = 0.0
    age_steps: int = 0
    primary_system_id: str = "unknown"
    allowed_read_systems: List[str] = field(default_factory=list)
    allowed_write_systems: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    memory_type: Optional[str] = None  # episodic | semantic | spatial | procedural
    provenance: Optional[SlotProvenance] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotReadResult:
    slot_ids: List[int]
    scores: List[float]
    values_shape: List[int]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotWriteResult:
    slot_ids: List[int]
    version_counter: int
    state_codes: List[int] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotWriteRequest:
    requester_system: str
    candidate_value_shape: List[int]
    target_slot_ids: Optional[List[int]] = None
    requested_state: SlotState = "volatile"
    requested_memory_type: Optional[str] = None
    confidence: float = 0.5
    semantic_tags: List[str] = field(default_factory=list)
    provenance_trace_ids: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_system_id(self.requester_system)
        validate_slot_state(self.requested_state)
        if not self.candidate_value_shape:
            raise ValueError("candidate_value_shape must not be empty")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")