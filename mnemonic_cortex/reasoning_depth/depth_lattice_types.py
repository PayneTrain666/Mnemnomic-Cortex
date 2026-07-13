"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth lattice types.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib, time

class DepthRole(str, Enum):
    Z0_CORE_IDENTITY = 'core_identity'
    Z1_SEMANTIC_INVARIANT = 'semantic_invariant'
    Z2_STRUCTURAL_RELATION = 'structural_relation'
    Z3_CONTEXTUAL_BINDING = 'contextual_binding'
    Z4_REASONING_TRANSFORM = 'reasoning_transform'
    Z5_TEMPORAL_EPISODE = 'temporal_episode'
    Z6_EXPERIMENTAL_HYPOTHESIS = 'experimental_hypothesis'
    Z7_VOLATILE_TRACE = 'volatile_trace'

CANONICAL_DEPTH_ROLES: List[DepthRole] = [
    DepthRole.Z0_CORE_IDENTITY, DepthRole.Z1_SEMANTIC_INVARIANT,
    DepthRole.Z2_STRUCTURAL_RELATION, DepthRole.Z3_CONTEXTUAL_BINDING,
    DepthRole.Z4_REASONING_TRANSFORM, DepthRole.Z5_TEMPORAL_EPISODE,
    DepthRole.Z6_EXPERIMENTAL_HYPOTHESIS, DepthRole.Z7_VOLATILE_TRACE,
]
class DepthReadMode(str, Enum):
    SINGLE='single'; TOP_K='top_k'; SOFT='soft'; ROLE_MASKED='role_masked'; EXPLOSIVE='explosive'
class DepthWriteMode(str, Enum):
    SINGLE='single'; MULTI='multi'; ROLE_CODED='role_coded'; REDUNDANT='redundant'; SHADOW_ONLY='shadow_only'
class BankKind(str, Enum):
    WM='wm'; MANN='mann'; LTM='ltm'; SPCP='spcp'; GENERIC='generic'

def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, dict): return {str(k): _safe_jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe_jsonable(v) for v in value]
    if isinstance(value, set): return sorted(_safe_jsonable(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return repr(value)

def stable_hash_payload(prefix: str, payload: Dict[str, Any]) -> str:
    raw = repr(_safe_jsonable(payload)).encode('utf-8')
    return f'{prefix}-{hashlib.sha256(raw).hexdigest()[:16]}'

@dataclass(frozen=True)
class DepthCellRef:
    bank_id: str; slot_index: int; depth_index: int; canonical_slot_id: Optional[str]=None; role: Optional[DepthRole]=None
    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable({'bank_id':self.bank_id,'slot_index':self.slot_index,'depth_index':self.depth_index,'canonical_slot_id':self.canonical_slot_id,'role':self.role})

@dataclass(frozen=True)
class QHDepthCode:
    depth_code: str; bank_code: str; geometry_code: str; triplet_code: str; memory_type_code: str; task_mode_code: str
    def to_dict(self) -> Dict[str, Any]:
        return {'depth_code':self.depth_code,'bank_code':self.bank_code,'geometry_code':self.geometry_code,'triplet_code':self.triplet_code,'memory_type_code':self.memory_type_code,'task_mode_code':self.task_mode_code,'fake_quantum_hardware_claim':False}

@dataclass(frozen=True)
class DepthWriteProposal:
    bank_id: str; slot_indices: List[int]; depth_indices: List[int]; mode: DepthWriteMode; value_shape: List[int]
    key_shape: Optional[List[int]]=None; canonical_slot_id: Optional[str]=None; qh_code: Optional[QHDepthCode]=None
    metadata: Dict[str, Any]=field(default_factory=dict); proposal_id: Optional[str]=None; created_at: float=field(default_factory=lambda: time.time())
    def __post_init__(self) -> None:
        if self.proposal_id is None:
            object.__setattr__(self, 'proposal_id', stable_hash_payload('depthwrite', {'bank_id':self.bank_id,'slot_indices':self.slot_indices,'depth_indices':self.depth_indices,'mode':self.mode,'value_shape':self.value_shape,'key_shape':self.key_shape,'canonical_slot_id':self.canonical_slot_id,'qh_code':self.qh_code.to_dict() if self.qh_code else None,'metadata':self.metadata}))
    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable({'proposal_id':self.proposal_id,'bank_id':self.bank_id,'slot_indices':self.slot_indices,'depth_indices':self.depth_indices,'mode':self.mode,'value_shape':self.value_shape,'key_shape':self.key_shape,'canonical_slot_id':self.canonical_slot_id,'qh_code':self.qh_code.to_dict() if self.qh_code else None,'metadata':{'shadow_only_by_default':True,'no_permanent_mutation_without_permission':True,**self.metadata},'created_at':self.created_at})
