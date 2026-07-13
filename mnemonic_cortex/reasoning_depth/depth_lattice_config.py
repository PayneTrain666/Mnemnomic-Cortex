"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth lattice config.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .depth_lattice_types import BankKind, CANONICAL_DEPTH_ROLES, DepthRole
class DepthLatticeConfigError(ValueError): pass
@dataclass(frozen=True)
class DepthLatticeConfig:
    bank_id: str='reasoning.generic'; bank_kind: BankKind=BankKind.GENERIC; slot_count: int=128; num_depths: int=8; key_dim: int=256; value_dim: int=256
    read_top_k_slots: int=8; read_top_k_depths: int=2; max_explosive_slots: int=64; max_trace_items: int=64; finite_checks: bool=True; no_mutation_by_default: bool=True; epsilon: float=1e-8
    depth_roles: List[DepthRole]=field(default_factory=lambda: list(CANONICAL_DEPTH_ROLES)); depth_role_prior: Optional[Dict[int,float]]=None
    def __post_init__(self) -> None:
        if self.num_depths != 8: raise DepthLatticeConfigError('num_depths must be 8 for the canonical REASON-1A lattice')
        if len(self.depth_roles) != self.num_depths: raise DepthLatticeConfigError('depth_roles length must equal num_depths')
        if self.slot_count <= 0: raise DepthLatticeConfigError('slot_count must be positive')
        if self.key_dim <= 0 or self.value_dim <= 0: raise DepthLatticeConfigError('key_dim and value_dim must be positive')
        if not (1 <= self.read_top_k_slots <= self.slot_count): raise DepthLatticeConfigError('read_top_k_slots must be within [1, slot_count]')
        if not (1 <= self.read_top_k_depths <= self.num_depths): raise DepthLatticeConfigError('read_top_k_depths must be within [1, num_depths]')
        if self.max_explosive_slots < self.read_top_k_slots: raise DepthLatticeConfigError('max_explosive_slots must be >= read_top_k_slots')
        if self.max_trace_items <= 0: raise DepthLatticeConfigError('max_trace_items must be positive')
        if self.epsilon <= 0: raise DepthLatticeConfigError('epsilon must be positive')
    @classmethod
    def wm(cls, slot_count:int=64, key_dim:int=256, value_dim:int=256): return cls(bank_id='wm.depth_lattice', bank_kind=BankKind.WM, slot_count=slot_count, key_dim=key_dim, value_dim=value_dim)
    @classmethod
    def mann(cls, slot_count:int=512, key_dim:int=256, value_dim:int=256): return cls(bank_id='mann.depth_lattice', bank_kind=BankKind.MANN, slot_count=slot_count, key_dim=key_dim, value_dim=value_dim)
    @classmethod
    def ltm(cls, slot_count:int=2048, key_dim:int=256, value_dim:int=256, bank_id:str='ltm.depth_lattice'): return cls(bank_id=bank_id, bank_kind=BankKind.LTM, slot_count=slot_count, key_dim=key_dim, value_dim=value_dim)
    @property
    def effective_subslots(self) -> int: return self.slot_count * self.num_depths
    @property
    def theoretical_capacity_multiplier(self) -> int: return self.num_depths
    def to_dict(self) -> dict:
        return {'bank_id':self.bank_id,'bank_kind':self.bank_kind.value,'slot_count':self.slot_count,'num_depths':self.num_depths,'key_dim':self.key_dim,'value_dim':self.value_dim,'read_top_k_slots':self.read_top_k_slots,'read_top_k_depths':self.read_top_k_depths,'max_explosive_slots':self.max_explosive_slots,'max_trace_items':self.max_trace_items,'finite_checks':self.finite_checks,'no_mutation_by_default':self.no_mutation_by_default,'epsilon':self.epsilon,'depth_roles':[r.value for r in self.depth_roles],'effective_subslots':self.effective_subslots,'theoretical_capacity_multiplier':self.theoretical_capacity_multiplier}
