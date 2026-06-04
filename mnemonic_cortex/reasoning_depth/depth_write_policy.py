from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import torch
from .depth_lattice_config import DepthLatticeConfig
from .depth_lattice_types import DepthRole, DepthWriteMode, DepthWriteProposal, QHDepthCode
class DepthWritePolicyError(ValueError): pass
_ROLE_TO_DEPTH={DepthRole.Z0_CORE_IDENTITY:0,DepthRole.Z1_SEMANTIC_INVARIANT:1,DepthRole.Z2_STRUCTURAL_RELATION:2,DepthRole.Z3_CONTEXTUAL_BINDING:3,DepthRole.Z4_REASONING_TRANSFORM:4,DepthRole.Z5_TEMPORAL_EPISODE:5,DepthRole.Z6_EXPERIMENTAL_HYPOTHESIS:6,DepthRole.Z7_VOLATILE_TRACE:7}
@dataclass
class DepthWritePolicy:
    config: DepthLatticeConfig
    def _check_slot(self, slot_index:int)->None:
        if not (0 <= int(slot_index) < self.config.slot_count): raise DepthWritePolicyError('slot_index out of range')
    def _check_depth(self, depth_index:int)->None:
        if not (0 <= int(depth_index) < self.config.num_depths): raise DepthWritePolicyError('depth_index out of range')
    def single_depth(self, *, slot_index:int, depth_index:int, value:torch.Tensor, key:Optional[torch.Tensor]=None, canonical_slot_id:Optional[str]=None, qh_code:Optional[QHDepthCode]=None) -> DepthWriteProposal:
        self._check_slot(slot_index); self._check_depth(depth_index)
        return DepthWriteProposal(self.config.bank_id,[int(slot_index)],[int(depth_index)],DepthWriteMode.SINGLE,list(value.shape),list(key.shape) if key is not None else None,canonical_slot_id,qh_code,{'shadow_only':True})
    def multi_depth(self, *, slot_index:int, depth_indices:Sequence[int], value:torch.Tensor, key:Optional[torch.Tensor]=None, canonical_slot_id:Optional[str]=None) -> DepthWriteProposal:
        self._check_slot(slot_index)
        if not depth_indices: raise DepthWritePolicyError('depth_indices cannot be empty')
        for d in depth_indices: self._check_depth(int(d))
        return DepthWriteProposal(self.config.bank_id,[int(slot_index)],[int(d) for d in depth_indices],DepthWriteMode.MULTI,list(value.shape),list(key.shape) if key is not None else None,canonical_slot_id,None,{'shadow_only':True})
    def role_coded(self, *, slot_index:int, role_values:Dict[DepthRole, torch.Tensor], canonical_slot_id:Optional[str]=None) -> DepthWriteProposal:
        self._check_slot(slot_index)
        if not role_values: raise DepthWritePolicyError('role_values cannot be empty')
        depths=[]; shapes={}
        for role,tensor in role_values.items():
            if role not in _ROLE_TO_DEPTH: raise DepthWritePolicyError(f'unknown role: {role}')
            depth=_ROLE_TO_DEPTH[role]; self._check_depth(depth); depths.append(depth); shapes[role.value]=list(tensor.shape)
        first=next(iter(role_values.values()))
        return DepthWriteProposal(self.config.bank_id,[int(slot_index)],sorted(set(depths)),DepthWriteMode.ROLE_CODED,list(first.shape),None,canonical_slot_id,None,{'role_value_shapes':shapes,'shadow_only':True})
