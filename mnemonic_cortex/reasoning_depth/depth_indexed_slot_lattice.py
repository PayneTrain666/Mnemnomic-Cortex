"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth indexed slot lattice.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import math, torch
from .depth_attention import compute_slot_depth_attention, DepthAttentionResult
from .depth_capacity_metrics import compute_depth_capacity_metrics
from .depth_lattice_config import DepthLatticeConfig
from .depth_lattice_types import DepthReadMode, DepthWriteMode, DepthWriteProposal, QHDepthCode
from .depth_trace import DepthLatticeTrace
from .depth_write_policy import DepthWritePolicy
class DepthIndexedSlotLatticeError(ValueError): pass
@dataclass
class DepthIndexedSlotLattice:
    config: DepthLatticeConfig
    keys: torch.Tensor=field(init=False); values: torch.Tensor=field(init=False); importance: torch.Tensor=field(init=False); confidence: torch.Tensor=field(init=False); usage: torch.Tensor=field(init=False); age: torch.Tensor=field(init=False)
    canonical_slot_ids: Dict[int,str]=field(default_factory=dict); qh_codes: Dict[Tuple[int,int], QHDepthCode]=field(default_factory=dict)
    def __post_init__(self)->None:
        self.keys=torch.randn(self.config.slot_count,self.config.num_depths,self.config.key_dim)*(1.0/math.sqrt(float(self.config.key_dim)))
        self.values=torch.randn(self.config.slot_count,self.config.num_depths,self.config.value_dim)*(1.0/math.sqrt(float(self.config.value_dim)))
        self.importance=torch.zeros(self.config.slot_count,self.config.num_depths); self.confidence=torch.ones(self.config.slot_count,self.config.num_depths); self.usage=torch.zeros(self.config.slot_count,self.config.num_depths); self.age=torch.zeros(self.config.slot_count,self.config.num_depths); self._finite_check_all()
    def _finite_check_all(self)->None:
        if not self.config.finite_checks: return
        for name in ('keys','values','importance','confidence','usage','age'):
            if not torch.isfinite(getattr(self,name)).all(): raise DepthIndexedSlotLatticeError(f'{name} contains NaN/Inf')
    def _validate_query(self, query:torch.Tensor)->torch.Tensor:
        if not isinstance(query, torch.Tensor): raise DepthIndexedSlotLatticeError('query must be a torch.Tensor')
        if query.dim() not in {2,3}: raise DepthIndexedSlotLatticeError('query must be [B,D] or [B,T,D]')
        if query.size(-1)!=self.config.key_dim: raise DepthIndexedSlotLatticeError(f'query last dim must be {self.config.key_dim}')
        if self.config.finite_checks and not torch.isfinite(query).all(): raise DepthIndexedSlotLatticeError('query contains NaN/Inf')
        return query
    def read(self, query:torch.Tensor, *, read_mode:DepthReadMode=DepthReadMode.TOP_K, return_trace:bool=False):
        query=self._validate_query(query)
        attention=compute_slot_depth_attention(query,self.keys,importance=self.importance,confidence=self.confidence,read_top_k_slots=self.config.read_top_k_slots if read_mode != DepthReadMode.EXPLOSIVE else min(self.config.max_explosive_slots,self.config.slot_count),read_top_k_depths=self.config.read_top_k_depths,eps=self.config.epsilon)
        value=self._read_from_attention(attention); trace=self._make_read_trace(attention, read_mode)
        return (value, attention, trace.to_dict()) if return_trace else value
    def _read_from_attention(self, attention:DepthAttentionResult)->torch.Tensor:
        weights=attention.slot_attention.unsqueeze(-1)*attention.depth_attention; weights=weights/weights.sum(dim=(1,2),keepdim=True).clamp_min(self.config.epsilon)
        value=torch.einsum('bsz,szv->bv', weights, self.values)
        if self.config.finite_checks and not torch.isfinite(value).all(): raise DepthIndexedSlotLatticeError('read value contains NaN/Inf')
        with torch.no_grad(): self.usage += weights.detach().mean(dim=0)
        return value
    def _make_read_trace(self, attention:DepthAttentionResult, read_mode:DepthReadMode)->DepthLatticeTrace:
        td=attention.trace_dict(max_items=self.config.max_trace_items); support=float(attention.selected_scores.sum(dim=(-1,-2)).mean().detach().cpu())
        cids=[self.canonical_slot_ids.get(int(i),'') for i in attention.top_slot_indices.detach().cpu().reshape(-1).tolist()[:self.config.max_trace_items] if self.canonical_slot_ids.get(int(i),'')]
        return DepthLatticeTrace('depth_indexed_lattice_trace',self.config.bank_id,'read',td['selected_slots'],td['selected_depths'],td['slot_attention_entropy'],td['depth_entropy'],read_mode.value,None,float(attention.selected_scores.mean().detach().cpu()),max(0.0,1.0-support),support,cids,{'score_shape':list(attention.slot_depth_scores.shape),'slot_attention_shape':list(attention.slot_attention.shape),'depth_attention_shape':list(attention.depth_attention.shape),'capacity_multiplier':self.config.theoretical_capacity_multiplier})
    def propose_write(self, *, slot_index:int, value:torch.Tensor, key:Optional[torch.Tensor]=None, depth_index:Optional[int]=None, depth_indices:Optional[Sequence[int]]=None, mode:DepthWriteMode=DepthWriteMode.SHADOW_ONLY, canonical_slot_id:Optional[str]=None, qh_code:Optional[QHDepthCode]=None)->DepthWriteProposal:
        if not isinstance(value, torch.Tensor): raise DepthIndexedSlotLatticeError('value must be a torch.Tensor')
        if value.size(-1)!=self.config.value_dim: raise DepthIndexedSlotLatticeError(f'value last dim must be {self.config.value_dim}')
        if key is not None and key.size(-1)!=self.config.key_dim: raise DepthIndexedSlotLatticeError(f'key last dim must be {self.config.key_dim}')
        if self.config.finite_checks and not torch.isfinite(value).all(): raise DepthIndexedSlotLatticeError('value contains NaN/Inf')
        if key is not None and self.config.finite_checks and not torch.isfinite(key).all(): raise DepthIndexedSlotLatticeError('key contains NaN/Inf')
        policy=DepthWritePolicy(self.config)
        if mode == DepthWriteMode.SINGLE:
            if depth_index is None: raise DepthIndexedSlotLatticeError('depth_index required for SINGLE write')
            return policy.single_depth(slot_index=slot_index, depth_index=depth_index, value=value, key=key, canonical_slot_id=canonical_slot_id, qh_code=qh_code)
        if mode in {DepthWriteMode.MULTI, DepthWriteMode.REDUNDANT}:
            if not depth_indices: raise DepthIndexedSlotLatticeError('depth_indices required for MULTI/REDUNDANT write')
            return policy.multi_depth(slot_index=slot_index, depth_indices=depth_indices, value=value, key=key, canonical_slot_id=canonical_slot_id)
        return policy.single_depth(slot_index=slot_index, depth_index=7 if depth_index is None else depth_index, value=value, key=key, canonical_slot_id=canonical_slot_id, qh_code=qh_code)
    def commit_write(self, proposal:DepthWriteProposal, *, value:torch.Tensor, key:Optional[torch.Tensor]=None, allow_mutation:bool=False, write_permission:bool=False)->Dict[str,Any]:
        if not allow_mutation or not write_permission: return {'committed':False,'reason':'explicit allow_mutation=True and write_permission=True required','proposal':proposal.to_dict()}
        if proposal.bank_id != self.config.bank_id: raise DepthIndexedSlotLatticeError('proposal bank_id does not match lattice')
        if value.size(-1)!=self.config.value_dim: raise DepthIndexedSlotLatticeError('value dim mismatch')
        if key is not None and key.size(-1)!=self.config.key_dim: raise DepthIndexedSlotLatticeError('key dim mismatch')
        for slot in proposal.slot_indices:
            for depth in proposal.depth_indices:
                self.values[int(slot), int(depth)] = value.reshape(-1,self.config.value_dim)[0].detach()
                if key is not None: self.keys[int(slot), int(depth)] = key.reshape(-1,self.config.key_dim)[0].detach()
                if proposal.canonical_slot_id: self.canonical_slot_ids[int(slot)] = proposal.canonical_slot_id
                if proposal.qh_code: self.qh_codes[(int(slot), int(depth))] = proposal.qh_code
        self._finite_check_all(); return {'committed':True,'proposal':proposal.to_dict()}
    def capacity_metrics(self)->Dict[str,Any]: return compute_depth_capacity_metrics(self.config, self.usage).to_dict()
