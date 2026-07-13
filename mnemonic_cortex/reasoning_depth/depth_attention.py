"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth attention.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F
from .depth_entropy import depth_entropy
class DepthAttentionError(ValueError): pass
@dataclass(frozen=True)
class DepthAttentionResult:
    slot_depth_scores: torch.Tensor; slot_attention: torch.Tensor; depth_attention: torch.Tensor; top_slot_indices: torch.Tensor; top_depth_indices: torch.Tensor; selected_scores: torch.Tensor; depth_entropy: torch.Tensor; slot_entropy: torch.Tensor
    def trace_dict(self, max_items:int=64) -> Dict[str, object]:
        return {'selected_slots':[int(x) for x in self.top_slot_indices.detach().cpu().reshape(-1).tolist()[:max_items]], 'selected_depths':[int(x) for x in self.top_depth_indices.detach().cpu().reshape(-1).tolist()[:max_items]], 'slot_attention_entropy':float(self.slot_entropy.mean().detach().cpu()), 'depth_entropy':float(self.depth_entropy.mean().detach().cpu())}
def _validate_query(query: torch.Tensor, key_dim:int) -> torch.Tensor:
    if not isinstance(query, torch.Tensor): raise DepthAttentionError('query must be a torch.Tensor')
    if query.dim()==3: query=query.mean(dim=1)
    if query.dim()!=2: raise DepthAttentionError('query must be [B,D] or [B,T,D]')
    if query.size(-1)!=key_dim: raise DepthAttentionError(f'query dim must be {key_dim}, got {query.size(-1)}')
    if not torch.isfinite(query).all(): raise DepthAttentionError('query contains NaN/Inf')
    return query
def compute_slot_depth_attention(query: torch.Tensor, keys: torch.Tensor, *, importance=None, confidence=None, read_top_k_slots:int=8, read_top_k_depths:int=2, eps:float=1e-8) -> DepthAttentionResult:
    if keys.dim()!=3: raise DepthAttentionError('keys must be [S,Z,K]')
    if not torch.isfinite(keys).all(): raise DepthAttentionError('keys contain NaN/Inf')
    s,z,k=keys.shape; query2=_validate_query(query,k)
    if not (1 <= read_top_k_slots <= s): raise DepthAttentionError('read_top_k_slots out of range')
    if not (1 <= read_top_k_depths <= z): raise DepthAttentionError('read_top_k_depths out of range')
    q=F.normalize(query2, dim=-1, eps=eps); kk=F.normalize(keys, dim=-1, eps=eps)
    scores=torch.einsum('bk,szk->bsz', q, kk)
    if importance is not None:
        if importance.shape!=(s,z): raise DepthAttentionError('importance must be [S,Z]')
        scores=scores+0.05*importance.unsqueeze(0)
    if confidence is not None:
        if confidence.shape!=(s,z): raise DepthAttentionError('confidence must be [S,Z]')
        scores=scores+0.05*confidence.unsqueeze(0)
    if not torch.isfinite(scores).all(): raise DepthAttentionError('slot-depth scores contain NaN/Inf')
    slot_scores=scores.max(dim=-1).values; slot_attention=torch.softmax(slot_scores, dim=-1); depth_probs=torch.softmax(scores, dim=-1)
    _, top_slots=torch.topk(slot_attention, k=read_top_k_slots, dim=-1)
    gathered=torch.gather(depth_probs, dim=1, index=top_slots.unsqueeze(-1).expand(-1,-1,z))
    top_depth_scores, top_depth_indices=torch.topk(gathered, k=read_top_k_depths, dim=-1)
    return DepthAttentionResult(scores, slot_attention, depth_probs, top_slots, top_depth_indices, top_depth_scores, depth_entropy(depth_probs, dim=-1, eps=eps, normalise=True), depth_entropy(slot_attention, dim=-1, eps=eps, normalise=True))
