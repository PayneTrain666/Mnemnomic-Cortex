"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth trace.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
from .depth_lattice_types import _safe_jsonable
@dataclass(frozen=True)
class DepthLatticeTrace:
    trace_type: str; bank_id: str; operation: str; selected_slots: List[int]=field(default_factory=list); selected_depths: List[int]=field(default_factory=list)
    slot_attention_entropy: Optional[float]=None; depth_entropy: Optional[float]=None; read_mode: Optional[str]=None; write_mode: Optional[str]=None
    confidence: Optional[float]=None; disagreement: Optional[float]=None; support_mass: Optional[float]=None; canonical_slot_ids: List[str]=field(default_factory=list); metadata: Dict[str, Any]=field(default_factory=dict); created_at: float=field(default_factory=lambda: time.time())
    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable({'trace_type':self.trace_type,'bank_id':self.bank_id,'operation':self.operation,'selected_slots':self.selected_slots,'selected_depths':self.selected_depths,'slot_attention_entropy':self.slot_attention_entropy,'depth_entropy':self.depth_entropy,'read_mode':self.read_mode,'write_mode':self.write_mode,'confidence':self.confidence,'disagreement':self.disagreement,'support_mass':self.support_mass,'canonical_slot_ids':self.canonical_slot_ids,'metadata':self.metadata,'created_at':self.created_at,'paamax_metadata':{'trace_governance':True,'write_permission_required':self.operation.startswith('write'),'write_permission_granted':False,'no_memory_store_mutation':True,'depth_lattice_trace':True}})
