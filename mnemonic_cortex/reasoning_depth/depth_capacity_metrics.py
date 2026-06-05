from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from .depth_lattice_config import DepthLatticeConfig
@dataclass(frozen=True)
class DepthCapacityMetrics:
    raw_slots:int; depth_layers:int; effective_subslots:int; theoretical_capacity_multiplier:int; active_depth_usage_histogram:List[float]
    def to_dict(self) -> Dict[str, object]: return {'raw_slots':self.raw_slots,'depth_layers':self.depth_layers,'effective_subslots':self.effective_subslots,'theoretical_capacity_multiplier':self.theoretical_capacity_multiplier,'active_depth_usage_histogram':list(self.active_depth_usage_histogram)}
def compute_depth_capacity_metrics(config: DepthLatticeConfig, usage: Optional[torch.Tensor]=None) -> DepthCapacityMetrics:
    hist=[0.0 for _ in range(config.num_depths)]
    if usage is not None:
        if usage.shape != (config.slot_count, config.num_depths): raise ValueError(f'usage must have shape {(config.slot_count, config.num_depths)}')
        if not torch.isfinite(usage).all(): raise ValueError('usage contains NaN/Inf')
        totals=usage.sum(dim=0); denom=totals.sum().clamp_min(config.epsilon); hist=(totals/denom).detach().cpu().tolist()
    return DepthCapacityMetrics(config.slot_count, config.num_depths, config.slot_count*config.num_depths, config.num_depths, hist)
