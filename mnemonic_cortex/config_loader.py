"""Config loader for YAML-based cortex configuration."""
import yaml
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import torch


@dataclass
class AHGConfig:
    """Architecture Hypergraph config."""
    ask_on_uncertain: bool = False
    
    def __post_init__(self):
        if isinstance(self.ask_on_uncertain, str):
            self.ask_on_uncertain = self.ask_on_uncertain.lower() in ('true', '1', 'yes')


@dataclass
class DistillConfig:
    """Distillation config."""
    enabled: bool = False
    student_domains: list = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.enabled, str):
            self.enabled = self.enabled.lower() in ('true', '1', 'yes')
        if self.student_domains and isinstance(self.student_domains, str):
            self.student_domains = [self.student_domains]


@dataclass
class StoreConfig:
    """Store configuration."""
    senses: int = 4
    conformal_b: float = 0.03
    geometries: Dict[str, float] = field(default_factory=dict)


@dataclass
class GlobalConfig:
    """Global configuration."""
    stores: Dict[str, StoreConfig] = field(default_factory=dict)
    ahg: AHGConfig = field(default_factory=AHGConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)


@dataclass
class CortexConfig:
    """Cortex model configuration."""
    input_dim: int = 32
    output_dim: int = 32
    wm_slots: int = 7
    fusion: str = 'weighted'
    hgm_enabled: bool = False
    
    def to_dict(self):
        return asdict(self)


@dataclass
class FeaturesConfig:
    """Features configuration."""
    hgm_enabled: Optional[bool] = None
    reasoning_bridge_enabled: Optional[bool] = None


@dataclass
class UnifiedConfig:
    """Unified configuration combining cortex, features, and global configs."""
    cortex: CortexConfig = field(default_factory=CortexConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)


def load_yaml_config(path: str) -> GlobalConfig:
    """Load legacy YAML config (backward compatible)."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Parse stores
    stores = {}
    for store_name, store_data in (data.get('stores') or {}).items():
        if store_data:
            geometries = store_data.get('geometries', {})
            stores[store_name] = StoreConfig(
                senses=store_data.get('senses', 4),
                conformal_b=store_data.get('conformal_b', 0.03),
                geometries=geometries
            )
        else:
            stores[store_name] = StoreConfig()
    
    # Parse AHG
    ahg_data = data.get('ahg') or {}
    ahg = AHGConfig(ask_on_uncertain=ahg_data.get('ask_on_uncertain', False))
    
    # Parse Distill
    distill_data = data.get('distill') or {}
    distill = DistillConfig(
        enabled=distill_data.get('enabled', False),
        student_domains=distill_data.get('student_domains', [])
    )
    
    return GlobalConfig(stores=stores, ahg=ahg, distill=distill)


def load_unified_yaml_config(path: str) -> UnifiedConfig:
    """Load unified YAML config with cortex, features, and global sections."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Parse cortex config
    cortex_data = data.get('cortex') or {}
    cortex = CortexConfig(
        input_dim=cortex_data.get('input_dim', 32),
        output_dim=cortex_data.get('output_dim', 32),
        wm_slots=cortex_data.get('wm_slots', 7),
        fusion=cortex_data.get('fusion', 'weighted'),
        hgm_enabled=cortex_data.get('hgm_enabled', False)
    )
    
    # Parse features config (can override cortex settings)
    features_data = data.get('features') or {}
    features = FeaturesConfig(
        hgm_enabled=features_data.get('hgm_enabled'),
        reasoning_bridge_enabled=features_data.get('reasoning_bridge_enabled')
    )
    
    # If features override cortex settings
    if features.hgm_enabled is not None:
        cortex.hgm_enabled = features.hgm_enabled
    
    # Parse global config (stores, ahg, distill)
    global_config = load_yaml_config(path)
    
    return UnifiedConfig(cortex=cortex, features=features, global_config=global_config)


def build_cortex_from_yaml(path: str) -> Tuple[Any, UnifiedConfig]:
    """Build cortex model from YAML config file.
    
    Returns:
        Tuple of (model, unified_config)
    """
    from .cortex import EnhancedMnemonicCortex
    
    unified = load_unified_yaml_config(path)
    
    # Build model with cortex config
    model = EnhancedMnemonicCortex(
        input_dim=unified.cortex.input_dim,
        output_dim=unified.cortex.output_dim,
        wm_slots=unified.cortex.wm_slots,
        fusion=unified.cortex.fusion
    )
    
    # Apply feature flags
    if unified.features.hgm_enabled is not None:
        model.enable_hypergraph_manifold_bridge(unified.features.hgm_enabled)
    
    return model, unified
