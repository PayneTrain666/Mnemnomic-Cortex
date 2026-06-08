"""Long-term-memory extension package.

Exports:
  - HG episodic LTM (shared-slot episodic wrapper)
  - Spatial LTM + geometry-aware MANN reconstruction subsystem
  - Dual transformer policy (bank inherit + fixed advantage stacks)
  - Geometry/manifold utilities and transformer building blocks
"""

from .hg_episodic_ltm import EpisodeRecord, HGEpisodicLTM
from .transformer_policy import DualTransformerPolicy
from .config import (
    DEFAULT_CGMN_DEPTH_CHART,
    DEFAULT_CURVED_DEPTH_CHART,
    DEFAULT_HG_DEPTH_CHART,
    DEFAULT_MANN_DEPTH_CHART,
    DEFAULT_PROCEDURAL_DEPTH_CHART,
    DEFAULT_SPATIAL_DEPTH_CHART,
    SpatialLtmMannConfig,
    default_config,
)
from .shared_memory import SharedValueStore, SharedWriteTrace
from .geometry_keys import DepthRouter, GeometryKeyProjector
from .memory_bank import BankReadTrace, GeometryMemoryBank
from .ltm_system import LTMReadResult, LTMSubsystem, TripleHybridLTM
from .mann_reasoner import MANNReadResult, MANNReasoner
from .working_memory import WorkingMemory
from .sensory_buffer import SensoryContextBuffer, SensoryToken, ContextToken
from .mnemonic_cortex import EnhancedSpatialMnemonicCortex
from .transformer_utils import TransformerBlock, TransformerStack
from .reasoning_stack import ReasoningStack
from .manifold_utils import (
    conformal_scale,
    euclid_dist,
    hyperbolic_dist,
    metric_distance,
    quaternion_distance,
    spatial_dist,
    spherical_dist,
    torus_dist,
    wrap_angles,
)
from .manifold_ops import exp_map, frechet_mean, log_map, parallel_transport

__all__ = [
    "EpisodeRecord",
    "HGEpisodicLTM",
    "DualTransformerPolicy",
    "SpatialLtmMannConfig",
    "default_config",
    "DEFAULT_MANN_DEPTH_CHART",
    "DEFAULT_HG_DEPTH_CHART",
    "DEFAULT_CGMN_DEPTH_CHART",
    "DEFAULT_CURVED_DEPTH_CHART",
    "DEFAULT_PROCEDURAL_DEPTH_CHART",
    "DEFAULT_SPATIAL_DEPTH_CHART",
    "SharedValueStore",
    "SharedWriteTrace",
    "DepthRouter",
    "GeometryKeyProjector",
    "GeometryMemoryBank",
    "BankReadTrace",
    "LTMReadResult",
    "LTMSubsystem",
    "TripleHybridLTM",
    "MANNReadResult",
    "MANNReasoner",
    "WorkingMemory",
    "SensoryContextBuffer",
    "SensoryToken",
    "ContextToken",
    "EnhancedSpatialMnemonicCortex",
    "TransformerBlock",
    "TransformerStack",
    "ReasoningStack",
    "wrap_angles",
    "euclid_dist",
    "spherical_dist",
    "torus_dist",
    "hyperbolic_dist",
    "spatial_dist",
    "quaternion_distance",
    "conformal_scale",
    "metric_distance",
    "log_map",
    "exp_map",
    "parallel_transport",
    "frechet_mean",
]
