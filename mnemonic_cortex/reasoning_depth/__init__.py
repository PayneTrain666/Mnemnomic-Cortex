"""Reasoning-depth lattice for Mnemonic Cortex.
REASON-1A implements additive slots × 8 depth capacity primitives.
"""
from .depth_lattice_config import DepthLatticeConfig, DepthLatticeConfigError
from .depth_lattice_types import BankKind, CANONICAL_DEPTH_ROLES, DepthCellRef, DepthReadMode, DepthRole, DepthWriteMode, DepthWriteProposal, QHDepthCode
from .depth_entropy import DepthEntropyError, depth_entropy, validate_probability_tensor
from .depth_trace import DepthLatticeTrace
from .depth_attention import DepthAttentionError, DepthAttentionResult, compute_slot_depth_attention
from .depth_write_policy import DepthWritePolicy, DepthWritePolicyError
from .depth_capacity_metrics import DepthCapacityMetrics, compute_depth_capacity_metrics
from .depth_capacity_validation import depth_capacity_validation_contract
from .depth_integration_readiness import depth_integration_readiness_contract
from .depth_lattice_benchmarks import depth_lattice_benchmark_contract
from .depth_indexed_slot_lattice import DepthIndexedSlotLattice, DepthIndexedSlotLatticeError
from .shared_depth_slot_registry import SharedDepthSlotRecord, SharedDepthSlotRegistry, SharedDepthSlotRegistryError
from .consolidation_gate import consolidation_gate_contract
__all__ = ['DepthLatticeConfig','DepthLatticeConfigError','BankKind','CANONICAL_DEPTH_ROLES','DepthCellRef','DepthReadMode','DepthRole','DepthWriteMode','DepthWriteProposal','QHDepthCode','DepthEntropyError','depth_entropy','validate_probability_tensor','DepthLatticeTrace','DepthAttentionError','DepthAttentionResult','compute_slot_depth_attention','DepthWritePolicy','DepthWritePolicyError','DepthCapacityMetrics','compute_depth_capacity_metrics','depth_capacity_validation_contract','depth_integration_readiness_contract','depth_lattice_benchmark_contract','consolidation_gate_contract','DepthIndexedSlotLattice','DepthIndexedSlotLatticeError','SharedDepthSlotRecord','SharedDepthSlotRegistry','SharedDepthSlotRegistryError']

from .wm_depth_adapter import WMDepthAdapter, WMDepthAdapterConfig, WMDepthAdapterError, wm_depth_contract
from .wm_depth_controller import WMDepthController, WMDepthControllerError, attach_wm_depth_controller, wm_depth_controller_contract

__all__ = list(__all__) + [
    "WMDepthAdapter",
    "WMDepthAdapterConfig",
    "WMDepthAdapterError",
    "wm_depth_contract",
    "WMDepthController",
    "WMDepthControllerError",
    "attach_wm_depth_controller",
    "wm_depth_controller_contract",
]

from .mann_slotkv_depth_bank import (
    MANNSlotKVDepthBankConfig,
    MANNSlotKVDepthBank,
    MANNSlotKVDepthBankError,
    mann_slotkv_depth_contract,
)
from .mann_depth_adapter import (
    MANNDepthAdapterConfig,
    MANNDepthAdapter,
    MANNDepthAdapterError,
    mann_depth_contract,
)

__all__ = list(__all__) + [
    "MANNSlotKVDepthBankConfig",
    "MANNSlotKVDepthBank",
    "MANNSlotKVDepthBankError",
    "mann_slotkv_depth_contract",
    "MANNDepthAdapterConfig",
    "MANNDepthAdapter",
    "MANNDepthAdapterError",
    "mann_depth_contract",
]

from .ltm_depth_banks import (
    LTMDepthBankConfig,
    LTMDepthBank,
    LTMDepthBanks,
    LTMDepthBankError,
    ltm_depth_banks_contract,
)
from .ltm_depth_adapter import (
    LTMDepthAdapterConfig,
    LTMDepthAdapter,
    LTMDepthAdapterError,
    ltm_depth_adapter_contract,
)
from .shared_depth_slot_registry import shared_depth_registry_contract

__all__ = list(__all__) + [
    "LTMDepthBankConfig",
    "LTMDepthBank",
    "LTMDepthBanks",
    "LTMDepthBankError",
    "ltm_depth_banks_contract",
    "LTMDepthAdapterConfig",
    "LTMDepthAdapter",
    "LTMDepthAdapterError",
    "ltm_depth_adapter_contract",
    "shared_depth_registry_contract",
]

from .mann_ltm_shared_slot_geometry import (
    SharedGeometrySlotConfig,
    MANNLTMSharedSlotGeometry,
    MANNLTMSharedSlotGeometryError,
    mann_ltm_shared_slot_geometry_contract,
)
from .parameter_loop_adapter import (
    ParameterLoopAdapterConfig,
    ParameterLoopAdapter,
    ParameterLoopAdapterError,
    parameter_loop_adapter_contract,
)

__all__ = list(__all__) + [
    "SharedGeometrySlotConfig",
    "MANNLTMSharedSlotGeometry",
    "MANNLTMSharedSlotGeometryError",
    "mann_ltm_shared_slot_geometry_contract",
    "ParameterLoopAdapterConfig",
    "ParameterLoopAdapter",
    "ParameterLoopAdapterError",
    "parameter_loop_adapter_contract",
]

from .reasoning_policy_router import ReasoningPolicyRouterConfig
from .evidence_reasoning_pass import EvidenceReasoningConfig
from .counterfactual_reasoning_probe import CounterfactualProbeConfig
from .conflict_aware_consolidation import ConflictAwareConsolidationConfig
from .reasoning_controller import (
    SharedGeometryRoutingPolicyConfig,
    ReasoningControllerConfig,
    ReasoningPassResult,
    ReasoningController,
    reasoning_controller_contract,
)
from .reasoning_controller_api import (
    ReasoningControllerAPIError,
    ReasoningControllerAPIConfig,
    ReasoningControllerAPIResult,
    ReasoningControllerAPI,
    reasoning_controller_api_contract,
)

__all__ = list(__all__) + [
    "ReasoningPolicyRouterConfig",
    "EvidenceReasoningConfig",
    "CounterfactualProbeConfig",
    "ConflictAwareConsolidationConfig",
    "SharedGeometryRoutingPolicyConfig",
    "ReasoningControllerConfig",
    "ReasoningPassResult",
    "ReasoningController",
    "reasoning_controller_contract",
    "ReasoningControllerAPIError",
    "ReasoningControllerAPIConfig",
    "ReasoningControllerAPIResult",
    "ReasoningControllerAPI",
    "reasoning_controller_api_contract",
]

# Backward-compatible lazy export bridge.
# Some tests and downstream callers import advanced symbols directly from
# mnemonic_cortex.reasoning_depth. Resolve those symbols on demand from child
# modules to preserve import-surface compatibility while keeping startup light.
import importlib
import pkgutil
from types import ModuleType
from typing import Optional


_LAZY_MODULE_CACHE: dict[str, Optional[ModuleType]] = {}
_LAZY_MODULE_NAMES = tuple(
    name
    for _, name, _ in pkgutil.iter_modules(__path__)
    if not name.startswith("_")
)


def _load_reasoning_module(module_name: str) -> Optional[ModuleType]:
    cached = _LAZY_MODULE_CACHE.get(module_name, None)
    if module_name in _LAZY_MODULE_CACHE:
        return cached
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception:
        module = None
    _LAZY_MODULE_CACHE[module_name] = module
    return module


def __getattr__(name: str):
    for module_name in _LAZY_MODULE_NAMES:
        module = _load_reasoning_module(module_name)
        if module is None:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            if name not in __all__:
                __all__.append(name)
            return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
