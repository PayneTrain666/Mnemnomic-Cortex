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
from .depth_indexed_slot_lattice import DepthIndexedSlotLattice, DepthIndexedSlotLatticeError
from .shared_depth_slot_registry import SharedDepthSlotRecord, SharedDepthSlotRegistry, SharedDepthSlotRegistryError
__all__ = ['DepthLatticeConfig','DepthLatticeConfigError','BankKind','CANONICAL_DEPTH_ROLES','DepthCellRef','DepthReadMode','DepthRole','DepthWriteMode','DepthWriteProposal','QHDepthCode','DepthEntropyError','depth_entropy','validate_probability_tensor','DepthLatticeTrace','DepthAttentionError','DepthAttentionResult','compute_slot_depth_attention','DepthWritePolicy','DepthWritePolicyError','DepthCapacityMetrics','compute_depth_capacity_metrics','DepthIndexedSlotLattice','DepthIndexedSlotLatticeError','SharedDepthSlotRecord','SharedDepthSlotRegistry','SharedDepthSlotRegistryError']

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
