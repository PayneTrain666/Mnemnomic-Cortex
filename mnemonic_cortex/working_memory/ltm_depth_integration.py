from __future__ import annotations

from typing import Any, Optional

from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig


class LTMDepthIntegrationError(ValueError):
    """Raised when optional LTM depth integration cannot be installed."""


def install_optional_ltm_depth_adapter(
    target: Any,
    adapter: Optional[LTMDepthAdapter] = None,
    *,
    attr_name: str = "ltm_depth_adapter",
) -> Any:
    """Install an optional disabled-by-default LTM depth adapter.

    This does not replace existing LTM modules and does not activate
    consolidation or memory-store writes.
    """

    if target is None:
        raise LTMDepthIntegrationError("target cannot be None")
    if adapter is None:
        adapter = LTMDepthAdapter(LTMDepthAdapterConfig.disabled())
    if hasattr(target, attr_name):
        existing = getattr(target, attr_name)
        if existing is not None:
            return target
    setattr(target, attr_name, adapter)
    return target


def ltm_reason1d_integration_contract() -> dict:
    return {
        "module": "ltm_depth_integration",
        "stage": "REASON-1D",
        "destructive_ltm_replacement": False,
        "default_enabled": False,
        "writes": "shadow consolidation proposals only unless explicit downstream gate permits mutation",
        "shared_physical_tensor": False,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
