from __future__ import annotations

from typing import Any, Optional

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig


class MANNDepthIntegrationError(ValueError):
    """Raised when optional MANN depth integration cannot be installed."""


def install_optional_mann_depth_adapter(
    target: Any,
    adapter: Optional[MANNDepthAdapter] = None,
    *,
    attr_name: str = "mann_depth_adapter",
) -> Any:
    """Install an optional disabled-by-default MANN depth adapter.

    This does not replace existing MANN modules and does not activate depth
    routing unless the caller supplies an enabled adapter explicitly.
    """

    if target is None:
        raise MANNDepthIntegrationError("target cannot be None")
    if adapter is None:
        adapter = MANNDepthAdapter(MANNDepthAdapterConfig.disabled())
    if hasattr(target, attr_name):
        existing = getattr(target, attr_name)
        if existing is not None:
            return target
    setattr(target, attr_name, adapter)
    return target


def mann_reason1c_integration_contract() -> dict:
    return {
        "module": "mann_depth_integration",
        "stage": "REASON-1C",
        "destructive_mann_replacement": False,
        "default_enabled": False,
        "writes": "shadow proposals only unless explicit downstream gate permits mutation",
        "shared_physical_tensor_with_ltm": False,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
