from __future__ import annotations

from typing import Any, Optional

from mnemonic_cortex.reasoning_depth import WMDepthController, attach_wm_depth_controller


def install_optional_wm_depth_controller(
    target: Any,
    controller: Optional[WMDepthController] = None,
    *,
    attr_name: str = "wm_depth_controller",
) -> Any:
    """Install an optional disabled-by-default WM depth controller on a target object.

    This is intentionally non-destructive. It does not replace QDTWorkingMemory,
    does not activate depth routing by default, and does not mutate memory stores.
    """

    return attach_wm_depth_controller(target, controller=controller, attr_name=attr_name)


def wm_qd_reason1b_integration_contract() -> dict:
    return {
        "module": "wm_depth_integration",
        "stage": "REASON-1B",
        "destructive_qdt_replacement": False,
        "default_enabled": False,
        "writes": "shadow proposals only unless explicit downstream gate permits mutation",
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
