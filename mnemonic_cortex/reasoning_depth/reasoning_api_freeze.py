from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid

from .reasoning_controller_api import ReasoningControllerAPIConfig, reasoning_controller_api_contract
from .reasoning_release_candidate import _safe_jsonable


class ReasoningAPIFreezeError(ValueError):
    """Raised when API freeze metadata is unsafe."""


@dataclass(frozen=True)
class ReasoningAPIFreezeConfig:
    """API-freeze metadata config."""

    enabled: bool = False
    freeze_name: str = "reasoning-controller-api-rc"
    require_public_contract: bool = True
    require_config_serialization: bool = True
    require_result_serialization: bool = True
    require_no_write_permission: bool = True
    allow_api_breaking_changes: bool = False
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.freeze_name:
            raise ReasoningAPIFreezeError("freeze_name is required")
        if self.allow_api_breaking_changes:
            raise ReasoningAPIFreezeError("API-breaking changes are forbidden in freeze stage")
        if not self.no_mutation_by_default:
            raise ReasoningAPIFreezeError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "ReasoningAPIFreezeConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningAPIFreezeConfig":
        return cls(enabled=True)


@dataclass
class ReasoningAPIFreezeReport:
    """JSON-safe public API freeze report."""

    enabled: bool
    freeze_name: str
    frozen_symbols: List[str] = field(default_factory=list)
    config_fields: List[str] = field(default_factory=list)
    contract_hash: str = ""
    compatible: bool = False
    blocking_items: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"reasoning_api_freeze_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "freeze_name": self.freeze_name,
            "frozen_symbols": list(self.frozen_symbols),
            "config_fields": list(self.config_fields),
            "contract_hash": self.contract_hash,
            "compatible": bool(self.compatible),
            "blocking_items": list(self.blocking_items),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningAPIFreeze:
    """Builds API freeze metadata without altering runtime behavior."""

    FROZEN_SYMBOLS = [
        "ReasoningControllerAPI",
        "ReasoningControllerAPIConfig",
        "ReasoningControllerAPIResult",
        "ReasoningControllerAPIError",
        "reasoning_controller_api_contract",
    ]

    def __init__(self, config: Optional[ReasoningAPIFreezeConfig] = None):
        self.config = config or ReasoningAPIFreezeConfig.disabled()
        self.config.validate()

    def freeze(self, *, lineage: Optional[Dict[str, Any]] = None) -> ReasoningAPIFreezeReport:
        if not self.config.enabled:
            return ReasoningAPIFreezeReport(
                enabled=False,
                freeze_name=self.config.freeze_name,
                blocking_items=["api_freeze_disabled"],
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        contract = reasoning_controller_api_contract()
        fields = list(ReasoningControllerAPIConfig.__dataclass_fields__.keys())
        payload_for_hash = {"symbols": self.FROZEN_SYMBOLS, "fields": fields, "contract": contract}
        contract_hash = hashlib.sha256(json.dumps(_safe_jsonable(payload_for_hash), sort_keys=True).encode("utf-8")).hexdigest()

        blocking: List[str] = []
        if self.config.require_public_contract and not contract.get("public_api_wrapper"):
            blocking.append("missing_public_api_contract")
        if self.config.require_config_serialization and not contract.get("config_serialization"):
            blocking.append("missing_config_serialization")
        if self.config.require_result_serialization and not contract.get("json_safe_result"):
            blocking.append("missing_result_serialization")
        if self.config.require_no_write_permission and contract.get("write_permission_public_api") is not False:
            blocking.append("write_permission_exposed")

        report = ReasoningAPIFreezeReport(
            enabled=True,
            freeze_name=self.config.freeze_name,
            frozen_symbols=list(self.FROZEN_SYMBOLS),
            config_fields=fields,
            contract_hash=contract_hash,
            compatible=not blocking,
            blocking_items=blocking,
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "metadata_only": True,
            "api_breaking_changes_allowed": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
        }


def reasoning_api_freeze_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_api_freeze",
        "stage": "REASON-3D",
        "default_enabled": False,
        "api_breaking_changes_allowed": False,
        "metadata_only": True,
        "json_safe_report": True,
    }
