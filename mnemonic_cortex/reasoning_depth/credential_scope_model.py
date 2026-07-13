"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: credential scope model.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class CredentialScopeError(ValueError):
    """Raised when credential scope metadata is unsafe."""


class CredentialScopeKind(str, Enum):
    NONE = "none"
    TEST_INJECTED = "test_injected"
    FUTURE_SECRET_MANAGER = "future_secret_manager"


@dataclass(frozen=True)
class CredentialScopeConfig:
    enabled: bool = False
    scope_kind: CredentialScopeKind = CredentialScopeKind.NONE
    allow_real_secret_loading: bool = False
    require_least_privilege: bool = True
    require_redaction: bool = True
    require_rotation_plan: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.allow_real_secret_loading:
            raise CredentialScopeError("real secret loading is not allowed in REAL-BACKEND-IMPLEMENTATION-A")
        if not self.no_mutation_by_default:
            raise CredentialScopeError("no_mutation_by_default must remain true")
        if not self.require_least_privilege:
            raise CredentialScopeError("least privilege is required")

    @classmethod
    def disabled(cls) -> "CredentialScopeConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_test_injected(cls) -> "CredentialScopeConfig":
        return cls(enabled=True, scope_kind=CredentialScopeKind.TEST_INJECTED)


@dataclass
class CredentialScopeReport:
    enabled: bool
    scope_kind: CredentialScopeKind
    allowed_scopes: List[str]
    blocked_scopes: List[str]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"credential_scope_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "scope_kind": self.scope_kind.value,
            "allowed_scopes": list(self.allowed_scopes),
            "blocked_scopes": list(self.blocked_scopes),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "real_secret_loaded": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class CredentialScopeModel:
    def __init__(self, config: Optional[CredentialScopeConfig] = None):
        self.config = config or CredentialScopeConfig.disabled()
        self.config.validate()

    def build(self) -> CredentialScopeReport:
        if not self.config.enabled:
            return CredentialScopeReport(
                enabled=False,
                scope_kind=self.config.scope_kind,
                allowed_scopes=[],
                blocked_scopes=self._blocked_scopes() + ["credential_scope_disabled"],
                safety_flags=self._safety_flags(),
            )

        return CredentialScopeReport(
            enabled=True,
            scope_kind=self.config.scope_kind,
            allowed_scopes=[
                "dry_run_metadata_validation",
                "test_injected_fake_credentials_only",
                "no_secret_trace_logging",
            ],
            blocked_scopes=self._blocked_scopes(),
            safety_flags=self._safety_flags(),
        )

    @staticmethod
    def _blocked_scopes() -> List[str]:
        return [
            "real_secret_loading",
            "production_secret_manager_access",
            "write_credentials",
            "schema_migration_credentials",
            "admin_credentials",
            "credential_logging",
        ]

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "real_secret_loaded": False,
            "credential_logging_allowed": False,
            "least_privilege_required": True,
            "redaction_required": True,
            "rotation_plan_required_before_real_use": True,
        }


def credential_scope_model_contract() -> Dict[str, Any]:
    return {
        "module": "credential_scope_model",
        "stage": "REAL-BACKEND-IMPLEMENTATION-A",
        "real_secret_loaded": False,
        "test_injected_only": True,
        "least_privilege_required": True,
        "json_safe_report": True,
    }
