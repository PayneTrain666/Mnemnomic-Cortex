"""
Plain-language summary
----------------------
What this file is for: Dry-run or sandbox helper (dry_run_backend_interface).
How it fits in the system: Lets engineers rehearse a path safely without committing live side effects.
Status: LOW-USE / SAFETY SCAFFOLD
Important notes for non-coders: Not the everyday training path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import json

from .backend_interface_protocol import (
    BackendDryRunResult,
    BackendInterfaceConfig,
    BackendInterfaceError,
    BackendInterfaceKind,
    BackendPayloadEnvelope,
)


@dataclass
class DryRunBackendInterface:
    """In-memory dry-run backend interface.

    It records idempotency keys only in process memory for duplicate detection.
    It does not open files, connect to databases, load credentials, execute
    schema migrations, or mutate a permanent memory store.
    """

    config: BackendInterfaceConfig = field(default_factory=BackendInterfaceConfig.disabled)
    seen_idempotency_keys: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.config.validate()

    def dry_run_write(self, envelope: BackendPayloadEnvelope) -> BackendDryRunResult:
        if not self.config.enabled:
            return BackendDryRunResult(
                accepted=False,
                dry_run=True,
                backend_kind=self.config.backend_kind,
                idempotency_key=envelope.idempotency_key,
                would_write_payload={},
                blocked_actions=["backend_interface_disabled", *self._blocked_actions()],
            )

        envelope_payload = envelope.to_dict()
        if self.config.require_json_safe_payload:
            json.dumps(envelope_payload, sort_keys=True)
        if self.config.require_redaction_status and envelope.redaction_status not in {"not_required", "redacted", "approved_safe"}:
            raise BackendInterfaceError("redaction_status must be not_required, redacted, or approved_safe")
        if envelope.idempotency_key in self.seen_idempotency_keys:
            return BackendDryRunResult(
                accepted=False,
                dry_run=True,
                backend_kind=self.config.backend_kind,
                idempotency_key=envelope.idempotency_key,
                would_write_payload=envelope_payload,
                blocked_actions=["duplicate_idempotency_key", *self._blocked_actions()],
            )

        self.seen_idempotency_keys.add(envelope.idempotency_key)
        return BackendDryRunResult(
            accepted=True,
            dry_run=True,
            backend_kind=self.config.backend_kind,
            idempotency_key=envelope.idempotency_key,
            would_write_payload=envelope_payload,
            blocked_actions=self._blocked_actions(),
        )

    def dependency_check(self) -> Dict[str, Any]:
        payload = {
            "backend_kind": self.config.backend_kind.value,
            "enabled": bool(self.config.enabled),
            "dry_run_only": True,
            "requires_external_connection": False,
            "credentials_loaded": False,
            "schema_migration_executed": False,
            "real_store_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "real_store_write",
            "credential_loading",
            "schema_migration_execution",
            "backend_activation",
            "permanent_memory_store_mutation",
        ]


def create_default_dry_run_backend(kind: BackendInterfaceKind = BackendInterfaceKind.JSONL_SHADOW) -> DryRunBackendInterface:
    return DryRunBackendInterface(config=BackendInterfaceConfig(enabled=True, backend_kind=kind))


def dry_run_backend_interface_contract() -> Dict[str, Any]:
    return {
        "module": "dry_run_backend_interface",
        "stage": "REAL-BACKEND-IMPLEMENTATION-A",
        "dry_run_only": True,
        "real_store_write_performed": False,
        "credentials_loaded": False,
        "schema_migration_executed": False,
        "idempotency_guard": True,
    }
