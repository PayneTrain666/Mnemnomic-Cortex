"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: validation.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Validation primitives for HGM/HPME.

Validation is deliberately fail-closed: invalid data produces explicit
errors, not silent coercion or best-effort guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .enums import ValidationSeverity


@dataclass
class ValidationMessage:
    severity: ValidationSeverity
    code: str
    message: str
    path: Optional[str] = None

    def __post_init__(self) -> None:
        self.severity = ValidationSeverity.coerce(self.severity)


@dataclass
class ValidationResult:
    """Aggregate validation result used by all foundation types."""

    ok: bool = True
    messages: List[ValidationMessage] = field(default_factory=list)

    def add(self, severity, code: str, message: str, path: Optional[str] = None) -> None:
        sev = ValidationSeverity.coerce(severity)
        self.messages.append(ValidationMessage(sev, code, message, path))
        if sev == ValidationSeverity.ERROR:
            self.ok = False

    def error(self, code: str, message: str, path: Optional[str] = None) -> None:
        self.add(ValidationSeverity.ERROR, code, message, path)

    def warning(self, code: str, message: str, path: Optional[str] = None) -> None:
        self.add(ValidationSeverity.WARNING, code, message, path)

    def info(self, code: str, message: str, path: Optional[str] = None) -> None:
        self.add(ValidationSeverity.INFO, code, message, path)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        for msg in other.messages:
            self.messages.append(msg)
        self.ok = self.ok and other.ok
        return self

    @property
    def errors(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == ValidationSeverity.WARNING]

    def raise_if_errors(self) -> None:
        if not self.ok:
            detail = "; ".join(f"{m.code}: {m.message}" for m in self.errors)
            raise ValueError(detail or "Validation failed")

    @classmethod
    def combine(cls, results: Iterable["ValidationResult"]) -> "ValidationResult":
        merged = cls()
        for result in results:
            merged.merge(result)
        return merged
