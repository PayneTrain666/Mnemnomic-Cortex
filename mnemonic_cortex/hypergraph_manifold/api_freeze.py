"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: api freeze.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
API freeze helpers for HGM v0.1.

The freeze is a documented public-symbol snapshot. It must not remove, rename,
or mutate existing exports.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Mapping, Optional

from .hgm10_result import HGM10ReleaseOptions, APIFreezeRecord, APIFreezeSymbol, hgm10_stable_hash, trace_hgm10
from .validation import ValidationResult


def coerce_hgm10_options(options: Optional[HGM10ReleaseOptions | Mapping[str, Any]] = None) -> HGM10ReleaseOptions:
    if options is None:
        return HGM10ReleaseOptions()
    if isinstance(options, HGM10ReleaseOptions):
        return options
    return HGM10ReleaseOptions(**dict(options))


def _symbol_type(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value):
        return "function"
    if inspect.ismodule(value):
        return "module"
    return type(value).__name__


def freeze_hgm_public_api(config=None, options: Optional[HGM10ReleaseOptions | Mapping[str, Any]] = None) -> APIFreezeRecord:
    """Return a deterministic APIFreezeRecord for current HGM package exports."""

    opts = coerce_hgm10_options(options)
    validation = ValidationResult()
    traces = []
    package = importlib.import_module("mnemonic_cortex.hypergraph_manifold")
    exported_names = tuple(str(name) for name in getattr(package, "__all__", tuple()))
    if not exported_names:
        validation.error("hgm10_api.no_exports", "Package __all__ is empty or missing", "__all__")
    if len(exported_names) > opts.max_symbols:
        validation.warning("hgm10_api.symbol_bound", "API symbol list truncated by max_symbols", "symbols")
    selected = tuple(sorted(exported_names))[: opts.max_symbols]

    module_names = []
    package_path = getattr(package, "__path__", None)
    if package_path is not None:
        module_names = sorted(f"{package.__name__}.{info.name}" for info in pkgutil.iter_modules(package_path))
    if not module_names:
        validation.warning("hgm10_api.no_modules_detected", "No submodules detected for API freeze", "module_names")

    symbols = []
    for name in selected:
        value = getattr(package, name, None)
        module_name = getattr(value, "__module__", package.__name__) if value is not None else package.__name__
        if value is None:
            validation.warning("hgm10_api.export_missing_attr", f"Exported symbol {name!r} is not an attribute", f"symbols.{name}")
        if not opts.include_private_symbols and name.startswith("_"):
            validation.warning("hgm10_api.private_symbol_skipped", f"Private symbol {name!r} is not considered stable", f"symbols.{name}")
        trace = trace_hgm10("api_freeze.freeze_hgm_public_api.symbol", validation, {"symbol": name, "module": module_name})
        traces.append(trace)
        symbols.append(
            APIFreezeSymbol(
                symbol_id=f"hgm10_symbol_{hgm10_stable_hash(opts.release_version, name, module_name)}",
                symbol_name=name,
                symbol_type=_symbol_type(value),
                module_name=module_name,
                exported=True,
                stable=not name.startswith("_"),
                trace_id=trace.trace_id,
                metadata={"api_freeze_version": opts.release_version, "live_qdt_write": False},
            )
        )

    frozen = bool(validation.ok and symbols)
    trace = trace_hgm10("api_freeze.freeze_hgm_public_api", validation, {"symbol_count": len(symbols), "frozen": frozen, "secret_token": "must_redact"})
    traces.append(trace)
    return APIFreezeRecord(
        freeze_id=f"hgm10_api_freeze_{hgm10_stable_hash(opts.release_version, len(symbols), tuple(module_names))}",
        release_version=opts.release_version,
        symbols=tuple(symbols),
        module_names=tuple(module_names),
        frozen=frozen,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "api_freeze_strict": opts.api_freeze_strict,
            "symbol_count": len(symbols),
            "module_count": len(module_names),
            "live_qdt_write": False,
            "production_enabled": False,
        },
    )
