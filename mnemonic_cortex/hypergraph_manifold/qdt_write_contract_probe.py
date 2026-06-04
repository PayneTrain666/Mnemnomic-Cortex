"""Signature-level QDT/WM contract probes for HGM write-prep.

The probe inspects class/function signatures only.  It does not instantiate
commit gates, create proposals, stage writes, or mutate QDT/WM state.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Mapping, Optional, Tuple

from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    QDTContractSymbolProbe,
    QDTWriteContractProbeResult,
    trace_write_prep,
    write_prep_stable_hash,
)
from .validation import ValidationResult


def _coerce_options(options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]]) -> HGMQDTWritePrepOptions:
    if options is None:
        return HGMQDTWritePrepOptions()
    if isinstance(options, HGMQDTWritePrepOptions):
        return options
    return HGMQDTWritePrepOptions(**dict(options))


def _probe_symbol(module_path: str, symbol_name: str, required_fields: Tuple[str, ...]) -> tuple[QDTContractSymbolProbe, ValidationResult, Any]:
    validation = ValidationResult()
    present = False
    compatible = False
    signature = ""
    missing: Tuple[str, ...] = required_fields
    reason = "symbol not available"
    try:
        module = importlib.import_module(module_path)
        symbol = getattr(module, symbol_name)
        present = True
        if inspect.isclass(symbol):
            params = set(getattr(symbol, "__dataclass_fields__", {}).keys())
            create = getattr(symbol, "create", None)
            if callable(create):
                signature = str(inspect.signature(create))
                params |= set(inspect.signature(create).parameters.keys())
            else:
                signature = str(inspect.signature(symbol)) if callable(symbol) else "class"
            missing = tuple(f for f in required_fields if f not in params)
        elif callable(symbol):
            sig = inspect.signature(symbol)
            signature = str(sig)
            params = set(sig.parameters.keys())
            missing = tuple(f for f in required_fields if f not in params)
        else:
            signature = type(symbol).__name__
            missing = tuple()
        compatible = present and not missing
        reason = "compatible" if compatible else f"missing required fields: {', '.join(missing)}"
        if not compatible:
            validation.warning("qdt_probe.incomplete_symbol", reason, f"{module_path}.{symbol_name}")
    except Exception as exc:  # pragma: no cover - exact import failures are environment dependent
        validation.warning("qdt_probe.import_failed", f"{module_path}.{symbol_name}: {exc}", f"{module_path}.{symbol_name}")
        reason = str(exc)
    trace = trace_write_prep("qdt_write_contract_probe.symbol", validation, {"symbol": symbol_name, "module": module_path, "present": present, "compatible": compatible})
    probe = QDTContractSymbolProbe(
        symbol_name=symbol_name,
        module_path=module_path,
        present=present,
        signature=signature,
        required_fields=required_fields,
        missing_fields=missing,
        compatible=compatible,
        reason=reason,
        trace_id=trace.trace_id,
        metadata={"read_only_probe": True},
    )
    return probe, validation, trace


def probe_qdt_wm_write_contracts(config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> QDTWriteContractProbeResult:
    opts = _coerce_options(options)
    validation = ValidationResult()
    probes = []
    traces = []
    specs = [
        ("mnemonic_cortex.working_memory.wm_system_commit_gate", "SystemWriteProposal", ("content", "local_slot_id", "geometry_map", "depth_index", "write_permission", "confidence")),
        ("mnemonic_cortex.working_memory.wm_system_commit_gate", "SystemCommitGate", ("dim", "shared_slot_store", "qh_storage")),
        ("mnemonic_cortex.working_memory.wm_shared_slot_registry", "canonical_slot_id", ("namespace", "local_slot_id")),
        ("mnemonic_cortex.working_memory.wm_quantum_holographic_storage", "build_qh_code_schema", ("depth_index", "bank_name", "geometry_name", "triplet_index", "memory_type")),
        ("mnemonic_cortex.working_memory.wm_quantum_holographic_storage", "QHStorageRecord", ("record_id", "canonical_slot_id", "code_schema", "vector_fingerprint", "vector_norm")),
    ][: opts.max_probe_symbols]
    for module_path, symbol_name, required in specs:
        probe, result, trace = _probe_symbol(module_path, symbol_name, tuple(required))
        probes.append(probe)
        validation.merge(result)
        traces.append(trace)
    compatible = bool(probes and all(p.compatible for p in probes))
    if not compatible:
        validation.warning("qdt_probe.not_write_ready", "one or more QDT/WM contract probes are unavailable or incomplete", "probes")
    final_trace = trace_write_prep("qdt_write_contract_probe.probe_qdt_wm_write_contracts", validation, {"compatible": compatible, "probe_count": len(probes)})
    traces.append(final_trace)
    return QDTWriteContractProbeResult(
        probes=tuple(probes),
        compatible=compatible,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"read_only": True, "probe_id": f"qdt_probe_{write_prep_stable_hash(len(probes), compatible)}"},
    )
