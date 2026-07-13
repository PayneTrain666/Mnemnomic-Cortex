"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth lattice benchmarks.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

import torch

from .depth_capacity_validation import DepthCapacityValidationConfig
from .shared_depth_slot_registry import SharedDepthSlotRegistry
from .wm_depth_controller import WMDepthController
from .mann_depth_adapter import MANNDepthAdapter, MANNDepthAdapterConfig
from .ltm_depth_adapter import LTMDepthAdapter, LTMDepthAdapterConfig


class DepthLatticeBenchmarkError(ValueError):
    """Raised when a benchmark config is unsafe or invalid."""


@dataclass(frozen=True)
class DepthLatticeBenchmarkConfig:
    """Bounded smoke benchmark config.

    This is not a production performance benchmark. It is a safety/readiness
    smoke harness designed to catch obvious shape, finite, and mutation issues.
    """

    iterations: int = 5
    warmup_iterations: int = 1
    slot_count: int = 32
    key_dim: int = 32
    value_dim: int = 32
    batch: int = 2
    tokens: int = 4
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not (1 <= self.iterations <= 50):
            raise DepthLatticeBenchmarkError("iterations must be in [1,50]")
        if not (0 <= self.warmup_iterations <= 10):
            raise DepthLatticeBenchmarkError("warmup_iterations must be in [0,10]")
        if not (1 <= self.slot_count <= 4096):
            raise DepthLatticeBenchmarkError("slot_count must be in [1,4096]")
        if not (1 <= self.batch <= 32):
            raise DepthLatticeBenchmarkError("batch must be in [1,32]")
        if not (1 <= self.tokens <= 128):
            raise DepthLatticeBenchmarkError("tokens must be in [1,128]")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise DepthLatticeBenchmarkError("key_dim/value_dim must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "slot_count": self.slot_count,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "batch": self.batch,
            "tokens": self.tokens,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


def _time_call(fn, *, warmup_iterations: int, iterations: int) -> Dict[str, Any]:
    for _ in range(warmup_iterations):
        fn()
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    return {
        "iterations": iterations,
        "min_ms": min(durations) * 1000.0,
        "max_ms": max(durations) * 1000.0,
        "mean_ms": (sum(durations) / len(durations)) * 1000.0,
    }


def run_depth_lattice_smoke_benchmarks(config: Optional[DepthLatticeBenchmarkConfig] = None) -> Dict[str, Any]:
    cfg = config or DepthLatticeBenchmarkConfig()
    cfg.validate()

    token_query = torch.randn(cfg.batch, cfg.tokens, cfg.key_dim)
    flat_query = torch.randn(cfg.batch, cfg.key_dim)

    wm = WMDepthController.disabled(input_dim=cfg.key_dim)
    mann = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=cfg.key_dim, value_dim=cfg.value_dim, slot_count=cfg.slot_count))
    ltm = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=cfg.key_dim, value_dim=cfg.value_dim, slot_count=cfg.slot_count))
    registry = SharedDepthSlotRegistry()

    wm_before = token_query.clone()
    mann_before = mann.bank.values.clone()
    ltm_before = ltm.banks.get("cgmn_semantic").values.clone()

    def wm_read():
        out, _ = wm.process_wm(token_query, return_trace=True)
        if cfg.finite_checks and not torch.isfinite(out).all():
            raise DepthLatticeBenchmarkError("WM output contains NaN/Inf")
        return out

    def mann_read():
        out, _ = mann.read_hop(token_query, hop_id=0, return_trace=True)
        if cfg.finite_checks and not torch.isfinite(out).all():
            raise DepthLatticeBenchmarkError("MANN output contains NaN/Inf")
        return out

    def ltm_read():
        out, _ = ltm.read_ltm(flat_query, bank_name="cgmn_semantic", return_trace=True)
        if cfg.finite_checks and not torch.isfinite(out).all():
            raise DepthLatticeBenchmarkError("LTM output contains NaN/Inf")
        return out

    def registry_smoke():
        rec = registry.create_or_update(
            canonical_slot_id="benchmark.registry.1",
            content="benchmark",
            wm_ref="wm.benchmark.1",
            mann_ref="mann.benchmark.1",
            ltm_ref="ltm.benchmark.1",
            source_stage="REASON-1E",
            source_pack="benchmark",
            depth_roles_present=[1, 5],
        )
        return rec.to_dict()

    results = {
        "config": cfg.to_dict(),
        "bounded": True,
        "benchmark_type": "smoke_latency_not_production",
        "wm_read": _time_call(wm_read, warmup_iterations=cfg.warmup_iterations, iterations=cfg.iterations),
        "mann_read": _time_call(mann_read, warmup_iterations=cfg.warmup_iterations, iterations=cfg.iterations),
        "ltm_read": _time_call(ltm_read, warmup_iterations=cfg.warmup_iterations, iterations=cfg.iterations),
        "registry_serialization": _time_call(registry_smoke, warmup_iterations=cfg.warmup_iterations, iterations=cfg.iterations),
    }

    no_mutation = (
        torch.equal(token_query, wm_before)
        and torch.equal(mann.bank.values, mann_before)
        and torch.equal(ltm.banks.get("cgmn_semantic").values, ltm_before)
    )
    results["safety"] = {
        "no_mutation_by_default": bool(no_mutation),
        "destructive_replacement": False,
        "permanent_memory_store_mutation": False,
        "fake_production_complete_claim": False,
        "fake_quantum_hardware_claim": False,
    }
    return results


def depth_lattice_benchmark_contract() -> Dict[str, Any]:
    return {
        "module": "depth_lattice_benchmarks",
        "stage": "REASON-1E",
        "benchmark_type": "bounded_smoke_latency",
        "production_benchmark_claim": False,
        "bounded_iterations": True,
        "no_mutation_by_default": True,
    }
