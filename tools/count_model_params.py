"""
Plain-language summary
----------------------
What this file is for: Prints trainable parameter counts for common copy-task configs.
How it fits in the system: Quick capacity check before training.
Status: WORKING
Important notes for non-coders: Useful when deciding if you have VRAM headroom.

Technical notes (original):
Count parameter space for copy-task model configurations.
"""
import os
import sys
from typing import Any, Dict, Iterable, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from benchmark.tasks import VOCAB_SIZE
from data.copy_task_dataloader import CopyTaskDataConfig, CopyTaskMasteryModel


def count_params(module, trainable_only=False):
    if module is None:
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if trainable_only:
        return trainable, trainable
    return total, trainable


def _tensor_bytes(values: Iterable[Any]) -> int:
    """Count each live tensor once, including nested optimizer state."""
    seen = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if isinstance(value, torch.Tensor):
            ident = id(value)
            if ident not in seen:
                seen.add(ident)
                total += int(value.numel()) * int(value.element_size())
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    for item in values:
        visit(item)
    return total


def literal_training_bytes(
    module,
    *,
    optimizer=None,
    cps_store=None,
    cuda_device: Optional[torch.device] = None,
) -> Dict[str, Optional[int]]:
    """Return literal, non-overlapping runtime memory categories.

    CPS rollback storage is deliberately not inferred from private store fields.
    It is reported only when the public capacity report exposes that byte count.
    """
    parameters = list(module.parameters()) if module is not None else []
    capacity = (
        cps_store.capacity_report()
        if cps_store is not None and hasattr(cps_store, "capacity_report")
        else {}
    )
    rollback_bytes = capacity.get("rollback_bytes")
    cuda_peak = None
    if cuda_device is not None and torch.device(cuda_device).type == "cuda":
        cuda_peak = int(torch.cuda.max_memory_allocated(torch.device(cuda_device)))
    return {
        "parameter_bytes": _tensor_bytes(parameters),
        "gradient_bytes": _tensor_bytes(
            parameter.grad for parameter in parameters if parameter.grad is not None
        ),
        "optimizer_bytes": _tensor_bytes(
            optimizer.state.values() if optimizer is not None else ()
        ),
        "cps_rollback_bytes": (
            int(rollback_bytes) if rollback_bytes is not None else None
        ),
        "cuda_peak_allocated_bytes": cuda_peak,
    }


def print_literal_training_bytes(
    label: str,
    module,
    *,
    optimizer=None,
    cps_store=None,
    cuda_device: Optional[torch.device] = None,
) -> Dict[str, Optional[int]]:
    report = literal_training_bytes(
        module,
        optimizer=optimizer,
        cps_store=cps_store,
        cuda_device=cuda_device,
    )
    rendered = " ".join(
        f"{key}={'unavailable' if value is None else value}"
        for key, value in report.items()
    )
    print(f"[capacity] {label} {rendered}", flush=True)
    return report


def fmt(n):
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.3f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def breakdown_cortex(model):
    c = model.cortex
    parts = {
        "working_memory": getattr(c, "working_memory", None),
        "long_term_memory": getattr(c, "long_term_memory", None),
        "topology": getattr(c, "topology", None),
        "consolidated_lexicon": getattr(c, "consolidated_lexicon", None),
        "advanced_broker": getattr(c, "advanced_broker", None),
        "shared_memory_subsystem": getattr(c, "shared_memory_subsystem", None),
        "hg_episodic_ltm": getattr(c, "hg_episodic_ltm", None),
        "reasoning_controller_api": getattr(c, "reasoning_controller_api", None),
        "trainable_parameter_cps": getattr(c, "trainable_parameter_cps", None),
    }
    out = {}
    for k, v in parts.items():
        if v is None:
            continue
        if hasattr(v, "parameters"):
            out[k] = count_params(v)
    return out


def build_model(d_model=160, task_decoder=True, full_stack=False):
    # Keep accounting helpers importable without constructing the benchmark model.
    from benchmark.models import CortexSeqModel

    m = CortexSeqModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        ltm_curved_hidden_dim=0,
        ltm_curved_hidden_mult=1.5,
        cms_enabled=True,
        cms_senses=3,
        recall_loss_weight=0.12,
        cms_aux_weight=0.015,
        task_decoder_enabled=task_decoder,
        task_decoder_layers=4,
        task_decoder_heads=8,
        task_decoder_use_sinusoidal=True,
    )
    if full_stack:
        m.cortex.enable_cps_cms_full_stack(
            vocab_size=VOCAB_SIZE,
            cms_senses=3,
            enable_broker=True,
            enable_advanced=True,
            enable_reasoning_bridge=True,
            enable_qdt_wm_bridge=True,
        )
        if hasattr(m.cortex, "enable_shared_memory_subsystem"):
            m.cortex.enable_shared_memory_subsystem()
    return m


def report(label, model, mastery=None):
    print(f"\n=== {label} ===")
    segs = [
        ("embedding", model.embedding),
        ("cortex (total)", model.cortex),
        ("output_proj", model.proj),
    ]
    if getattr(model, "task_decoder_enabled", False):
        for name in [
            "pre_fusion_stack_attn",
            "memory_compare_attn",
            "memory_compare_encoder",
            "memory_compare_norm",
            "system_specialization_head",
            "mann_proxy",
            "final_system_gate",
            "task_decoder",
            "task_decoder_norm",
        ]:
            if hasattr(model, name):
                segs.append((f"task_decoder/{name}", getattr(model, name)))

    for name, mod in segs:
        t, tr = count_params(mod)
        print(f"  {name:30s} {fmt(t):>10s}  trainable {fmt(tr)}")

    bd = breakdown_cortex(model)
    if bd:
        print("  --- cortex submodules ---")
        for k, (t, tr) in sorted(bd.items(), key=lambda x: -x[1][0]):
            print(f"    {k:28s} {fmt(t):>10s}  trainable {fmt(tr)}")

    total, trainable = count_params(model)
    trainable_cps = getattr(model, "trainable_parameter_cps", None)
    if trainable_cps is None:
        trainable_cps = getattr(model.cortex, "trainable_parameter_cps", None)
    if trainable_cps is not None and hasattr(trainable_cps, "capacity_report"):
        cap = trainable_cps.capacity_report()
        print("  --- trainable CPS literal capacity ---")
        for key in (
            "original_eligible_scalars",
            "literal_trainable_scalars",
            "unique_scalars_after_sharing",
            "compressed_scalars",
            "real_compression_ratio",
            "literal_bytes",
            "estimated_adam_training_bytes",
        ):
            if key in cap:
                print(f"    {key:34s} {cap[key]}")
    if mastery is not None:
        mt, mtr = count_params(mastery)
        print(f"  {'copy_mastery_model':30s} {fmt(mt):>10s}  trainable {fmt(mtr)}")
        total += mt
        trainable += mtr

    print(f"  {'TOTAL':30s} {fmt(total):>10s}  trainable {fmt(trainable)}")
    return total


def main():
    configs = [
        ("baseline d_model=160", 160, False, False),
        ("+ task_decoder", 160, True, False),
        ("+ full fusion stack", 160, False, True),
        ("full stack + task_decoder (GPU plan)", 160, True, True),
        ("smoke test d_model=96 full", 96, True, True),
    ]
    for label, d, td, fs in configs:
        report(label, build_model(d, td, fs))

    cfg = CopyTaskDataConfig(
        n_samples=64,
        max_len=16,
        encoder_d_model=160,
        encoder_heads=4,
        decoder_heads=4,
        encoder_layers=2,
        decoder_layers=2,
    )
    mastery = CopyTaskMasteryModel(cfg, vocab_size=VOCAB_SIZE)
    mt, _ = count_params(mastery)
    print(f"\n=== copy mastery alone (d_model=160, 2+2 layers) ===")
    print(f"  TOTAL {fmt(mt)}")
    report("full training stack (+ mastery)", build_model(160, True, True), mastery)


if __name__ == "__main__":
    main()
