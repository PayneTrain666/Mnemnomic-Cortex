"""Count parameter space for copy-task model configurations."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from benchmark.models import CortexSeqModel
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
    }
    out = {}
    for k, v in parts.items():
        if v is None:
            continue
        if hasattr(v, "parameters"):
            out[k] = count_params(v)
    return out


def build_model(d_model=160, task_decoder=True, full_stack=False):
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
