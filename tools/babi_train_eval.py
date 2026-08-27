"""
Plain-language summary
----------------------
What this file is for: Trains or evaluates on facebook/babi_qa style question answering.
How it fits in the system: Language reasoning benchmark path for CortexSeqModel.
Status: WORKING (needs dataset)
Important notes for non-coders: Depends on external bAbI data availability.
"""

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import sys

# Ensure project root is importable when executed as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_DIR = os.path.abspath(os.path.dirname(__file__))
if TOOL_DIR in sys.path:
    sys.path.remove(TOOL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark.models import CortexSeqModel
from mnemonic_cortex.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_warmup_cosine_scheduler,
    load_optimizer_state_dict_checked,
    optimizer_parameter_layout,
)
from mnemonic_cortex.trainable_parameter_cps import TrainableParameterCPSConfig
from mnemonic_cortex.parameter_audit import ParameterAuditLogger
from tools.count_model_params import print_literal_training_bytes


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Example:
    tokens: List[str]
    answer: str


def build_examples(split_rows) -> List[Example]:
    ex = []
    for row in split_rows:
        story = row["story"]
        texts = story["text"]
        types = story["type"]
        answers = story["answer"]
        for i, t in enumerate(types):
            if int(t) == 1:
                context = [texts[j] for j in range(i) if int(types[j]) == 0]
                q = texts[i]
                a = answers[i].strip()
                if not a:
                    continue
                # marker structure improves position consistency.
                flat = ("<ctx> " + " ".join(context) + " <q> " + q).lower()
                toks = flat.replace("?", " ?").replace(".", " .").split()
                ex.append(Example(tokens=toks, answer=a.lower()))
    return ex


def build_flat_examples(split_rows) -> List[Example]:
    """Convert the maintained script-free bAbI mirror into local examples."""
    examples = []
    for row in split_rows:
        answer = str(row["answer"]).strip().lower()
        if not answer:
            continue
        flat = (
            "<ctx> "
            + str(row["passage"]).replace("\n", " ")
            + " <q> "
            + str(row["question"])
        ).lower()
        tokens = flat.replace("?", " ?").replace(".", " .").split()
        examples.append(Example(tokens=tokens, answer=answer))
    return examples


def _task_number(config: str) -> int:
    match = re.search(r"qa(\d+)", str(config).lower())
    if match is None:
        raise ValueError(f"cannot determine bAbI task number from config={config!r}")
    task = int(match.group(1))
    if not 1 <= task <= 20:
        raise ValueError("bAbI task number must be in [1, 20]")
    return task


def load_babi_examples(load_dataset, *, source: str, config: str):
    """Load a working bAbI source and return train/validation/test examples."""
    if source == "facebook":
        dataset = load_dataset(
            "facebook/babi_qa",
            config,
            trust_remote_code=True,
        )
        return build_examples(dataset["train"]), None, build_examples(dataset["test"])
    task = _task_number(config)
    dataset = load_dataset("Muennighoff/babi")

    def task_rows(split_name: str):
        split = dataset[split_name]
        return split.filter(lambda row: int(row["task"]) == task)

    return (
        build_flat_examples(task_rows("train")),
        build_flat_examples(task_rows("validation")),
        build_flat_examples(task_rows("test")),
    )


def build_vocab(train_ex: List[Example], test_ex: List[Example]):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in train_ex + test_ex:
        for t in ex.tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
        if ex.answer not in vocab:
            vocab[ex.answer] = len(vocab)
    return vocab


class BabiQADataset(Dataset):
    def __init__(self, examples: List[Example], stoi: dict, max_len: int):
        self.examples = examples
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in ex.tokens][: self.max_len]
        ans = self.stoi.get(ex.answer, self.stoi["<unk>"])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ans, dtype=torch.long)


def collate_pad(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    src = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        src[i, : x.size(0)] = x
    y = torch.stack(ys)
    return src, y


def _answer_logits(logits: torch.Tensor, src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Select each story's final real token rather than a padded position."""
    lengths = src.ne(int(pad_id)).sum(dim=1).clamp_min(1)
    row = torch.arange(src.size(0), device=src.device)
    return logits[row, lengths - 1]


def _resolve_amp_dtype(
    device: torch.device, enabled: bool, requested: str
) -> Tuple[bool, torch.dtype, str]:
    if not bool(enabled) or device.type != "cuda":
        return False, torch.float16, "disabled"
    choice = str(requested).strip().lower()
    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if choice == "bf16" and not bf16_supported:
        return True, torch.float16, "fp16_fallback"
    if choice == "bf16" or (choice == "auto" and bf16_supported):
        return True, torch.bfloat16, "bf16"
    return True, torch.float16, "fp16"


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=bool(enabled and device.type == "cuda"),
    )


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _configure_sdpa_backends(device: torch.device) -> Dict[str, Any]:
    """Prefer fused SDPA when available; leave standard MHA behavior intact."""
    report: Dict[str, Any] = {
        "requested": True,
        "device": str(device),
        "flash": None,
        "mem_efficient": None,
        "math": None,
        "status": "unavailable",
    }
    if device.type != "cuda" or not hasattr(torch.backends, "cuda"):
        report["status"] = "skipped_non_cuda"
        return report
    cuda_backends = torch.backends.cuda
    try:
        if hasattr(cuda_backends, "enable_flash_sdp"):
            cuda_backends.enable_flash_sdp(True)
            report["flash"] = True
        if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
            cuda_backends.enable_mem_efficient_sdp(True)
            report["mem_efficient"] = True
        if hasattr(cuda_backends, "enable_math_sdp"):
            cuda_backends.enable_math_sdp(True)
            report["math"] = True
        report["status"] = "configured"
    except Exception as exc:  # pragma: no cover - backend availability varies
        report["status"] = f"error:{type(exc).__name__}"
    return report


def _clear_complex_grads(params) -> None:
    """GradScaler cannot safely unscale complex parameter gradients."""
    for parameter in params:
        if parameter.grad is not None and parameter.grad.is_complex():
            parameter.grad = None


def evaluate(
    model,
    loader,
    device,
    *,
    pad_id: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    with torch.no_grad():
        for src, y in loader:
            src = src.to(device)
            y = y.to(device)
            with _autocast(torch.device(device), use_amp, amp_dtype):
                logits = model(src)  # [B,T,V]
                last_logits = _answer_logits(logits, src, pad_id)
                loss = F.cross_entropy(last_logits, y)
            total_loss += loss.item() * src.size(0)
            pred = last_logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            n += src.size(0)
    return total_loss / max(1, n), correct / max(1, n)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def _checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_acc: float,
    vocab: Dict[str, int],
    args,
) -> Dict[str, Any]:
    manifest = (
        model.trainable_parameter_cps_manifest()
        if hasattr(model, "trainable_parameter_cps_manifest")
        else None
    )
    return {
        "format_version": 2,
        "task": "babi_qa",
        "dataset_source": str(args.dataset_source),
        "babi_config": str(args.config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_acc": float(best_val_acc),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_parameter_layout": optimizer_parameter_layout(optimizer),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "trainable_parameter_cps_manifest": manifest,
        "vocab": dict(vocab),
        "args": vars(args),
        "saved_at": float(time.time()),
    }


def _load_model_state(model, state: Dict[str, torch.Tensor], strict: bool) -> None:
    if strict:
        model.load_state_dict(state, strict=True)
        return
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)


def _print_metric_legend():
    print("[babi] metric guide:", flush=True)
    print("  train logs:", flush=True)
    print("    loss      = current batch total loss (task + enabled aux); lower is better", flush=True)
    print("    mean      = running average train loss in current epoch; lower is better", flush=True)
    print("    acc       = running answer accuracy in current epoch; higher is better", flush=True)
    print("    recall    = auxiliary recall objective from cortex memory retrieval; lower is better", flush=True)
    print("    cms       = consolidated memory / CPS auxiliary regularization term; small positive is expected", flush=True)
    print("    pre_grad  = global grad norm right after backward, before normalization/clipping", flush=True)
    print("    grad      = grad norm after normalization + safety clipping (the norm used for optimizer step)", flush=True)
    print("    lr        = current learning rate after warmup/cosine schedule", flush=True)
    print("    topo_fit  = topology manager fitness EMA (derived from loss); higher generally means healthier dynamics", flush=True)
    print("    gstep     = global optimizer update step count", flush=True)
    print("    router    = grouped advanced-router snapshot (domain probs, reg, entropy, feature means)", flush=True)
    print("    ltm_router= HG/CGMN/Curved fusion weights from memory-level router", flush=True)
    print("  epoch logs:", flush=True)
    print("    train_loss= full-epoch average training loss; lower is better", flush=True)
    print("    val_loss  = loss on held-out training examples; lower is better", flush=True)
    print("    val_acc   = held-out answer accuracy used for checkpoint selection", flush=True)
    print("    final_test= official bAbI test score, evaluated once after selection", flush=True)
    print("    router_*  = epoch-level router snapshot means/counters (only when advanced router is enabled)", flush=True)
    print("    ltm_router_* = epoch-level memory-fusion router means per bank", flush=True)


def _format_router_snapshot(metrics: dict) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    dom_probs = []
    for k, v in metrics.items():
        if k.startswith("router_prob_"):
            dom = k.replace("router_prob_", "")
            dom_probs.append((dom, float(v)))
    dom_probs.sort(key=lambda x: x[1], reverse=True)
    dom_probs = dom_probs[:3]
    if not dom_probs:
        return ""
    reg = float(metrics.get("router_reg", 0.0))
    ent = float(metrics.get("router_entropy", 0.0))
    sph = float(metrics.get("router_sparsity_mass", 0.0))
    feat_triplet = (
        float(metrics.get("router_feat_hg_entropy", 0.0)),
        float(metrics.get("router_feat_cgmn_entropy", 0.0)),
        float(metrics.get("router_feat_curved_entropy", 0.0)),
    )
    probs_txt = ", ".join([f"{d}:{p:.3f}" for d, p in dom_probs])
    return (
        f"probs[{probs_txt}] reg={reg:.4f} ent={ent:.4f} "
        f"mass@k={sph:.4f} mem_ent[hg/cg/cur]={feat_triplet[0]:.3f}/{feat_triplet[1]:.3f}/{feat_triplet[2]:.3f}"
    )


def _format_ltm_router_snapshot(metrics: dict) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    hg = metrics.get("ltm_router_hg", None)
    cg = metrics.get("ltm_router_cgmn", None)
    cv = metrics.get("ltm_router_curved", None)
    if hg is None or cg is None or cv is None:
        return ""
    return (
        f"weights[hg:{float(hg):.3f}, cgmn:{float(cg):.3f}, curved:{float(cv):.3f}] "
        f"argmax={max([('hg', float(hg)), ('cgmn', float(cg)), ('curved', float(cv))], key=lambda x: x[1])[0]}"
    )


def _safe_mean_scalar(v, default: float) -> float:
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return float(default)
        return float(v.detach().float().mean().item())
    try:
        return float(v)
    except Exception:
        return float(default)


def _build_topology_telemetry(cortex) -> dict:
    """
    Build per-bank telemetry dict expected by step_topology_schedulers().
    """
    out = {"WM": {}, "HG": {}, "CGMN": {}}
    if cortex is None:
        return out

    def _extract(mem_obj):
        feats = getattr(mem_obj, "last_router_features", None)
        if not isinstance(feats, dict):
            return {}
        return {
            "entropy": _safe_mean_scalar(feats.get("entropy", 0.5), 0.5),
            "dist_mean": _safe_mean_scalar(feats.get("dist_mean", 1.0), 1.0),
        }

    wm = getattr(cortex, "working_memory", None)
    ltm = getattr(cortex, "long_term_memory", None)
    hg = getattr(ltm, "hg", None) if ltm is not None else None
    cg = getattr(ltm, "cgmn", None) if ltm is not None else None

    out["WM"] = _extract(wm)
    out["HG"] = _extract(hg)
    out["CGMN"] = _extract(cg)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train/evaluate CortexSeqModel on Facebook bAbI QA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="en-qa1", help="bAbI config, e.g. en-qa1 or en-10k-qa1")
    p.add_argument(
        "--dataset_source",
        choices=["mirror", "facebook"],
        default="mirror",
        help="The maintained mirror avoids the original Facebook archive's dead URL.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--d_model", type=int, default=160)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--working_memory_fabric",
        choices=["qdt", "legacy"],
        default="qdt",
        help="Working-memory implementation; QDT is the training default.",
    )
    p.add_argument(
        "--qdt_hardware_profile",
        choices=["compact", "single_gpu_8_12gb", "deep"],
        default="single_gpu_8_12gb",
    )
    p.add_argument("--qdt_num_slots", type=int, default=0)
    p.add_argument("--qdt_transformer_layers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--max_train_examples", type=int, default=0)
    p.add_argument("--max_test_examples", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10, help="Print live training stats every N steps")
    p.add_argument("--enable_diagnostics", action="store_true", help="Enable cortex diagnostics")
    p.add_argument("--disable_diagnostics", dest="enable_diagnostics", action="store_false")
    p.add_argument("--diag_log_path", default=None, help="Optional diagnostics JSONL path")
    p.add_argument("--diagnostics_flush_every", type=int, default=100)
    p.add_argument("--cps_stage", type=int, default=1, help="CPS curriculum stage (0..3)")
    p.add_argument("--cps_curriculum", action="store_true", help="Auto-progress CPS stage across epochs")
    p.add_argument("--warmup_ratio", type=float, default=0.10, help="Warmup fraction for dynamic LR")
    p.add_argument("--min_lr_ratio", type=float, default=0.08, help="Min LR as fraction of base LR")
    p.add_argument("--label_smoothing", type=float, default=0.03, help="Label smoothing for CE loss")
    p.add_argument("--target_grad_norm", type=float, default=0.8, help="Normalize gradients to this norm")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Safety clip after normalization")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--recall_loss_weight", type=float, default=0.12)
    p.add_argument("--cms_aux_weight", type=float, default=0.015)
    p.add_argument("--cms_senses", type=int, default=3)
    p.add_argument("--enable_task_decoder", action="store_true")
    p.add_argument("--disable_task_decoder", dest="enable_task_decoder", action="store_false")
    p.add_argument("--task_decoder_layers", type=int, default=4)
    p.add_argument("--task_decoder_heads", type=int, default=8)
    p.add_argument("--task_decoder_dropout", type=float, default=0.10)
    p.add_argument("--enable_full_fusion_stack", action="store_true")
    p.add_argument("--ltm_curved_hidden_dim", type=int, default=0)
    p.add_argument("--ltm_curved_hidden_mult", type=float, default=1.5)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.add_argument(
        "--amp_dtype",
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="CUDA autocast dtype; unsupported bf16 safely falls back to fp16.",
    )
    p.add_argument(
        "--report_capacity",
        action="store_true",
        help="Report literal parameter, gradient, optimizer, CPS rollback, and CUDA peak bytes.",
    )
    p.add_argument("--metrics_jsonl", default="logs/babi_metrics.jsonl")
    p.add_argument("--checkpoint_dir", default="logs/babi_checkpoints")
    p.add_argument("--checkpoint_every_epochs", type=int, default=1)
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--resume_strict", action="store_true")
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--target_accuracy", type=float, default=0.95)
    p.add_argument("--param_audit", action="store_true", help="Enable detailed parameter creation/content audit logging")
    p.add_argument("--param_audit_path", default="logs/babi_param_audit.jsonl", help="JSONL path for parameter audit logs")
    p.add_argument("--param_audit_max_dynamic", type=int, default=5000, help="Max dynamic creation events to record")
    p.add_argument("--enable_trainable_parameter_cps", action="store_true")
    p.add_argument("--trainable_cps_compress", action="store_true")
    p.add_argument("--trainable_cps_max_rank", type=int, default=8)
    p.add_argument("--trainable_cps_reconstruction_tolerance", type=float, default=1e-4)
    p.set_defaults(enable_task_decoder=True, use_amp=True, enable_diagnostics=True)
    return p


def main():
    p = build_arg_parser()
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    use_amp, amp_dtype, amp_dtype_name = _resolve_amp_dtype(
        device, bool(args.use_amp), str(args.amp_dtype)
    )
    sdpa_report = _configure_sdpa_backends(device)
    print(f"[babi] sdpa={sdpa_report}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "bAbI training requires Hugging Face datasets; install it with "
            "`python -m pip install datasets`."
        ) from exc
    print(f"[babi] loading source={args.dataset_source} config={args.config}")
    train_ex, val_ex, test_ex = load_babi_examples(
        load_dataset,
        source=str(args.dataset_source),
        config=str(args.config),
    )
    rng = random.Random(int(args.seed))
    rng.shuffle(train_ex)
    if int(args.max_train_examples) > 0:
        train_ex = train_ex[: int(args.max_train_examples)]
    if int(args.max_test_examples) > 0:
        test_ex = test_ex[: int(args.max_test_examples)]
    if len(train_ex) < 2:
        raise RuntimeError("bAbI train split must contain at least two examples")
    if val_ex is None:
        val_count = min(
            len(train_ex) - 1,
            max(1, int(round(len(train_ex) * float(args.val_ratio)))),
        )
        val_ex = train_ex[:val_count]
        train_ex = train_ex[val_count:]
    elif int(args.max_train_examples) > 0:
        # Keep smoke runs proportional to the reduced training sample.
        val_ex = val_ex[: max(1, min(len(val_ex), int(args.max_train_examples) // 5))]
    # Build vocabulary from the official training split only; the held-out test
    # set must not influence model construction.
    vocab = build_vocab(train_ex + val_ex, [])
    print(
        f"[babi] examples train={len(train_ex)} val={len(val_ex)} "
        f"test={len(test_ex)} vocab={len(vocab)}"
    )

    train_ds = BabiQADataset(train_ex, vocab, args.max_len)
    val_ds = BabiQADataset(val_ex, vocab, args.max_len)
    test_ds = BabiQADataset(test_ex, vocab, args.max_len)
    collate = partial(collate_pad, pad_id=vocab["<pad>"])
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "collate_fn": collate,
    }
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    model = CortexSeqModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        ltm_curved_hidden_dim=int(args.ltm_curved_hidden_dim),
        ltm_curved_hidden_mult=float(args.ltm_curved_hidden_mult),
        cms_enabled=True,
        cms_senses=int(args.cms_senses),
        cms_aux_weight=float(args.cms_aux_weight),
        recall_loss_weight=float(args.recall_loss_weight),
        task_decoder_enabled=bool(args.enable_task_decoder),
        task_decoder_layers=int(args.task_decoder_layers),
        task_decoder_heads=int(args.task_decoder_heads),
        task_decoder_dropout=float(args.task_decoder_dropout),
        task_decoder_use_sinusoidal=True,
        enable_full_fusion_stack=bool(args.enable_full_fusion_stack),
        ltm_enable_spatial_ltm=False,
        ltm_auto_wire_spatial=False,
        working_memory_fabric=str(args.working_memory_fabric),
        qdt_hardware_profile=str(args.qdt_hardware_profile),
        qdt_num_slots=int(args.qdt_num_slots),
        qdt_transformer_layers=int(args.qdt_transformer_layers),
        qdt_qspin_guarded_shadow=True,
        qdt_qspin_live_activation=False,
        qdt_qspin_live_kill_switch_enabled=True,
    ).to(device)
    fabric = model.cortex.describe_working_memory_fabric()
    print(
        f"[babi] working_memory={fabric['fabric']} "
        f"class={fabric['working_memory_class']} qspin_live=false",
        flush=True,
    )
    resume = None
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device)
        if str(resume.get("babi_config", args.config)) != str(args.config):
            raise ValueError("resume checkpoint uses a different bAbI configuration")
        if str(resume.get("dataset_source", args.dataset_source)) != str(
            args.dataset_source
        ):
            raise ValueError("resume checkpoint uses a different bAbI dataset source")
        saved_vocab = resume.get("vocab")
        if saved_vocab is not None and dict(saved_vocab) != vocab:
            raise ValueError("resume checkpoint vocabulary does not match loaded dataset")
        manifest = resume.get("trainable_parameter_cps_manifest")
        if manifest and hasattr(model, "prepare_trainable_parameter_cps_from_manifest"):
            model.prepare_trainable_parameter_cps_from_manifest(manifest)
            model.to(device)
    store = None
    if args.enable_trainable_parameter_cps:
        store = getattr(model, "trainable_parameter_cps", None)
        if store is None:
            store = model.enable_trainable_parameter_cps(
                TrainableParameterCPSConfig(
                    exclude=("*qspin*", "*trainable_parameter_cps*"),
                    enable_compression=bool(args.trainable_cps_compress),
                    max_rank=int(args.trainable_cps_max_rank),
                    reconstruction_tolerance=float(
                        args.trainable_cps_reconstruction_tolerance
                    ),
                )
            )
            store.commit(store.stage(model))
            if args.trainable_cps_compress:
                probe_tokens = (
                    torch.arange(8, device=device, dtype=torch.long)
                    .remainder(max(1, len(vocab)))
                    .view(1, -1)
                )
                was_training = model.training
                model.eval()

                def compression_probe(root):
                    with torch.no_grad():
                        return root(probe_tokens)

                evaluation = store.compress_committed(probe=compression_probe)
                model.train(was_training)
                print(
                    "[babi] trainable_cps_compression_probe="
                    f"applied={evaluation.compression_applied} "
                    f"max_output_error={evaluation.max_output_error}",
                    flush=True,
                )
        model.to(device)
        print(f"[babi] trainable_cps={store.capacity_report()}", flush=True)
    audit = None
    if args.param_audit:
        audit = ParameterAuditLogger(
            log_path=args.param_audit_path,
            max_dynamic_events=args.param_audit_max_dynamic,
        )
        inv = audit.log_model_inventory(model, tag="babi_startup", include_preview=True)
        audit.attach_to_cortex(model.cortex)
        print(
            f"[babi] param_audit enabled path={args.param_audit_path} "
            f"total={inv['total_params']} trainable={inv['trainable_params']}",
            flush=True,
        )
    # Recommended current policy for this stack.
    if hasattr(model, "cortex") and hasattr(model.cortex, "topology") and hasattr(model.cortex.topology, "register_babi_qhm_v2_policy"):
        try:
            model.cortex.topology.register_babi_qhm_v2_policy()
            model.cortex.apply_topology_policy("babi_qhm_v2")
        except Exception as exc:
            print(f"[babi][warn] failed to apply babi_qhm_v2 topology policy: {exc}", flush=True)
    if hasattr(model.cortex, "set_cps_curriculum_stage"):
        model.cortex.set_cps_curriculum_stage(int(args.cps_stage))
    if hasattr(model.cortex, "enable_diagnostics"):
        model.cortex.enable_diagnostics(
            enabled=bool(args.enable_diagnostics),
            log_path=args.diag_log_path,
            flush_every=int(args.diagnostics_flush_every),
        )

    if resume is not None:
        _load_model_state(
            model,
            resume.get("model_state_dict", {}),
            strict=bool(args.resume_strict),
        )
    opt = build_optimizer(
        model.parameters(),
        OptimizerConfig(
            name="adamw",
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            warmup_ratio=float(args.warmup_ratio),
            min_lr_ratio=float(args.min_lr_ratio),
        ),
    )
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = max(1, args.epochs * updates_per_epoch)
    scheduler = build_warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = _grad_scaler(bool(use_amp and amp_dtype == torch.float16))
    if args.report_capacity:
        print_literal_training_bytes(
            "startup",
            model,
            optimizer=opt,
            cps_store=store,
            cuda_device=device,
        )
    start_epoch = 1
    global_step = 0
    best_acc = -1.0
    if resume is not None:
        if not args.evaluate_only:
            if resume.get("optimizer_state_dict"):
                try:
                    load_optimizer_state_dict_checked(
                        opt,
                        resume["optimizer_state_dict"],
                        saved_layout=resume.get("optimizer_parameter_layout"),
                    )
                except RuntimeError as exc:
                    print(
                        f"[babi][warn] refusing stale optimizer state after CPS/layout change: {exc}",
                        flush=True,
                    )
            if resume.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            if resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        global_step = int(resume.get("global_step", 0))
        best_acc = float(resume.get("best_val_acc", -1.0))

    print(
        f"[babi] training epochs={args.epochs} batch={args.batch_size} d_model={args.d_model} "
        f"lr={args.lr} wd={args.weight_decay} device={device} "
        f"warmup={args.warmup_ratio} min_lr_ratio={args.min_lr_ratio} "
        f"label_smoothing={args.label_smoothing} amp={use_amp} "
        f"amp_dtype={amp_dtype_name} "
        f"grad_accum={grad_accum_steps} task_decoder={args.enable_task_decoder}"
    )
    _print_metric_legend()
    if args.evaluate_only:
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            device,
            pad_id=vocab["<pad>"],
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        print(
            f"[babi] evaluate_only test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}",
            flush=True,
        )
        return

    best = None
    stale_epochs = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        running_correct = 0
        running_seen = 0
        if args.cps_curriculum and hasattr(model.cortex, "set_cps_curriculum_stage"):
            stage = min(3, int(((ep - 1) * 4) / max(1, args.epochs)))
            model.cortex.set_cps_curriculum_stage(stage)
        opt.zero_grad(set_to_none=True)
        pre_norm_v = 0.0
        grad_norm_v = 0.0
        for step, (src, y) in enumerate(train_loader, start=1):
            src = src.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast(device, use_amp, amp_dtype):
                logits, aux = model(src, return_aux_losses=True)
                last_logits = _answer_logits(logits, src, vocab["<pad>"])
                loss = F.cross_entropy(
                    last_logits,
                    y,
                    label_smoothing=float(args.label_smoothing),
                )
                recall_loss_val = 0.0
                cms_loss_val = 0.0
                if isinstance(aux.get("recall_loss", None), torch.Tensor):
                    recall_loss_val = float(aux["recall_loss"].detach().item())
                    loss = loss + model.recall_loss_weight * aux["recall_loss"]
                if isinstance(aux.get("cms_loss", None), torch.Tensor):
                    cms_loss_val = float(aux["cms_loss"].detach().item())
                    loss = loss + aux["cms_loss"]
            raw_loss = loss
            scaler.scale(loss / grad_accum_steps).backward()
            should_update = (
                step % grad_accum_steps == 0 or step == len(train_loader)
            )
            if should_update:
                if use_amp:
                    _clear_complex_grads(model.parameters())
                scaler.unscale_(opt)
                pre_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1e9,
                )
                pre_norm_v = float(pre_norm)
                if pre_norm_v > 0.0 and math.isfinite(pre_norm_v):
                    grad_scale = float(args.target_grad_norm) / (pre_norm_v + 1e-8)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(grad_scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(args.grad_clip),
                )
                grad_norm_v = float(grad_norm)
                if args.report_capacity and global_step == 0:
                    print_literal_training_bytes(
                        "first_backward",
                        model,
                        optimizer=opt,
                        cps_store=store,
                        cuda_device=device,
                    )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if hasattr(model, "cortex") and hasattr(model.cortex, "topology") and hasattr(model.cortex, "topology_step"):
                loss_val = float(raw_loss.detach().cpu())
                model.cortex.topology_step(loss_val)
                if hasattr(model.cortex, "step_topology_schedulers"):
                    telem = _build_topology_telemetry(model.cortex)
                    model.cortex.step_topology_schedulers(loss_val, telemetry_by_bank=telem)
            if model.cortex.consolidated_lexicon is not None:
                model.cortex.consolidated_lexicon.renorm_constraints_()
            pred = last_logits.argmax(dim=-1)
            batch_correct = int((pred == y).sum().item())
            running_correct += batch_correct
            running_seen += int(src.size(0))
            total += raw_loss.item() * src.size(0)
            seen += src.size(0)
            if step % max(1, int(args.log_every)) == 0:
                run_acc = running_correct / max(1, running_seen)
                mean_loss = total / max(1, seen)
                topo_fit = 0.0
                if hasattr(model, "cortex") and hasattr(model.cortex, "topology"):
                    topo_fit = float(getattr(model.cortex.topology, "fitness_ema", 0.0) or 0.0)
                lr_now = float(opt.param_groups[0]["lr"])
                m = model.cortex.get_metrics() if hasattr(model, "cortex") else {}
                router_line = _format_router_snapshot(m)
                ltm_router_line = _format_ltm_router_snapshot(m)
                print(
                    f"[train ep={ep:02d} step={step:04d}/{len(train_loader):04d}] "
                    f"loss={float(raw_loss.detach().item()):.4f} mean={mean_loss:.4f} "
                    f"acc={run_acc:.4f} recall={recall_loss_val:.4f} cms={cms_loss_val:.4f} "
                    f"grad={grad_norm_v:.4f} pre_grad={pre_norm_v:.4f} lr={lr_now:.6f} "
                    f"topo_fit={topo_fit:.4f} gstep={global_step}",
                    flush=True,
                )
                if audit is not None:
                    print(
                        f"[param_audit] dynamic_creation_events={audit.dynamic_events} "
                        f"log_path={args.param_audit_path}",
                        flush=True,
                    )
                if router_line:
                    print(f"[router step ep={ep:02d} step={step:04d}] {router_line}", flush=True)
                if ltm_router_line:
                    print(f"[ltm_router step ep={ep:02d} step={step:04d}] {ltm_router_line}", flush=True)

        train_loss = total / max(1, seen)
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            pad_id=vocab["<pad>"],
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        m_epoch = model.cortex.get_metrics() if hasattr(model, "cortex") else {}
        router_epoch_line = _format_router_snapshot(m_epoch)
        ltm_router_epoch_line = _format_ltm_router_snapshot(m_epoch)
        print(
            f"[epoch {ep:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        epoch_metrics = {
            "event": "epoch",
            "epoch": int(ep),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(opt.param_groups[0]["lr"]),
            "working_memory_fabric": str(args.working_memory_fabric),
            "qspin_live_activation": False,
        }
        _append_jsonl(args.metrics_jsonl, epoch_metrics)
        if router_epoch_line:
            print(f"[router epoch {ep:02d}] {router_epoch_line}", flush=True)
        if ltm_router_epoch_line:
            print(f"[ltm_router epoch {ep:02d}] {ltm_router_epoch_line}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best = (ep, train_loss, val_loss, val_acc)
            stale_epochs = 0
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=ep,
                    global_step=global_step,
                    best_val_acc=best_acc,
                    vocab=vocab,
                    args=args,
                ),
                str(checkpoint_dir / "best.pt"),
            )
        else:
            stale_epochs += 1
        if int(args.checkpoint_every_epochs) > 0 and (
            ep % int(args.checkpoint_every_epochs) == 0
        ):
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=ep,
                    global_step=global_step,
                    best_val_acc=best_acc,
                    vocab=vocab,
                    args=args,
                ),
                str(checkpoint_dir / "last.pt"),
            )
        if val_acc >= float(args.target_accuracy):
            print(
                f"[babi] target accuracy reached ({val_acc:.4f}); stopping.",
                flush=True,
            )
            break
        if int(args.early_stop_patience) > 0 and stale_epochs >= int(
            args.early_stop_patience
        ):
            print(
                f"[babi] early stopping after {stale_epochs} stale epochs.",
                flush=True,
            )
            break

    if best is not None:
        ep, tr, vl, acc = best
        print(
            f"[babi] best epoch={ep} train_loss={tr:.4f} "
            f"val_loss={vl:.4f} val_acc={acc:.4f}"
        )
    best_path = checkpoint_dir / "best.pt"
    if best is not None and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        _load_model_state(
            model,
            best_checkpoint.get("model_state_dict", {}),
            strict=True,
        )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device,
        pad_id=vocab["<pad>"],
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    print(
        f"[babi] final_test loss={test_loss:.4f} acc={test_acc:.4f}",
        flush=True,
    )
    _append_jsonl(
        args.metrics_jsonl,
        {
            "event": "final_test",
            "global_step": int(global_step),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "best_val_acc": float(best_acc),
        },
    )
    if hasattr(model, "flush_cms_logger"):
        model.flush_cms_logger()
    if hasattr(model.cortex, "flush_diagnostics"):
        model.cortex.flush_diagnostics()
    if args.report_capacity:
        print_literal_training_bytes(
            "final",
            model,
            optimizer=opt,
            cps_store=store,
            cuda_device=device,
        )


if __name__ == "__main__":
    main()
