import argparse
import contextlib
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable when executed as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_DIR = os.path.abspath(os.path.dirname(__file__))
if TOOL_DIR in sys.path:
    sys.path.remove(TOOL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.models import CortexSeqModel
from benchmark.tasks import CopyTask, TOK2IDX, VOCAB_SIZE, collate_fn
from data.copy_task_dataloader import (
    CopyTaskDataConfig,
    CopyTaskMasteryModel,
    align_logits_targets,
    build_copy_task_dataloader,
)
from mnemonic_cortex.optimizer import OptimizerConfig, build_optimizer, build_warmup_cosine_scheduler


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _flatten_metrics(metrics: Dict) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        key = str(k)
        if isinstance(v, bool):
            out[key] = float(1.0 if v else 0.0)
        elif isinstance(v, (float, int)):
            out[key] = float(v)
        elif isinstance(v, str):
            out[key] = v
        elif isinstance(v, torch.Tensor):
            out[key] = _safe_mean_scalar(v, 0.0)
        elif isinstance(v, (list, tuple, dict)):
            try:
                out[f"{key}_json"] = json.dumps(v, ensure_ascii=True)
            except Exception:
                out[f"{key}_json"] = str(v)
        else:
            out[key] = str(v)
    return out


def _unpack_copy_batch(batch):
    if isinstance(batch, dict):
        return batch["src"], batch["tgt"]
    return batch


def _copy_task_cfg_from_args(
    args,
    *,
    n_samples: int,
    max_len: int,
    shuffle: bool = True,
    epoch: int = 1,
) -> CopyTaskDataConfig:
    return CopyTaskDataConfig(
        n_samples=int(n_samples),
        min_len=int(getattr(args, "copy_min_len", 1)),
        max_len=int(max_len),
        fixed_len=int(getattr(args, "copy_fixed_len", 0)),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        num_workers=int(getattr(args, "copy_num_workers", 0)),
        pin_memory=bool(getattr(args, "copy_pin_memory", True)),
        shuffle=bool(shuffle),
        val_ratio=float(getattr(args, "copy_val_ratio", 0.0)),
        curriculum_enabled=bool(getattr(args, "copy_curriculum_enabled", False)),
        curriculum_start_len=int(getattr(args, "copy_curriculum_start_len", 0) or args.len8),
        curriculum_end_len=int(getattr(args, "copy_curriculum_end_len", 0) or args.len16),
        curriculum_ramp_epochs=int(getattr(args, "copy_curriculum_ramp_epochs", 0) or args.len8_epochs),
        noise_prob=float(getattr(args, "copy_noise_prob", 0.0)),
        replace_prob=float(getattr(args, "copy_replace_prob", 0.0)),
        repeat_factor=int(getattr(args, "copy_repeat_factor", 1)),
        delayed_copy_gap=int(getattr(args, "copy_delayed_gap", 0)),
        vocab_mode=str(getattr(args, "copy_vocab_mode", "full")),
        enable_sinusoidal_encoder=bool(getattr(args, "copy_enable_sin_encoder", True)),
        enable_sinusoidal_decoder=bool(getattr(args, "copy_enable_sin_decoder", True)),
        encoder_d_model=int(getattr(args, "copy_encoder_d_model", 0) or args.d_model),
        encoder_layers=int(getattr(args, "copy_encoder_layers", 2)),
        encoder_heads=int(getattr(args, "copy_encoder_heads", 4)),
        encoder_dropout=float(getattr(args, "copy_encoder_dropout", 0.1)),
        decoder_layers=int(getattr(args, "copy_decoder_layers", 2)),
        decoder_heads=int(getattr(args, "copy_decoder_heads", 4)),
        decoder_dropout=float(getattr(args, "copy_decoder_dropout", 0.1)),
        mastery_loss_weight=float(getattr(args, "copy_mastery_loss_weight", 0.25)),
        return_position_ids=True,
        return_sinusoidal_features=bool(getattr(args, "copy_return_sin_features", False)),
    )


def _build_copy_task_loader(args, *, n_samples: int, max_len: int, shuffle: bool, epoch: int = 1):
    cfg = _copy_task_cfg_from_args(args, n_samples=n_samples, max_len=max_len, shuffle=shuffle, epoch=epoch)
    split = "all" if float(cfg.val_ratio) <= 0.0 else ("train" if shuffle else "val")
    return build_copy_task_dataloader(cfg, epoch=epoch, split=split)


def _build_fixed_copytask_dataset(n_samples: int, max_len: int, seed: int) -> CopyTask:
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    ds = CopyTask(n_samples=n_samples, max_len=max_len)
    random.setstate(py_state)
    torch.random.set_rng_state(torch_state)
    return ds


def _align_logits_targets(logits: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align logits/targets for copy-task style datasets where target may include an
    extra BOS/SOS token (target length = logits length + 1).
    """
    if tgt.size(1) == logits.size(1) + 1:
        L = min(logits.size(1), tgt.size(1) - 1)
        return logits[:, :L, :], tgt[:, 1 : 1 + L]
    L = min(logits.size(1), tgt.size(1))
    return logits[:, :L, :], tgt[:, :L]


def _move_qh_codebooks_to_device(model, device: torch.device) -> int:
    """
    Some QH codebook tensors are stored in plain dicts, not module buffers.
    Force-move them to target device to avoid CPU/CUDA mismatch at runtime.
    """
    moved = 0

    def _move_codebook(codebook):
        nonlocal moved
        if codebook is None:
            return
        for attr in ("depth_codes", "bank_codes", "triplet_codes", "slot_codes"):
            table = getattr(codebook, attr, None)
            if not isinstance(table, dict):
                continue
            for k, v in list(table.items()):
                if isinstance(v, torch.Tensor) and v.device != device:
                    table[k] = v.to(device)
                    moved += 1

    for mod in model.modules():
        qh_slot_bank = getattr(mod, "qh_slot_bank", None)
        if qh_slot_bank is not None:
            _move_codebook(getattr(qh_slot_bank, "codebook", None))
        qh_banks = getattr(mod, "qh_banks", None)
        if isinstance(qh_banks, torch.nn.ModuleDict):
            for bank in qh_banks.values():
                _move_codebook(getattr(bank, "codebook", None))

    return moved


def _disable_fusion_paths(model) -> None:
    """
    Disable fusion paths across LTM, WM<->LTM bridge, and MANN bridge.
    """
    cortex = getattr(model, "cortex", None)
    if cortex is None:
        return

    ltm = getattr(cortex, "long_term_memory", None)
    if ltm is not None:
        # Disable inter-memory exchange and LTM fusion by selecting HG stream directly.
        if hasattr(ltm, "_inter_memory_exchange"):
            ltm._inter_memory_exchange = lambda rhg, rcg, rcv: (rhg, rcg, rcv)
        if hasattr(ltm, "_fuse"):
            ltm._fuse = lambda rhg, rcg, rcv, routing_weights=None: rhg
        if hasattr(ltm, "inter_exchange_gate"):
            with torch.no_grad():
                ltm.inter_exchange_gate.fill_(-20.0)

    # Disable WM<->LTM bridge fusion.
    if hasattr(cortex, "_bridge_wm_ltm"):
        cortex._bridge_wm_ltm = lambda wm_seq, ltm_seq, phase: ltm_seq
    if hasattr(cortex, "mem_bridge_gate"):
        with torch.no_grad():
            cortex.mem_bridge_gate.fill_(-20.0)

    # Disable reasoning MANN bridge.
    if hasattr(cortex, "enable_reasoning_controller_bridge"):
        cortex.enable_reasoning_controller_bridge(enabled=False)
    else:
        cortex.reasoning_controller_api = None


def _restore_fusion_gates(model) -> Dict[str, float]:
    """
    Restore gate parameters that may have been hard-disabled in no-fusion checkpoints.
    Returns a dict of reset gate names -> new value.
    """
    restored: Dict[str, float] = {}
    cortex = getattr(model, "cortex", None)
    if cortex is None:
        return restored
    ltm = getattr(cortex, "long_term_memory", None)
    if ltm is not None and hasattr(ltm, "inter_exchange_gate"):
        with torch.no_grad():
            gate_p = ltm.inter_exchange_gate
            gate_v = float(torch.sigmoid(gate_p).detach().item())
            # Anything this close to zero almost certainly came from no-fusion hard disable.
            if gate_v < 1e-4:
                gate_p.fill_(0.20)
                restored["ltm_inter_exchange_gate"] = 0.20
    if hasattr(cortex, "mem_bridge_gate"):
        with torch.no_grad():
            gate_p = cortex.mem_bridge_gate
            gate_v = float(torch.sigmoid(gate_p).detach().item())
            if gate_v < 1e-4:
                gate_p.fill_(0.22)
                restored["wm_ltm_bridge_gate"] = 0.22
    return restored


def _relax_topology_mutation(cortex, factor: float = 0.35) -> None:
    """
    Reduce topology mutation aggressiveness for later epochs.
    """
    if cortex is None or not hasattr(cortex, "topology"):
        return
    topo = cortex.topology
    active = getattr(topo, "active_policy", None)
    if active is None or not hasattr(topo, "policies") or active not in topo.policies:
        return

    pol = dict(topo.policies[active])
    pol["curvature_rate"] = float(pol.get("curvature_rate", 0.004)) * float(factor)
    adaptive = dict(pol.get("adaptive", {}))
    if "step" in adaptive:
        adaptive["step"] = float(adaptive["step"]) * float(factor)
    if "b_hi" in adaptive and "b_lo" in adaptive:
        span = float(adaptive["b_hi"]) - float(adaptive["b_lo"])
        adaptive["b_hi"] = float(adaptive["b_lo"]) + span * max(0.2, float(factor))
    pol["adaptive"] = adaptive
    topo.policies[active] = pol
    topo.apply_to_model(cortex)


def _save_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch: int,
    global_step: int,
    best_acc: float,
    best_record,
):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_acc": float(best_acc),
        "best_record": best_record,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(payload, path)


def _append_success_memory(path: str, payload: Dict):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _resolve_best_metric(summary: Dict[str, float], name: str, alpha: float) -> float:
    key = str(name).strip().lower()
    if key == "val_acc_curriculum_len":
        return float(summary.get("val_acc_curriculum_len", 0.0))
    if key == "neg_val_loss_curriculum_len":
        return -float(summary.get("val_loss_curriculum_len", 1e9))
    if key == "composite":
        return float(summary.get("val_acc_curriculum_len", 0.0)) - float(alpha) * float(
            summary.get("val_loss_curriculum_len", 0.0)
        )
    return float(summary.get("val_acc_curriculum_len", 0.0))


def _auto_select_best_checkpoint(preferred_best_out: str, scope_dir: str = "") -> str:
    """
    Prefer the run's configured best-checkpoint path if present.
    Otherwise, fall back to the newest *_best.pt in logs/checkpoints.
    """
    if preferred_best_out:
        preferred_abs = os.path.abspath(preferred_best_out)
        if os.path.isfile(preferred_abs):
            return preferred_abs
    scan_dir = str(scope_dir).strip() if str(scope_dir).strip() else os.path.join("logs", "checkpoints")
    candidates = glob.glob(os.path.join(scan_dir, "*_best.pt"))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.abspath(candidates[0])


def _load_state_dict_compatible(model: torch.nn.Module, incoming: Dict[str, torch.Tensor], strict: bool) -> Dict[str, int]:
    """
    Load checkpoint weights while tolerating shape mismatches when strict=False.
    Returns counts for loaded / skipped keys.
    """
    if strict:
        model.load_state_dict(incoming, strict=True)
        return {"loaded": len(incoming), "skipped": 0}

    current = model.state_dict()
    compatible = {}
    skipped = 0
    for k, v in incoming.items():
        if k not in current:
            skipped += 1
            continue
        cur_v = current[k]
        if getattr(cur_v, "shape", None) != getattr(v, "shape", None):
            skipped += 1
            continue
        compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped}


def _evaluate(model, loader, device, *, use_aligned_targets: bool = False) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct_tok = 0
    total_seq = 0
    correct_seq = 0
    with torch.no_grad():
        for batch in loader:
            src, tgt = _unpack_copy_batch(batch)
            src = src.to(device)
            tgt = tgt.to(device)
            logits = model(src)
            if use_aligned_targets:
                logits, tgt = align_logits_targets(logits, tgt)
            else:
                logits, tgt = _align_logits_targets(logits, tgt)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=TOK2IDX["<pad>"],
            )
            preds = logits.argmax(dim=-1)
            mask = tgt != TOK2IDX["<pad>"]
            correct_tok += (preds == tgt).masked_select(mask).sum().item()
            total_tok += mask.sum().item()
            valid_rows = mask.any(dim=1)
            row_ok = ((preds == tgt) | (~mask)).all(dim=1)
            correct_seq += row_ok.masked_select(valid_rows).sum().item()
            total_seq += valid_rows.sum().item()
            total_loss += loss.item()
    return (
        total_loss / max(1, len(loader)),
        correct_tok / max(1, total_tok),
        correct_seq / max(1, total_seq),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--total_steps", type=int, default=500)
    p.add_argument("--len8_epochs", type=int, default=3)
    p.add_argument("--len8", type=int, default=8)
    p.add_argument("--len16", type=int, default=16)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=160)
    p.add_argument("--ltm_curved_hidden_dim", type=int, default=0)
    p.add_argument("--ltm_curved_hidden_mult", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--min_lr_ratio", type=float, default=0.08)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--target_grad_norm", type=float, default=0.8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_samples_per_epoch", type=int, default=4096)
    p.add_argument("--eval_samples", type=int, default=1024)
    p.add_argument("--metrics_jsonl", default="logs/copy_task_gpu_metrics.jsonl")
    p.add_argument("--recall_loss_weight", type=float, default=0.12)
    p.add_argument("--cms_aux_weight", type=float, default=0.015)
    p.add_argument("--disable_fusion", action="store_true")
    p.add_argument("--topology_relax_after_epoch", type=int, default=0)
    p.add_argument("--topology_relax_factor", type=float, default=0.35)
    p.add_argument("--phase1_lr_mult", type=float, default=1.0)
    p.add_argument("--phase2_lr_mult", type=float, default=1.0)
    p.add_argument("--phase_transition_epoch", type=int, default=0)
    p.add_argument("--phase_transition_step", type=int, default=0)
    p.add_argument("--len16_loss_weight", type=float, default=1.0)
    p.add_argument("--grad_noise_std", type=float, default=0.0)
    p.add_argument("--topology_freeze_after_epoch", type=int, default=0)
    p.add_argument("--best_metric", default="val_acc_curriculum_len")
    p.add_argument("--best_metric_alpha", type=float, default=0.05)
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--resume_weights_only", action="store_true")
    p.add_argument("--resume_strict", action="store_true")
    p.add_argument("--resume_mode", choices=["weights_only", "full"], default="")
    p.add_argument("--resume_scope_dir", default="")
    p.add_argument("--auto_resume_best", action="store_true")
    p.add_argument("--no_auto_resume_best", dest="auto_resume_best", action="store_false")
    p.add_argument("--auto_resume_weights_only", action="store_true")
    p.add_argument("--no_auto_resume_weights_only", dest="auto_resume_weights_only", action="store_false")
    p.add_argument("--enable_full_fusion_stack", action="store_true")
    p.add_argument("--full_stack_enable_advanced", action="store_true")
    p.add_argument("--full_stack_disable_advanced", action="store_true")
    p.add_argument("--full_stack_enable_qdt_wm_bridge", action="store_true")
    p.add_argument("--full_stack_disable_qdt_wm_bridge", action="store_true")
    p.add_argument("--full_stack_enable_shared_memory", action="store_true")
    p.add_argument("--full_stack_disable_shared_memory", action="store_true")
    p.add_argument("--enable_task_decoder", action="store_true")
    p.add_argument("--task_decoder_layers", type=int, default=4)
    p.add_argument("--task_decoder_heads", type=int, default=8)
    p.add_argument("--task_decoder_dropout", type=float, default=0.1)
    p.add_argument("--task_decoder_use_sinusoidal", action="store_true")
    p.add_argument("--adaptive_lr_enabled", action="store_true")
    p.add_argument("--adaptive_lr_up", type=float, default=1.015)
    p.add_argument("--adaptive_lr_down", type=float, default=0.82)
    p.add_argument("--adaptive_lr_min_mult", type=float, default=0.35)
    p.add_argument("--adaptive_lr_max_mult", type=float, default=1.9)
    p.add_argument("--adaptive_lr_patience", type=int, default=15)
    p.add_argument("--adaptive_lr_acc_threshold", type=float, default=0.12)
    p.add_argument("--adaptive_lr_loss_tolerance", type=float, default=0.003)
    p.add_argument("--success_memory_path", default="logs/success_memories.jsonl")
    p.add_argument("--success_acc_threshold", type=float, default=0.10)
    p.add_argument("--success_memory_cooldown_steps", type=int, default=25)
    p.add_argument("--checkpoint_out", default="")
    p.add_argument("--checkpoint_best_out", default="")
    p.add_argument("--enable_diagnostics", action="store_true")
    p.add_argument("--disable_diagnostics", dest="enable_diagnostics", action="store_false")
    p.add_argument("--diagnostics_log_path", default="")
    p.add_argument("--diagnostics_flush_every", type=int, default=100)
    p.add_argument("--accept_min_val_acc", type=float, default=-1.0)
    p.add_argument("--accept_max_val_loss", type=float, default=-1.0)
    p.add_argument("--accept_min_diag_events", type=float, default=-1.0)
    p.add_argument("--use_copy_task_dataloader", action="store_true")
    p.add_argument("--enable_copy_mastery", action="store_true")
    p.add_argument("--copy_min_len", type=int, default=1)
    p.add_argument("--copy_fixed_len", type=int, default=0)
    p.add_argument("--copy_num_workers", type=int, default=0)
    p.add_argument("--copy_pin_memory", action="store_true")
    p.add_argument("--copy_val_ratio", type=float, default=0.0)
    p.add_argument("--copy_curriculum_enabled", action="store_true")
    p.add_argument("--copy_curriculum_start_len", type=int, default=0)
    p.add_argument("--copy_curriculum_end_len", type=int, default=0)
    p.add_argument("--copy_curriculum_ramp_epochs", type=int, default=0)
    p.add_argument("--copy_noise_prob", type=float, default=0.0)
    p.add_argument("--copy_replace_prob", type=float, default=0.0)
    p.add_argument("--copy_repeat_factor", type=int, default=1)
    p.add_argument("--copy_delayed_gap", type=int, default=0)
    p.add_argument("--copy_vocab_mode", choices=["full", "alnum", "digits", "custom"], default="full")
    p.add_argument("--copy_enable_sin_encoder", action="store_true")
    p.add_argument("--copy_enable_sin_decoder", action="store_true")
    p.add_argument("--copy_encoder_d_model", type=int, default=0)
    p.add_argument("--copy_encoder_layers", type=int, default=2)
    p.add_argument("--copy_encoder_heads", type=int, default=4)
    p.add_argument("--copy_encoder_dropout", type=float, default=0.1)
    p.add_argument("--copy_decoder_layers", type=int, default=2)
    p.add_argument("--copy_decoder_heads", type=int, default=4)
    p.add_argument("--copy_decoder_dropout", type=float, default=0.1)
    p.add_argument("--copy_mastery_loss_weight", type=float, default=0.25)
    p.add_argument("--copy_return_sin_features", action="store_true")
    p.set_defaults(copy_enable_sin_encoder=True, copy_enable_sin_decoder=True)
    p.set_defaults(auto_resume_best=True, auto_resume_weights_only=True)
    p.set_defaults(enable_diagnostics=True)
    args = p.parse_args()

    seed_all(args.seed)
    use_cuda = str(args.device).startswith("cuda")
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.metrics_jsonl), exist_ok=True)

    mastery_model = None
    if bool(args.enable_copy_mastery):
        mastery_cfg = _copy_task_cfg_from_args(
            args,
            n_samples=max(64, int(args.train_samples_per_epoch)),
            max_len=int(args.len8),
            shuffle=True,
        )
        mastery_model = CopyTaskMasteryModel(mastery_cfg, vocab_size=VOCAB_SIZE).to(device)

    model = CortexSeqModel(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        ltm_curved_hidden_dim=int(args.ltm_curved_hidden_dim),
        ltm_curved_hidden_mult=float(args.ltm_curved_hidden_mult),
        cms_enabled=True,
        cms_senses=3,
        cms_aux_weight=float(args.cms_aux_weight),
        recall_loss_weight=float(args.recall_loss_weight),
        task_decoder_enabled=bool(args.enable_task_decoder),
        task_decoder_layers=int(args.task_decoder_layers),
        task_decoder_heads=int(args.task_decoder_heads),
        task_decoder_dropout=float(args.task_decoder_dropout),
        task_decoder_use_sinusoidal=bool(args.task_decoder_use_sinusoidal),
    ).to(device)
    if args.disable_fusion:
        _disable_fusion_paths(model)
    if hasattr(model, "cortex") and hasattr(model.cortex, "enable_diagnostics"):
        model.cortex.enable_diagnostics(
            enabled=bool(args.enable_diagnostics),
            log_path=str(args.diagnostics_log_path) if str(args.diagnostics_log_path).strip() else None,
            flush_every=int(args.diagnostics_flush_every),
        )
    full_stack_advanced_enabled = bool(args.enable_full_fusion_stack and not args.disable_fusion)
    full_stack_qdt_enabled = bool(args.enable_full_fusion_stack and not args.disable_fusion)
    full_stack_shared_enabled = bool(args.enable_full_fusion_stack and not args.disable_fusion)
    if args.full_stack_enable_qdt_wm_bridge:
        full_stack_qdt_enabled = True
    if args.full_stack_disable_qdt_wm_bridge:
        full_stack_qdt_enabled = False
    if args.full_stack_enable_shared_memory:
        full_stack_shared_enabled = True
    if args.full_stack_disable_shared_memory:
        full_stack_shared_enabled = False
    if args.full_stack_enable_advanced:
        full_stack_advanced_enabled = True
    if args.full_stack_disable_advanced:
        full_stack_advanced_enabled = False
    if args.enable_full_fusion_stack and not args.disable_fusion:
        try:
            model.cortex.enable_cps_cms_full_stack(
                vocab_size=VOCAB_SIZE,
                cms_senses=3,
                enable_broker=True,
                enable_advanced=bool(full_stack_advanced_enabled),
                enable_reasoning_bridge=True,
                enable_qdt_wm_bridge=bool(full_stack_qdt_enabled),
            )
            if full_stack_shared_enabled and hasattr(model.cortex, "enable_shared_memory_subsystem"):
                model.cortex.enable_shared_memory_subsystem()
            # Newly attached modules default to CPU; move full model back to target device.
            model = model.to(device)
        except Exception as exc:
            print(f"[copy][warn] full fusion stack enable failed: {exc}", flush=True)
    qh_moved = _move_qh_codebooks_to_device(model, device)
    if qh_moved > 0:
        print(f"[copy] moved_qh_codebook_tensors={qh_moved} to device={device}", flush=True)

    # Tailored topology policy for copy task: stable low-noise retrieval + adaptive geometry.
    if hasattr(model, "cortex") and hasattr(model.cortex, "topology"):
        topo = model.cortex.topology
        topo.register_policy(
            "copy_task_tailored",
            micro_b=0.014,
            micro_b_max=0.045,
            omega_max=0.16,
            temp_scale=0.95,
            use_heat_kernel=True,
            allowed_channels=[1, 1, 1, 1, 1, 1],
            gate_bias=[0.35, 0.25, 0.20, 0.12, 0.04, 0.00],
            curvature_mode="mix",
            curvature_rate=0.004,
            curvature_clip=1.25,
            curvature_mix=(0.45, 0.25, 0.20, 0.10),
            qhm_enable=True,
            qhm_temp=0.88,
            qhm_phase_noise=0.015,
            qhm_lightbulb=0.94,
            qhm_explosive_temp=0.52,
            qhm_explosive_alpha=0.70,
            qhm_lb_z_on=2.1,
            qhm_lb_z_off=1.3,
            qhm_lb_cooldown_steps=8,
            qhm_lb_max_stage2_steps=2,
            qhm_lb_budget_per_100=5,
            qhm_lb_prefocus_temp_mult=0.90,
            qhm_lb_prefocus_alpha_boost=0.10,
            qhm_lb_alpha_max=0.85,
            qhm_lb_temp_min=0.45,
            adaptive=dict(
                b_lo=0.010,
                b_hi=0.026,
                omega_lo=0.09,
                omega_hi=0.22,
                fit_up=0.83,
                fit_down=0.58,
                step=0.0015,
            ),
        )
        model.cortex.apply_topology_policy("copy_task_tailored")

    opt_params = list(model.parameters())
    if mastery_model is not None:
        opt_params += list(mastery_model.parameters())
    opt = build_optimizer(
        opt_params,
        OptimizerConfig(
            name="adamw",
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            warmup_ratio=float(args.warmup_ratio),
            min_lr_ratio=float(args.min_lr_ratio),
        ),
    )
    scheduler = build_warmup_cosine_scheduler(
        opt,
        total_steps=max(1, int(args.total_steps)),
        warmup_ratio=float(args.warmup_ratio),
        min_lr_ratio=float(args.min_lr_ratio),
    )

    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(1, args.total_steps // args.epochs)
    global_step = 0
    best_acc = -1.0
    best_record = None
    best_metric_value = -1e18

    if str(args.resume_mode).strip().lower() == "weights_only":
        args.resume_weights_only = True
    elif str(args.resume_mode).strip().lower() == "full":
        args.resume_weights_only = False

    if not args.resume_checkpoint and bool(args.auto_resume_best):
        auto_ckpt = _auto_select_best_checkpoint(
            str(args.checkpoint_best_out),
            scope_dir=str(args.resume_scope_dir),
        )
        if auto_ckpt:
            args.resume_checkpoint = auto_ckpt
            if bool(args.auto_resume_weights_only):
                args.resume_weights_only = True
            print(
                f"[copy] auto_resume_best selected checkpoint={args.resume_checkpoint} "
                f"weights_only={bool(args.resume_weights_only)}",
                flush=True,
            )
        else:
            scan_dir = str(args.resume_scope_dir).strip() or os.path.join("logs", "checkpoints")
            print(
                f"[copy] auto_resume_best enabled but no *_best.pt checkpoint found in {scan_dir}",
                flush=True,
            )

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        strict = bool(args.resume_strict)
        load_stats = _load_state_dict_compatible(model, ckpt.get("model_state_dict", {}), strict=strict)
        if not strict and int(load_stats.get("skipped", 0)) > 0:
            print(
                f"[copy] resume checkpoint loaded with shape filtering "
                f"(loaded={int(load_stats.get('loaded', 0))} skipped={int(load_stats.get('skipped', 0))})",
                flush=True,
            )
        if not args.disable_fusion:
            restored_gates = _restore_fusion_gates(model)
            if restored_gates:
                print(f"[copy] restored_fusion_gates={restored_gates}", flush=True)
        if not args.resume_weights_only:
            if ckpt.get("optimizer_state_dict", None) is not None:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict", None) is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict", None) is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            global_step = int(ckpt.get("global_step", 0))
            best_acc = float(ckpt.get("best_acc", -1.0))
            best_record = ckpt.get("best_record", None)
        print(
            f"[copy] resumed checkpoint={args.resume_checkpoint} "
            f"weights_only={bool(args.resume_weights_only)} strict={strict} start_gstep={global_step}",
            flush=True,
        )
        if best_record is not None:
            best_metric_value = _resolve_best_metric(
                best_record,
                name=str(args.best_metric),
                alpha=float(args.best_metric_alpha),
            )

    print(
        "[copy] start "
        f"device={device} batch={args.batch_size} epochs={args.epochs} total_steps={args.total_steps} "
        f"curriculum=(len={args.len8} x{args.len8_epochs} epochs, len={args.len16} x{max(0, args.epochs - args.len8_epochs)} epochs) "
        f"dynamic_lr=warmup+cosine grad_clip={args.grad_clip} target_grad_norm={args.target_grad_norm} "
        f"disable_fusion={bool(args.disable_fusion)}",
        flush=True,
    )
    print(
        f"[copy] phase_lr_mults phase1={float(args.phase1_lr_mult):.3f} phase2={float(args.phase2_lr_mult):.3f}",
        flush=True,
    )
    print(
        f"[copy] phase_transition epoch={int(args.phase_transition_epoch)} step={int(args.phase_transition_step)} "
        f"len16_loss_weight={float(args.len16_loss_weight):.3f} grad_noise_std={float(args.grad_noise_std):.5f} "
        f"curved_hidden_dim={int(getattr(model.cortex.long_term_memory, 'curved_hidden_dim', args.ltm_curved_hidden_dim or args.d_model))}",
        flush=True,
    )
    print(
        f"[copy] fusion_stack={bool(args.enable_full_fusion_stack)} task_decoder={bool(args.enable_task_decoder)} "
        f"decoder_layers={int(args.task_decoder_layers)} full_stack_advanced={bool(full_stack_advanced_enabled)} "
        f"full_stack_qdt={bool(full_stack_qdt_enabled)} full_stack_shared={bool(full_stack_shared_enabled)}",
        flush=True,
    )
    print(
        f"[copy] best_metric={str(args.best_metric)} best_metric_alpha={float(args.best_metric_alpha):.4f}",
        flush=True,
    )
    print(
        f"[copy] adaptive_lr={bool(args.adaptive_lr_enabled)} up={float(args.adaptive_lr_up):.3f} "
        f"down={float(args.adaptive_lr_down):.3f} min={float(args.adaptive_lr_min_mult):.3f} "
        f"max={float(args.adaptive_lr_max_mult):.3f} patience={int(args.adaptive_lr_patience)}",
        flush=True,
    )
    print(
        f"[copy] diagnostics_enabled={bool(args.enable_diagnostics)} "
        f"diagnostics_log_path={str(args.diagnostics_log_path) if str(args.diagnostics_log_path).strip() else 'none'}",
        flush=True,
    )
    print("[copy] metrics logged every 5 steps with live terminal output", flush=True)
    print(f"[copy] metrics_jsonl={args.metrics_jsonl}", flush=True)
    if args.checkpoint_out:
        print(f"[copy] checkpoint_out={args.checkpoint_out}", flush=True)
    if args.checkpoint_best_out:
        print(f"[copy] checkpoint_best_out={args.checkpoint_best_out}", flush=True)

    adaptive_mult = 1.0
    bad_step_streak = 0
    ema_loss = None
    best_running_acc = 0.0
    last_success_step = -10**9

    with open(args.metrics_jsonl, "w", encoding="utf-8") as jf:
        jf.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "run_start",
                    "time": float(time.time()),
                    "args": vars(args),
                }
            )
            + "\n"
        )
        jf.flush()
        if bool(args.use_copy_task_dataloader):
            eval_loader_8 = _build_copy_task_loader(
                args,
                n_samples=max(256, args.eval_samples // 2),
                max_len=args.len8,
                shuffle=False,
                epoch=0,
            )
            eval_loader_16 = _build_copy_task_loader(
                args,
                n_samples=max(256, args.eval_samples // 2),
                max_len=args.len16,
                shuffle=False,
                epoch=0,
            )
        else:
            eval_ds_8 = _build_fixed_copytask_dataset(max(256, args.eval_samples // 2), args.len8, seed=int(args.seed) + 8108)
            eval_ds_16 = _build_fixed_copytask_dataset(max(256, args.eval_samples // 2), args.len16, seed=int(args.seed) + 8116)
            eval_loader_8 = DataLoader(eval_ds_8, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            eval_loader_16 = DataLoader(eval_ds_16, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        for ep in range(1, args.epochs + 1):
            if int(args.topology_relax_after_epoch) > 0 and ep == int(args.topology_relax_after_epoch) + 1:
                _relax_topology_mutation(model.cortex, factor=float(args.topology_relax_factor))
                print(
                    f"[copy] topology_relaxed epoch={ep} factor={float(args.topology_relax_factor):.3f}",
                    flush=True,
                )
            cur_len = args.len8 if ep <= args.len8_epochs else args.len16
            phase_epoch_boundary = int(args.phase_transition_epoch) if int(args.phase_transition_epoch) > 0 else int(args.len8_epochs)
            if bool(args.use_copy_task_dataloader):
                train_loader = _build_copy_task_loader(
                    args,
                    n_samples=args.train_samples_per_epoch,
                    max_len=cur_len,
                    shuffle=True,
                    epoch=ep,
                )
                eval_loader_cur = eval_loader_8 if int(cur_len) == int(args.len8) else eval_loader_16
            else:
                train_ds = CopyTask(n_samples=args.train_samples_per_epoch, max_len=cur_len)
                eval_ds_cur = eval_ds_8 if int(cur_len) == int(args.len8) else eval_ds_16
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                eval_loader_cur = DataLoader(eval_ds_cur, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            train_iter = iter(train_loader)

            model.train()
            run_loss = 0.0
            run_correct = 0
            run_tok = 0
            run_seq_correct = 0
            run_seq_total = 0
            run_entropy = 0.0
            run_grad_scale = 0.0
            run_clip_hits = 0.0
            epoch_steps = 0

            for step_in_epoch in range(1, steps_per_epoch + 1):
                if global_step >= args.total_steps:
                    break
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                src, tgt = _unpack_copy_batch(batch)

                src = src.to(device)
                tgt = tgt.to(device)
                amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_cuda else contextlib.nullcontext()
                with amp_ctx:
                    logits, aux = model(src, return_aux_losses=True)
                    if bool(args.use_copy_task_dataloader):
                        logits_t, tgt_t = align_logits_targets(logits, tgt)
                    else:
                        logits_t, tgt_t = _align_logits_targets(logits, tgt)
                    loss = F.cross_entropy(
                        logits_t.reshape(-1, logits_t.size(-1)),
                        tgt_t.reshape(-1),
                        ignore_index=TOK2IDX["<pad>"],
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
                    if cur_len == int(args.len16):
                        loss = loss * float(args.len16_loss_weight)
                    mastery_loss_val = 0.0
                    if mastery_model is not None:
                        mastery_logits = mastery_model(src, tgt)
                        mastery_loss = mastery_model.mastery_loss(
                            mastery_logits,
                            tgt,
                            ignore_index=TOK2IDX["<pad>"],
                            label_smoothing=float(args.label_smoothing),
                        )
                        mastery_loss_val = float(mastery_loss.detach().item())
                        loss = loss + float(args.copy_mastery_loss_weight) * mastery_loss

                opt.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()

                if float(args.grad_noise_std) > 0.0:
                    std = float(args.grad_noise_std)
                    for p_ in model.parameters():
                        if p_.grad is not None:
                            p_.grad.add_(torch.randn_like(p_.grad) * std)

                clip_params = list(model.parameters())
                if mastery_model is not None:
                    clip_params += list(mastery_model.parameters())
                pre_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1e9)
                pre_norm_v = float(pre_norm)
                grad_scale = 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, float(args.grad_clip))
                clip_hit = 1.0 if pre_norm_v > float(args.grad_clip) + 1e-8 else 0.0

                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                if hasattr(model, "cortex") and getattr(model.cortex, "advanced_broker", None) is not None:
                    with torch.no_grad():
                        model.cortex._tick_advanced_consolidation()
                scheduler.step()
                phase_lr_mult = float(args.phase1_lr_mult)
                if int(args.phase_transition_step) > 0:
                    if global_step >= int(args.phase_transition_step):
                        phase_lr_mult = float(args.phase2_lr_mult)
                elif ep > phase_epoch_boundary:
                    phase_lr_mult = float(args.phase2_lr_mult)
                if bool(args.adaptive_lr_enabled):
                    cur_loss = float(loss.detach().item())
                    if ema_loss is None:
                        ema_loss = cur_loss
                    ema_loss = 0.92 * float(ema_loss) + 0.08 * cur_loss
                    run_acc_now = run_correct / max(1, run_tok)
                    if run_acc_now > best_running_acc:
                        best_running_acc = run_acc_now
                    bad_loss = cur_loss > float(ema_loss) + float(args.adaptive_lr_loss_tolerance)
                    if bad_loss and run_acc_now < max(float(args.adaptive_lr_acc_threshold), best_running_acc - 0.015):
                        bad_step_streak += 1
                    else:
                        bad_step_streak = max(0, bad_step_streak - 1)
                    if bad_step_streak >= int(args.adaptive_lr_patience):
                        adaptive_mult = max(float(args.adaptive_lr_min_mult), adaptive_mult * float(args.adaptive_lr_down))
                        bad_step_streak = 0
                    elif run_acc_now >= float(args.adaptive_lr_acc_threshold) and cur_loss <= float(ema_loss):
                        adaptive_mult = min(float(args.adaptive_lr_max_mult), adaptive_mult * float(args.adaptive_lr_up))
                if phase_lr_mult != 1.0:
                    base_lrs = scheduler.get_last_lr()
                    for g, base_lr in zip(opt.param_groups, base_lrs):
                        g["lr"] = float(base_lr) * float(phase_lr_mult)
                if bool(args.adaptive_lr_enabled) and adaptive_mult != 1.0:
                    for g in opt.param_groups:
                        g["lr"] = float(g["lr"]) * float(adaptive_mult)

                topology_frozen = int(args.topology_freeze_after_epoch) > 0 and ep > int(args.topology_freeze_after_epoch)
                if hasattr(model, "cortex") and hasattr(model.cortex, "topology_step") and not topology_frozen:
                    loss_val = float(loss.detach().item())
                    model.cortex.topology_step(loss_val)
                    if hasattr(model.cortex, "step_topology_schedulers"):
                        model.cortex.step_topology_schedulers(loss_val, telemetry_by_bank=_build_topology_telemetry(model.cortex))
                if hasattr(model, "cortex") and model.cortex.consolidated_lexicon is not None:
                    model.cortex.consolidated_lexicon.renorm_constraints_()

                preds = logits_t.argmax(dim=-1)
                mask = tgt_t != TOK2IDX["<pad>"]
                batch_correct = int((preds == tgt_t).masked_select(mask).sum().item())
                batch_tok = int(mask.sum().item())
                row_ok = ((preds == tgt_t) | (~mask)).all(dim=1)
                valid_rows = mask.any(dim=1)
                batch_seq_correct = int(row_ok.masked_select(valid_rows).sum().item())
                batch_seq_total = int(valid_rows.sum().item())
                probs = torch.softmax(logits_t.detach(), dim=-1)
                tok_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
                tok_entropy_mean = float(tok_entropy.masked_select(mask).mean().item()) if batch_tok > 0 else 0.0
                run_correct += batch_correct
                run_tok += batch_tok
                run_seq_correct += batch_seq_correct
                run_seq_total += batch_seq_total
                run_loss += float(loss.detach().item())
                run_entropy += tok_entropy_mean
                run_grad_scale += grad_scale
                run_clip_hits += clip_hit
                global_step += 1
                epoch_steps += 1
                run_acc_now = run_correct / max(1, run_tok)
                if (
                    run_acc_now >= float(args.success_acc_threshold)
                    and (global_step - last_success_step) >= int(args.success_memory_cooldown_steps)
                ):
                    last_success_step = global_step
                    _append_success_memory(
                        args.success_memory_path,
                        {
                            "time": time.time(),
                            "global_step": int(global_step),
                            "epoch": int(ep),
                            "seq_len": int(cur_len),
                            "acc": float(run_acc_now),
                            "mean_loss": float(run_loss / max(1, epoch_steps)),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "adaptive_lr_mult": float(adaptive_mult),
                            "phase_lr_mult": float(phase_lr_mult),
                            "hparams": vars(args),
                            "methodology": "fusion_stack+task_decoder+adaptive_lr+cms_cps",
                            "named_parameters_sample": [n for n, _ in list(model.named_parameters())[:32]],
                        },
                    )

                if global_step % max(1, int(args.log_every)) == 0:
                    mean_loss = run_loss / max(1, epoch_steps)
                    run_acc = run_correct / max(1, run_tok)
                    run_seq_acc = run_seq_correct / max(1, run_seq_total)
                    run_entropy_mean = run_entropy / max(1, epoch_steps)
                    run_grad_scale_mean = run_grad_scale / max(1, epoch_steps)
                    run_clip_rate = run_clip_hits / max(1, epoch_steps)
                    lr_now = float(opt.param_groups[0]["lr"])
                    lr_base_now = float(scheduler.get_last_lr()[0])
                    topo_fit = 0.0
                    extra_metrics = {}
                    if hasattr(model, "cortex"):
                        topo_fit = float(getattr(model.cortex.topology, "fitness_ema", 0.0) or 0.0)
                        extra_metrics = _flatten_metrics(model.cortex.get_metrics())
                    aux_metrics = {}
                    if isinstance(aux, dict):
                        for k, v in aux.items():
                            if isinstance(v, (float, int)):
                                aux_metrics[f"aux_{k}"] = float(v)
                            elif isinstance(v, torch.Tensor) and v.numel() > 0:
                                aux_metrics[f"aux_{k}"] = float(v.detach().float().mean().item())

                    event = {
                        "schema_version": 1,
                        "kind": "train_step",
                        "epoch": ep,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "seq_len": cur_len,
                        "loss": float(loss.detach().item()),
                        "mean_loss": float(mean_loss),
                        "acc": float(run_acc),
                        "seq_acc": float(run_seq_acc),
                        "token_entropy": float(run_entropy_mean),
                        "recall_loss": float(recall_loss_val),
                        "cms_loss": float(cms_loss_val),
                        "pre_grad_norm": float(pre_norm_v),
                        "grad_norm": float(grad_norm),
                        "grad_scale": float(grad_scale),
                        "grad_scale_mean": float(run_grad_scale_mean),
                        "clip_hit": float(clip_hit),
                        "clip_hit_rate": float(run_clip_rate),
                        "lr": lr_now,
                        "lr_base": lr_base_now,
                        "phase_lr_mult": float(phase_lr_mult),
                        "phase_id": 1.0 if phase_lr_mult == float(args.phase1_lr_mult) else 2.0,
                        "adaptive_lr_mult": float(adaptive_mult),
                        "adaptive_bad_streak": float(bad_step_streak),
                        "ema_loss": float(ema_loss if ema_loss is not None else 0.0),
                        "topology_fitness_ema": float(topo_fit),
                        "topology_frozen": float(topology_frozen),
                    }
                    event.update(extra_metrics)
                    event.update(aux_metrics)
                    jf.write(json.dumps(event) + "\n")
                    jf.flush()
                    print(
                        f"[train ep={ep:02d} gstep={global_step:04d}/{args.total_steps:04d} len={cur_len:02d}] "
                        f"loss={event['loss']:.4f} mean={event['mean_loss']:.4f} acc={event['acc']:.4f} seq_acc={event['seq_acc']:.4f} "
                        f"recall={event['recall_loss']:.4f} cms={event['cms_loss']:.4f} "
                        f"pre_grad={event['pre_grad_norm']:.4f} grad={event['grad_norm']:.4f} "
                        f"lr={event['lr']:.6f} topo_fit={event['topology_fitness_ema']:.4f} clip_rate={event['clip_hit_rate']:.3f}",
                        flush=True,
                    )
                    print(f"[train_full_metrics] {json.dumps(event, sort_keys=True)}", flush=True)

            train_mean = run_loss / max(1, epoch_steps)
            train_acc = run_correct / max(1, run_tok)
            train_seq_acc = run_seq_correct / max(1, run_seq_total)
            train_entropy_mean = run_entropy / max(1, epoch_steps)
            train_grad_scale_mean = run_grad_scale / max(1, epoch_steps)
            train_clip_hit_rate = run_clip_hits / max(1, epoch_steps)
            use_align = bool(args.use_copy_task_dataloader)
            val_loss_cur, val_acc_cur, val_seq_acc_cur = _evaluate(
                model, eval_loader_cur, device, use_aligned_targets=use_align
            )
            val_loss_8, val_acc_8, val_seq_acc_8 = _evaluate(
                model, eval_loader_8, device, use_aligned_targets=use_align
            )
            val_loss_16, val_acc_16, val_seq_acc_16 = _evaluate(
                model, eval_loader_16, device, use_aligned_targets=use_align
            )
            epoch_metrics = {}
            if hasattr(model, "cortex"):
                epoch_metrics = _flatten_metrics(model.cortex.get_metrics())
            summary = {
                "schema_version": 1,
                "kind": "epoch_end",
                "epoch": ep,
                "global_step": global_step,
                "train_mean_loss": float(train_mean),
                "train_acc": float(train_acc),
                "train_seq_acc": float(train_seq_acc),
                "train_token_entropy_mean": float(train_entropy_mean),
                "train_grad_scale_mean": float(train_grad_scale_mean),
                "train_clip_hit_rate": float(train_clip_hit_rate),
                "val_loss_curriculum_len": float(val_loss_cur),
                "val_acc_curriculum_len": float(val_acc_cur),
                "val_seq_acc_curriculum_len": float(val_seq_acc_cur),
                "val_loss_len8": float(val_loss_8),
                "val_acc_len8": float(val_acc_8),
                "val_seq_acc_len8": float(val_seq_acc_8),
                "val_loss_len16": float(val_loss_16),
                "val_acc_len16": float(val_acc_16),
                "val_seq_acc_len16": float(val_seq_acc_16),
            }
            summary["best_metric_value"] = _resolve_best_metric(
                summary,
                name=str(args.best_metric),
                alpha=float(args.best_metric_alpha),
            )
            summary.update(epoch_metrics)
            jf.write(json.dumps(summary) + "\n")
            jf.flush()
            print(
                f"[epoch {ep:02d}] train_mean={train_mean:.4f} train_acc={train_acc:.4f} "
                f"val(curr_len={cur_len}) loss={val_loss_cur:.4f} acc={val_acc_cur:.4f} "
                f"val(len8) loss={val_loss_8:.4f} acc={val_acc_8:.4f} "
                f"val(len16) loss={val_loss_16:.4f} acc={val_acc_16:.4f}",
                flush=True,
            )
            if val_acc_cur > best_acc:
                best_acc = val_acc_cur
            if float(summary["best_metric_value"]) > float(best_metric_value):
                best_metric_value = float(summary["best_metric_value"])
                best_record = summary
                _save_checkpoint(
                    args.checkpoint_best_out,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    epoch=ep,
                    global_step=global_step,
                    best_acc=best_acc,
                    best_record=best_record,
                )
            if global_step >= args.total_steps:
                break

    if hasattr(model, "flush_cms_logger"):
        model.flush_cms_logger()
    if hasattr(model, "cortex") and hasattr(model.cortex, "flush_diagnostics"):
        flushed = model.cortex.flush_diagnostics()
        if flushed is not None:
            print(f"[copy] diagnostics_flushed={int(flushed)}", flush=True)
    _save_checkpoint(
        args.checkpoint_out,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        epoch=args.epochs,
        global_step=global_step,
        best_acc=best_acc,
        best_record=best_record,
    )
    print(f"[copy] complete global_steps={global_step} best_curriculum_acc={best_acc:.4f}", flush=True)
    if best_record is not None:
        print(
            f"[copy] best epoch={best_record['epoch']} "
            f"train_mean_loss={best_record['train_mean_loss']:.4f} "
            f"train_acc={best_record['train_acc']:.4f} "
            f"val_loss_curriculum_len={best_record['val_loss_curriculum_len']:.4f} "
            f"val_acc_curriculum_len={best_record['val_acc_curriculum_len']:.4f}",
            flush=True,
        )
    print(f"[copy] full metrics captured at {args.metrics_jsonl}", flush=True)
    if best_record is not None:
        accept_failures = []
        if float(args.accept_min_val_acc) >= 0.0 and float(best_record.get("val_acc_curriculum_len", 0.0)) < float(args.accept_min_val_acc):
            accept_failures.append(
                f"val_acc_curriculum_len={float(best_record.get('val_acc_curriculum_len', 0.0)):.6f} < {float(args.accept_min_val_acc):.6f}"
            )
        if float(args.accept_max_val_loss) >= 0.0 and float(best_record.get("val_loss_curriculum_len", 1e9)) > float(args.accept_max_val_loss):
            accept_failures.append(
                f"val_loss_curriculum_len={float(best_record.get('val_loss_curriculum_len', 1e9)):.6f} > {float(args.accept_max_val_loss):.6f}"
            )
        if float(args.accept_min_diag_events) >= 0.0 and float(best_record.get("diag_events_buffered", 0.0)) < float(args.accept_min_diag_events):
            accept_failures.append(
                f"diag_events_buffered={float(best_record.get('diag_events_buffered', 0.0)):.1f} < {float(args.accept_min_diag_events):.1f}"
            )
        if accept_failures:
            print("[copy][acceptance] FAILED", flush=True)
            for msg in accept_failures:
                print(f"[copy][acceptance] {msg}", flush=True)
            raise SystemExit(2)
        print("[copy][acceptance] PASSED", flush=True)


if __name__ == "__main__":
    main()
