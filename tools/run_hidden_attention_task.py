"""
Plain-language summary
----------------------
What this file is for: Trains or evaluates the hidden-attention stress task.
How it fits in the system: Exercises the hidden-attention orchestrator under copy-like workloads.
Status: WORKING
Important notes for non-coders: May still call a defensive QH move helper for older paths.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_DIR = os.path.abspath(os.path.dirname(__file__))
if TOOL_DIR in sys.path:
    sys.path.remove(TOOL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.models import CortexSeqModel
from benchmark.tasks import TOK2IDX, VOCAB_SIZE
from data.copy_task_dataloader import align_logits_targets
from data.hidden_attention_task_loader import (
    HiddenAttentionTaskLoaderConfig,
    build_hidden_attention_task_loaders,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    return src, tgt


def _move_qh_codebooks_to_device(model, device: torch.device) -> int:
    """
    Some QH codebook tensors live in plain dicts rather than module buffers.
    Force-move them to avoid CPU/CUDA mismatch during holographic ops.
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
            if isinstance(qh_slot_bank, torch.nn.Module):
                qh_slot_bank.to(device)
            _move_codebook(getattr(qh_slot_bank, "codebook", None))
        qh_banks = getattr(mod, "qh_banks", None)
        if isinstance(qh_banks, (torch.nn.ModuleDict, dict)):
            for bank in qh_banks.values():
                if isinstance(bank, torch.nn.Module):
                    bank.to(device)
                _move_codebook(getattr(bank, "codebook", None))

    return moved


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct_tok = 0
    total_seq = 0
    correct_seq = 0
    for batch in loader:
        src, tgt = _batch_to_device(batch, device)
        logits = model(src)
        logits, tgt = align_logits_targets(logits, tgt)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=TOK2IDX["<pad>"],
        )
        preds = logits.argmax(dim=-1)
        mask = tgt != TOK2IDX["<pad>"]
        total_loss += float(loss.item())
        correct_tok += int((preds == tgt).masked_select(mask).sum().item())
        total_tok += int(mask.sum().item())
        valid_rows = mask.any(dim=1)
        row_ok = ((preds == tgt) | (~mask)).all(dim=1)
        correct_seq += int(row_ok.masked_select(valid_rows).sum().item())
        total_seq += int(valid_rows.sum().item())
    return {
        "loss": total_loss / max(1, len(loader)),
        "token_acc": correct_tok / max(1, total_tok),
        "seq_acc": correct_seq / max(1, total_seq),
    }


def main():
    p = argparse.ArgumentParser(description="Hidden-attention stress task runner")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--train_samples", type=int, default=1536)
    p.add_argument("--val_samples", type=int, default=384)
    p.add_argument("--max_len", type=int, default=24)
    p.add_argument("--report_out", default="reports/hidden_attention_task_report.json")
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
    args = p.parse_args()

    seed_all(int(args.seed))
    device = torch.device(args.device)
    cfg = HiddenAttentionTaskLoaderConfig(
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
        batch_size=int(args.batch_size),
        max_len=int(args.max_len),
        encoder_d_model=int(args.d_model),
    )
    train_loader, val_loader, meta = build_hidden_attention_task_loaders(cfg)

    model = CortexSeqModel(
        vocab_size=VOCAB_SIZE,
        d_model=int(args.d_model),
        ltm_curved_hidden_mult=1.5,
        task_decoder_enabled=True,
        task_decoder_layers=2,
        task_decoder_heads=8,
        ltm_enable_spatial_ltm=False,
        ltm_auto_wire_spatial=False,
        enable_global_hidden_attention=True,
        ltm_enable_global_hidden_attention=True,
        working_memory_fabric=str(args.working_memory_fabric),
        qdt_hardware_profile=str(args.qdt_hardware_profile),
        qdt_qspin_guarded_shadow=True,
        qdt_qspin_live_activation=False,
        qdt_qspin_live_kill_switch_enabled=True,
    ).to(device)
    fabric = model.cortex.describe_working_memory_fabric()
    print(
        f"[hidden-task] working_memory={fabric['fabric']} "
        f"class={fabric['working_memory_class']} qspin_live=false",
        flush=True,
    )
    moved = _move_qh_codebooks_to_device(model, device)
    if moved > 0:
        print(f"[hidden-task] moved_qh_codebook_tensors={moved} to device={device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model.train()
    iter_train = iter(train_loader)
    train_log = []
    start = time.time()

    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)
        src, tgt = _batch_to_device(batch, device)
        logits, aux = model(src, return_aux_losses=True)
        logits, tgt = align_logits_targets(logits, tgt)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=TOK2IDX["<pad>"],
            label_smoothing=0.02,
        )
        if isinstance(aux, dict) and isinstance(aux.get("recall_loss"), torch.Tensor):
            loss = loss + 0.08 * aux["recall_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = tgt != TOK2IDX["<pad>"]
                tok_acc = float((preds == tgt).masked_select(mask).float().mean().item()) if bool(mask.any()) else 0.0
            train_log.append({"step": int(step), "loss": float(loss.item()), "token_acc": tok_acc})
            print(
                f"[hidden-task] step={step:04d}/{int(args.steps):04d} "
                f"loss={float(loss.item()):.4f} token_acc={tok_acc:.4f}",
                flush=True,
            )

    metrics = evaluate(model, val_loader, device=device)
    elapsed = time.time() - start
    hidden_stats = {}
    if hasattr(model, "cortex"):
        hidden_stats = {
            "cortex_global_hidden": dict(getattr(model.cortex, "last_global_hidden_attention_stats", {})),
            "ltm_global_hidden": dict(getattr(model.cortex.long_term_memory, "last_global_hidden_attention_stats", {})),
            "cortex_secondary_hidden": dict(getattr(model.cortex, "last_secondary_hidden_stack_stats", {})),
            "ltm_hidden_stack": dict(getattr(model.cortex.long_term_memory, "last_hidden_stack_stats", {})),
        }

    report = {
        "task_meta": meta,
        "run": {
            "device": str(device),
            "steps": int(args.steps),
            "elapsed_sec": float(elapsed),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "d_model": int(args.d_model),
        },
        "train_log": train_log,
        "val_metrics": metrics,
        "hidden_attention_stats": hidden_stats,
    }
    out_path = os.path.abspath(args.report_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[hidden-task] done loss={metrics['loss']:.4f} "
        f"token_acc={metrics['token_acc']:.4f} seq_acc={metrics['seq_acc']:.4f}",
        flush=True,
    )
    print(f"[hidden-task] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
