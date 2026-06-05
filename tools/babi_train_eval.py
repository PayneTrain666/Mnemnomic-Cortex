import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple
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
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from benchmark.models import CortexSeqModel
from mnemonic_cortex.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_warmup_cosine_scheduler,
)
from mnemonic_cortex.parameter_audit import ParameterAuditLogger


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    with torch.no_grad():
        for src, y in loader:
            src = src.to(device)
            y = y.to(device)
            logits = model(src)  # [B,T,V]
            last_logits = logits[:, -1, :]
            loss = F.cross_entropy(last_logits, y)
            total_loss += loss.item() * src.size(0)
            pred = last_logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            n += src.size(0)
    return total_loss / max(1, n), correct / max(1, n)


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
    print("    test_loss = validation/test loss on held-out split; lower is better", flush=True)
    print("    test_acc  = validation/test answer accuracy; primary bAbI score (higher is better)", flush=True)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="en-qa1", help="bAbI config, e.g. en-qa1 or en-10k-qa1")
    p.add_argument("--epochs", type=int, default=12)
    # Modest default scale-up from 64 -> 96 while keeping CPU-safe batch.
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20, help="Print live training stats every N steps")
    p.add_argument("--enable_diagnostics", action="store_true", help="Enable cortex diagnostics")
    p.add_argument("--diag_log_path", default=None, help="Optional diagnostics JSONL path")
    p.add_argument("--cps_stage", type=int, default=1, help="CPS curriculum stage (0..3)")
    p.add_argument("--cps_curriculum", action="store_true", help="Auto-progress CPS stage across epochs")
    p.add_argument("--warmup_ratio", type=float, default=0.12, help="Warmup fraction for dynamic LR")
    p.add_argument("--min_lr_ratio", type=float, default=0.08, help="Min LR as fraction of base LR")
    p.add_argument("--label_smoothing", type=float, default=0.03, help="Label smoothing for CE loss")
    p.add_argument("--target_grad_norm", type=float, default=0.6, help="Normalize gradients to this norm")
    p.add_argument("--grad_clip", type=float, default=1.2, help="Safety clip after normalization")
    p.add_argument("--param_audit", action="store_true", help="Enable detailed parameter creation/content audit logging")
    p.add_argument("--param_audit_path", default="logs/babi_param_audit.jsonl", help="JSONL path for parameter audit logs")
    p.add_argument("--param_audit_max_dynamic", type=int, default=5000, help="Max dynamic creation events to record")
    args = p.parse_args()

    seed_all(args.seed)
    device = args.device
    print(f"[babi] loading facebook/babi_qa::{args.config}")
    ds = load_dataset("facebook/babi_qa", args.config, trust_remote_code=True)
    train_ex = build_examples(ds["train"])
    test_ex = build_examples(ds["test"])
    vocab = build_vocab(train_ex, test_ex)
    print(f"[babi] examples train={len(train_ex)} test={len(test_ex)} vocab={len(vocab)}")

    train_ds = BabiQADataset(train_ex, vocab, args.max_len)
    test_ds = BabiQADataset(test_ex, vocab, args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id=vocab["<pad>"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, pad_id=vocab["<pad>"]),
    )

    model = CortexSeqModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        cms_enabled=True,
        cms_senses=3,
        cms_aux_weight=0.01,
    ).to(device)
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
            flush_every=200,
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
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = _build_warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    print(
        f"[babi] training epochs={args.epochs} batch={args.batch_size} d_model={args.d_model} "
        f"lr={args.lr} wd={args.weight_decay} device={device} "
        f"warmup={args.warmup_ratio} min_lr_ratio={args.min_lr_ratio} "
        f"label_smoothing={args.label_smoothing}"
    )
    _print_metric_legend()
    best_acc = 0.0
    best = None
    global_step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        running_correct = 0
        running_seen = 0
        if args.cps_curriculum and hasattr(model.cortex, "set_cps_curriculum_stage"):
            stage = min(3, int(((ep - 1) * 4) / max(1, args.epochs)))
            model.cortex.set_cps_curriculum_stage(stage)
        for step, (src, y) in enumerate(train_loader, start=1):
            src = src.to(device)
            y = y.to(device)
            logits, aux = model(src, return_aux_losses=True)
            last_logits = logits[:, -1, :]
            loss = F.cross_entropy(last_logits, y, label_smoothing=float(args.label_smoothing))
            recall_loss_val = 0.0
            cms_loss_val = 0.0
            if isinstance(aux.get("recall_loss", None), torch.Tensor):
                recall_loss_val = float(aux["recall_loss"].detach().item())
                loss = loss + model.recall_loss_weight * aux["recall_loss"]
            if isinstance(aux.get("cms_loss", None), torch.Tensor):
                cms_loss_val = float(aux["cms_loss"].detach().item())
                loss = loss + aux["cms_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            pre_norm_v = float(pre_norm)
            if pre_norm_v > 0.0 and math.isfinite(pre_norm_v):
                scale = float(args.target_grad_norm) / (pre_norm_v + 1e-8)
                for p_ in model.parameters():
                    if p_.grad is not None:
                        p_.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            scheduler.step()
            global_step += 1
            if hasattr(model, "cortex") and hasattr(model.cortex, "topology") and hasattr(model.cortex, "topology_step"):
                loss_val = float(loss.detach().cpu())
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
            total += loss.item() * src.size(0)
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
                    f"loss={float(loss.detach().item()):.4f} mean={mean_loss:.4f} "
                    f"acc={run_acc:.4f} recall={recall_loss_val:.4f} cms={cms_loss_val:.4f} "
                    f"grad={float(grad_norm):.4f} pre_grad={pre_norm_v:.4f} lr={lr_now:.6f} "
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
        val_loss, val_acc = evaluate(model, test_loader, device)
        m_epoch = model.cortex.get_metrics() if hasattr(model, "cortex") else {}
        router_epoch_line = _format_router_snapshot(m_epoch)
        ltm_router_epoch_line = _format_ltm_router_snapshot(m_epoch)
        print(
            f"[epoch {ep:02d}] train_loss={train_loss:.4f} "
            f"test_loss={val_loss:.4f} test_acc={val_acc:.4f}"
        )
        if router_epoch_line:
            print(f"[router epoch {ep:02d}] {router_epoch_line}", flush=True)
        if ltm_router_epoch_line:
            print(f"[ltm_router epoch {ep:02d}] {ltm_router_epoch_line}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best = (ep, train_loss, val_loss, val_acc)

    if best is not None:
        ep, tr, vl, acc = best
        print(
            f"[babi] best epoch={ep} train_loss={tr:.4f} "
            f"test_loss={vl:.4f} test_acc={acc:.4f}"
        )
    if hasattr(model, "flush_cms_logger"):
        model.flush_cms_logger()
    if hasattr(model.cortex, "flush_diagnostics"):
        model.cortex.flush_diagnostics()


if __name__ == "__main__":
    main()
