import argparse
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
    # Recommended current policy for this stack.
    if hasattr(model.cortex, "topology") and hasattr(model.cortex.topology, "register_babi_qhm_v2_policy"):
        model.cortex.topology.register_babi_qhm_v2_policy()
        model.cortex.apply_topology_policy("babi_qhm_v2")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[babi] training epochs={args.epochs} batch={args.batch_size} d_model={args.d_model} "
        f"lr={args.lr} wd={args.weight_decay} device={device}"
    )
    best_acc = 0.0
    best = None
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for src, y in train_loader:
            src = src.to(device)
            y = y.to(device)
            logits, aux = model(src, return_aux_losses=True)
            last_logits = logits[:, -1, :]
            loss = F.cross_entropy(last_logits, y)
            if isinstance(aux.get("recall_loss", None), torch.Tensor):
                loss = loss + model.recall_loss_weight * aux["recall_loss"]
            if isinstance(aux.get("cms_loss", None), torch.Tensor):
                loss = loss + aux["cms_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if hasattr(model, "topology_step"):
                model.topology_step(float(loss.detach().cpu()))
            if model.cortex.consolidated_lexicon is not None:
                model.cortex.consolidated_lexicon.renorm_constraints_()
            total += loss.item() * src.size(0)
            seen += src.size(0)

        train_loss = total / max(1, seen)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(
            f"[epoch {ep:02d}] train_loss={train_loss:.4f} "
            f"test_loss={val_loss:.4f} test_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best = (ep, train_loss, val_loss, val_acc)

    if best is not None:
        ep, tr, vl, acc = best
        print(
            f"[babi] best epoch={ep} train_loss={tr:.4f} "
            f"test_loss={vl:.4f} test_acc={acc:.4f}"
        )


if __name__ == "__main__":
    main()
