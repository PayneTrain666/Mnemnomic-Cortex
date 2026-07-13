"""
Plain-language summary
----------------------
What this file is for: Operations for CMS logging, EMA consolidation, and shard dump/load.
How it fits in the system: Practical tools that move consolidated knowledge in and out.
Status: ACTIVE when CMS path on
Important notes for non-coders: Includes safety clamps for geometry.
"""

import os
import time
from typing import Dict, List, Optional

import torch


def _unit_complex_real(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q2 = z.size(-1)
    if q2 % 2 != 0:
        raise ValueError(f"Expected even last dim for packed complex, got {q2}")
    zc = torch.view_as_complex(z.view(*z.shape[:-1], q2 // 2, 2).contiguous())
    n = torch.linalg.norm(zc, dim=-1, keepdim=True).clamp_min(eps)
    zc = zc / n
    return torch.view_as_real(zc).reshape_as(z)


@torch.no_grad()
def cms_safety_clamp(lexicon) -> None:
    """
    Keep manifold factors numerically well-formed after updates.
    """
    if lexicon is None:
        return
    if hasattr(lexicon, "renorm_constraints_"):
        lexicon.renorm_constraints_()


class CMSRecordLogger:
    """
    Lightweight buffered logger for consolidated-memory records.
    """

    def __init__(self, out_dir: str, max_buffer: int = 100000, sample_rate: float = 0.1):
        self.out_dir = out_dir
        self.max_buffer = int(max_buffer)
        self.sample_rate = float(sample_rate)
        self._buffer: List[Dict] = []
        self._file_idx = 0
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def log_from_aux(self, token_ids: torch.Tensor, aux: Dict):
        if token_ids.numel() == 0:
            return
        cue_h = aux.get("cue_h")
        cue_p = aux.get("cue_p")
        cue_e = aux.get("cue_e")
        sense_w = aux.get("weights")
        if cue_h is None or cue_p is None or cue_e is None or sense_w is None:
            return

        n = token_ids.size(0)
        keep = torch.rand(n, device=token_ids.device) < self.sample_rate
        if keep.sum().item() == 0:
            return

        tids = token_ids[keep].detach().cpu().tolist()
        ws = sense_w[keep].detach().cpu().tolist()
        hs = cue_h[keep].detach().cpu().tolist()
        ps = _unit_complex_real(cue_p[keep].detach().cpu()).tolist()
        es = cue_e[keep].detach().cpu().tolist()

        for i in range(len(tids)):
            self._buffer.append(
                {
                    "token_id": int(tids[i]),
                    "sense_w": ws[i],
                    "cue_h": hs[i],
                    "cue_p": ps[i],
                    "cue_e": es[i],
                }
            )
        if len(self._buffer) >= self.max_buffer:
            self.flush()

    def flush(self) -> Optional[str]:
        if not self._buffer:
            return None
        ts = int(time.time())
        path = os.path.join(self.out_dir, f"cms_records_{ts}_{self._file_idx}.pt")
        torch.save(self._buffer, path)
        self._buffer = []
        self._file_idx += 1
        return path


@torch.no_grad()
def run_cms_consolidation_ema(lexicon, records: List[Dict], ema: float = 0.9):
    """
    Nightly-style EMA consolidation from logged cue records.
    Updates mu_h/mu_p/mu_e per token+sense bucket.
    """
    buckets: Dict[tuple, Dict[str, List[torch.Tensor]]] = {}
    device = lexicon.mu_h.device
    for rec in records:
        tid = int(rec["token_id"])
        sw = torch.tensor(rec["sense_w"], device=device, dtype=torch.float32)
        s = int(torch.argmax(sw).item())
        key = (tid, s)
        b = buckets.setdefault(key, {"h": [], "p": [], "e": []})
        b["h"].append(torch.tensor(rec["cue_h"], device=device, dtype=torch.float32))
        b["p"].append(torch.tensor(rec["cue_p"], device=device, dtype=torch.float32))
        b["e"].append(torch.tensor(rec["cue_e"], device=device, dtype=torch.float32))

    for (tid, s), b in buckets.items():
        h = torch.stack(b["h"]).mean(dim=0)
        p = _unit_complex_real(torch.stack(b["p"]).mean(dim=0, keepdim=True)).squeeze(0)
        e = torch.stack(b["e"]).mean(dim=0)
        lexicon.mu_h[tid, s].mul_(ema).add_((1.0 - ema) * h)
        lexicon.mu_p[tid, s].mul_(ema).add_((1.0 - ema) * p)
        lexicon.mu_e[tid, s].mul_(ema).add_((1.0 - ema) * e)
    cms_safety_clamp(lexicon)


class ConsolidationEMAJob:
    """
    Small helper wrapper for recurring consolidation jobs.
    """

    def __init__(self, lexicon, ema: float = 0.9):
        self.lexicon = lexicon
        self.ema = float(ema)

    @torch.no_grad()
    def step(self, records: List[Dict]) -> None:
        run_cms_consolidation_ema(self.lexicon, records, ema=self.ema)
        cms_safety_clamp(self.lexicon)


def dump_cpg_shards(
    lexicon,
    out_dir: str,
    shard_size: int = 4096,
    quantize: bool = False,
) -> List[str]:
    """
    Export lexicon tensors into shard files.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = ["mu_h", "mu_p", "mu_e", "pron", "morph", "char", "sense_logit", "w"]
    keys = []
    for start in range(0, lexicon.vocab_size, shard_size):
        end = min(start + shard_size, lexicon.vocab_size)
        shard_key = f"{start // shard_size:05d}"
        payload = {
            "ids": torch.arange(start, end, dtype=torch.long),
            "meta": {
                "start": start,
                "end": end,
                "k": lexicon.k,
                "dh": lexicon.dh,
                "q": lexicon.q,
                "de": lexicon.de,
                "dp": lexicon.dp,
                "dc": lexicon.dc,
                "quantized": bool(quantize),
            },
            "tensors": {},
        }
        for name in names:
            t = getattr(lexicon, name).detach().cpu()
            if name != "w":
                t = t[start:end]
            if quantize and t.dim() >= 2:
                mx = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = mx / 127.0
                q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)
                payload["tensors"][name] = {"q": q, "scale": scale, "quantized": True}
            else:
                payload["tensors"][name] = {"value": t, "quantized": False}
        torch.save(payload, os.path.join(out_dir, f"{shard_key}.pt"))
        keys.append(shard_key)
    torch.save(
        {
            "vocab_size": lexicon.vocab_size,
            "shard_size": shard_size,
            "keys": keys,
        },
        os.path.join(out_dir, "manifest.pt"),
    )
    return keys


@torch.no_grad()
def load_cpg_shards_into_lexicon(lexicon, shard_dir: str, keys: Optional[List[str]] = None):
    """
    Load one or more shard files back into an existing lexicon.
    """
    if keys is None:
        manifest = torch.load(os.path.join(shard_dir, "manifest.pt"), map_location="cpu")
        keys = manifest["keys"]
    for key in keys:
        payload = torch.load(os.path.join(shard_dir, f"{key}.pt"), map_location="cpu")
        ids = payload["ids"].long()
        for name, obj in payload["tensors"].items():
            if obj.get("quantized", False):
                value = obj["q"].float() * obj["scale"].float()
            else:
                value = obj["value"]
            target = getattr(lexicon, name)
            if name == "w":
                target.copy_(value.to(target.device))
            else:
                target[ids] = value.to(target.device)
    cms_safety_clamp(lexicon)
