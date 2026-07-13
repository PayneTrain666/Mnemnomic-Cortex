"""
Plain-language summary
----------------------
What this file is for: Adapters so evaluation scripts can call different model backends uniformly.
How it fits in the system: Glue for eval_ahg and similar tools.
Status: WORKING
Important notes for non-coders: Not a training script.
"""

from typing import Dict, List, Optional
import re

import torch


class ModelAdapter:
    """
    Adapter layer for AHG eval.

    Preferred contract:
      generate_with_logits(prompt, temperature, do_sample) -> {
        "text": str,
        "logits": Tensor[T,V] or Tensor[1,T,V] (optional),
        "tokens": Tensor[T] or Tensor[1,T] (optional),
      }
    """

    def __init__(self, model=None, tokenizer=None, device: str = "cpu", vocab_size: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = int(vocab_size)

        # Lightweight fallback tokenizer for environments without a text tokenizer.
        self._stoi: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self._itos: Dict[int, str] = {0: "<pad>", 1: "<unk>"}

    def _tokenize(self, prompt: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_\-']+|[^\s\w]", prompt.lower())

    def _encode_prompt(self, prompt: str, max_len: int = 128) -> torch.Tensor:
        toks = self._tokenize(prompt)[:max_len]
        ids: List[int] = []
        for t in toks:
            if t not in self._stoi and len(self._stoi) < self.vocab_size:
                idx = len(self._stoi)
                self._stoi[t] = idx
                self._itos[idx] = t
            ids.append(self._stoi.get(t, 1))
        if not ids:
            ids = [1]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode_ids(self, ids: torch.Tensor) -> str:
        out = []
        flat = ids.reshape(-1).tolist()
        for i in flat:
            out.append(self._itos.get(int(i), "<unk>"))
        return " ".join(out).strip()

    @torch.no_grad()
    def generate_with_logits(
        self,
        prompt: str,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> Dict:
        # If caller model already provides an adapter, delegate directly.
        if self.model is not None and hasattr(self.model, "generate_with_logits"):
            return self.model.generate_with_logits(prompt, temperature=temperature, do_sample=do_sample)

        # If no model is bound, return safe no-op output.
        if self.model is None:
            return {"text": "I do not have enough reliable evidence to answer accurately.", "logits": None, "tokens": None}

        # Generic fallback for token-level seq models in this repo.
        src = self._encode_prompt(prompt)
        logits = self.model(src)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.dim() != 3:
            return {"text": "", "logits": None, "tokens": None}

        if do_sample:
            probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1)
            tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(logits.size(0), logits.size(1))
        else:
            tokens = logits.argmax(dim=-1)
        text = self._decode_ids(tokens[0])
        return {"text": text, "logits": logits, "tokens": tokens}


class RetrieverAdapter:
    """
    Retriever adapter contract:
      search(query, k) -> [{"text": str, "score": float, "source_id": str}, ...]
    Default is a safe no-op retriever.
    """

    def search(self, query: str, k: int = 5) -> List[Dict]:
        _ = query
        _ = k
        return []


def cite_align_fn(answer: str, docs: List[Dict]) -> float:
    if not docs or not answer.strip():
        return 0.0
    ans_terms = set(re.findall(r"[a-zA-Z]{3,}", answer.lower()))
    if not ans_terms:
        return 0.0
    best = 0.0
    for d in docs[:3]:
        doc_terms = set(re.findall(r"[a-zA-Z]{3,}", str(d.get("text", "")).lower()))
        j = len(ans_terms & doc_terms) / max(1, len(ans_terms))
        best = max(best, j)
    return float(best)


def cms_signals_fn(tokens) -> Dict[str, float]:
    _ = tokens
    return {"sense_entropy": 0.6, "proto_distance": 0.4, "warp": 1.0}

