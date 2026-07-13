"""
Plain-language summary
----------------------
What this file is for: Helper math and thresholds used by anti-hallucination checks.
How it fits in the system: Supports AHG decisions with entropy/margin style signals.
Status: OPT-IN / WORKING
Important notes for non-coders: Companion to ahg.py.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import re

import torch
import torch.nn.functional as F


@dataclass
class AHGThresholds:
    # Uncertainty
    max_entropy: float = 3.0
    min_margin: float = 0.5
    min_consistency: float = 0.55
    # Grounding
    min_retrieval_score: float = 0.25
    min_coverage: float = 0.35
    min_cite_align: float = 0.2
    # CMS novelty/ambiguity
    max_cms_sense_entropy: float = 1.2
    max_cms_proto_dist: float = 0.85
    # Policy behavior
    require_citations_when_retrieved: bool = True
    allow_refuse: bool = False


@dataclass
class AHGDecision:
    action: str  # ALLOW | RETRIEVE | ASK | REFUSE
    reason: str
    signals: Dict[str, float]
    docs: Optional[List[dict]] = None


def _entropy_from_logits(logits: torch.Tensor) -> float:
    if logits.dim() == 3:
        logits = logits.mean(dim=0)
    probs = F.softmax(logits, dim=-1)
    ent = (-probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean().item()
    return float(ent)


def _margin_from_logits(logits: torch.Tensor) -> float:
    if logits.dim() == 3:
        logits = logits.mean(dim=0)
    top2 = torch.topk(logits, k=2, dim=-1).values
    return float((top2[..., 0] - top2[..., 1]).mean().item())


def _self_consistency(
    generate_fn: Callable,
    prompt: str,
    n: int = 5,
    temperature: float = 0.9,
) -> Tuple[float, List[str]]:
    outs = []
    for _ in range(n):
        y = generate_fn(prompt, temperature=temperature, do_sample=True)
        outs.append(str(y.get("out_text", "")).strip())
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    buckets: Dict[str, int] = {}
    for o in outs:
        k = norm(o)
        buckets[k] = buckets.get(k, 0) + 1
    agree = (max(buckets.values()) / max(1, n)) if buckets else 0.0
    return float(agree), outs


def _numeric_temporal_sanity(answer: str) -> float:
    # Basic sanity heuristic; can be extended.
    years = [int(y) for y in re.findall(r"\b(\d{4})\b", answer)]
    bad = 0
    for y in years:
        if y < 1800 or y > 2100:
            bad += 1
    if not years:
        return 1.0
    return max(0.0, 1.0 - (bad / max(1, len(years))))


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HallucinationGuard:
    """
    Guardrail wrapper that detects likely hallucinations and changes behavior.

    Required generate_fn contract:
        generate_fn(prompt, **kw) -> {"out_text": str, "logits": Tensor?, "tokens": Tensor?}
    Optional retriever_fn contract:
        retriever_fn(query, k) -> [{"text": str, "score": float, "source_id": str}, ...]
    Optional cite_align_fn contract:
        cite_align_fn(answer, docs) -> float in [0,1]
    Optional cms_signals_fn contract:
        cms_signals_fn(tokens) -> {"sense_entropy": float, "proto_distance": float, "warp": float}
    """

    def __init__(
        self,
        thresholds: AHGThresholds,
        generate_fn: Callable,
        retriever_fn: Optional[Callable] = None,
        cite_align_fn: Optional[Callable] = None,
        cms_signals_fn: Optional[Callable] = None,
    ):
        self.th = thresholds
        self.generate = generate_fn
        self.retrieve = retriever_fn
        self.cite_align = cite_align_fn
        self.cms_signals = cms_signals_fn

    def assess(self, prompt: str, draft: Optional[dict] = None) -> AHGDecision:
        if draft is None:
            draft = self.generate(prompt, temperature=0.2, do_sample=False)
        answer = str(draft.get("out_text", ""))
        logits = draft.get("logits", None)
        tokens = draft.get("tokens", None)

        signals: Dict[str, float] = {}
        if isinstance(logits, torch.Tensor):
            signals["entropy"] = _entropy_from_logits(logits)
            signals["margin"] = _margin_from_logits(logits)
        else:
            signals["entropy"] = 9.99
            signals["margin"] = 0.0

        signals["consistency"], _ = _self_consistency(
            self.generate, prompt, n=5, temperature=0.9
        )

        if self.cms_signals is not None and tokens is not None:
            cs = self.cms_signals(tokens)
            signals["cms_sense_entropy"] = float(cs.get("sense_entropy", 0.0))
            signals["cms_proto_dist"] = float(cs.get("proto_distance", 0.0))
            signals["cms_warp"] = float(cs.get("warp", 1.0))
        else:
            signals["cms_sense_entropy"] = 0.0
            signals["cms_proto_dist"] = 0.0
            signals["cms_warp"] = 1.0

        docs = None
        if self.retrieve is not None:
            docs = self.retrieve(prompt, k=5) or []
            scores = _normalize([float(d.get("score", 0.0)) for d in docs])
            signals["retrieval_max"] = max(scores) if scores else 0.0
            signals["retrieval_cov"] = float(len([s for s in scores if s >= 0.5])) / 5.0
            if self.cite_align is not None and docs:
                signals["cite_align"] = float(self.cite_align(answer, docs))
            else:
                signals["cite_align"] = 0.0
        else:
            signals["retrieval_max"] = 0.0
            signals["retrieval_cov"] = 0.0
            signals["cite_align"] = 0.0

        signals["sanity"] = _numeric_temporal_sanity(answer)
        action, reason = self._decide(signals)
        return AHGDecision(action=action, reason=reason, signals=signals, docs=docs)

    def _decide(self, s: Dict[str, float]) -> Tuple[str, str]:
        th = self.th
        red_flags = [
            s["entropy"] > th.max_entropy,
            s["margin"] < th.min_margin,
            s["consistency"] < th.min_consistency,
            s["cms_sense_entropy"] > th.max_cms_sense_entropy,
            s["cms_proto_dist"] > th.max_cms_proto_dist,
        ]
        weak_grounding = (
            s["retrieval_max"] < th.min_retrieval_score
            or s["retrieval_cov"] < th.min_coverage
        )
        weak_citation = (
            th.require_citations_when_retrieved
            and self.retrieve is not None
            and s.get("cite_align", 0.0) < th.min_cite_align
        )

        if any(red_flags) and weak_grounding:
            if self.retrieve is not None:
                return "RETRIEVE", "High uncertainty with weak grounding."
            if th.allow_refuse:
                return "REFUSE", "High uncertainty and no retrieval source."
            return "ASK", "Need clarification or external evidence."
        if any(red_flags):
            return "ASK", "Uncertainty/novelty above threshold."
        if weak_grounding and th.require_citations_when_retrieved:
            if self.retrieve is not None:
                return "RETRIEVE", "Insufficient grounding; retrieving citations."
            return "ASK", "Insufficient grounding; provide source or context."
        if weak_citation:
            if self.retrieve is not None:
                return "RETRIEVE", "Citation alignment is weak; retrieving stronger evidence."
            return "ASK", "Citation support is weak; provide source or context."
        if s["sanity"] < 0.6:
            return "ASK", "Sanity checks failed for numeric/temporal claims."
        return "ALLOW", "Signals within bounds."

    def answer(self, prompt: str) -> Dict:
        draft = self.generate(prompt, temperature=0.2, do_sample=False)
        decision = self.assess(prompt, draft=draft)

        if decision.action == "ALLOW":
            return {
                "mode": "answer",
                "text": str(draft.get("out_text", "")),
                "signals": decision.signals,
                "citations": None,
            }

        if decision.action == "RETRIEVE" and self.retrieve is not None:
            docs = decision.docs or self.retrieve(prompt, k=5) or []
            context = "\n\n".join([str(d.get("text", "")) for d in docs[:3]])
            grounded = self.generate(
                (
                    "[CONTEXT]\n"
                    f"{context}\n\n"
                    "[QUESTION]\n"
                    f"{prompt}\n\n"
                    "[INSTRUCTIONS]\n"
                    "Use only CONTEXT and cite source ids."
                ),
                temperature=0.2,
                do_sample=False,
            )
            cites = [
                {"source_id": d.get("source_id", ""), "score": float(d.get("score", 0.0))}
                for d in docs[:3]
            ]
            return {
                "mode": "grounded",
                "text": str(grounded.get("out_text", "")),
                "signals": decision.signals,
                "citations": cites,
            }

        if decision.action == "ASK":
            return {
                "mode": "ask",
                "text": (
                    "I can answer accurately if you provide one more detail "
                    "(timeframe/source/scope), or allow retrieval."
                ),
                "signals": decision.signals,
            }

        return {
            "mode": "refuse",
            "text": "I do not have enough reliable evidence to answer accurately.",
            "signals": decision.signals,
        }
