"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context compression memory.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, safe_jsonable, foundation_trace, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .wm_shared_slot_store import SharedSlotStore, tensor_fingerprint
except Exception:  # pragma: no cover - fallback for isolated import
    SharedSlotStore = Any  # type: ignore

    def tensor_fingerprint(x: torch.Tensor, max_values: int = 64) -> str:
        if x.numel() == 0:
            return "empty"
        flat = x.detach().float().cpu().reshape(-1)[:max_values]
        payload = ",".join(f"{float(v):.6f}" for v in flat).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = repr(safe_jsonable(payload)).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"


def _safe_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().cpu())


@dataclass
class ContextCompressionConfig:
    """Configuration for context compression and episodic candidate creation.

    The compressor is intentionally bounded: it computes salience over the
    current context/response tensors only and never scans global memory.
    """

    dim: int
    max_context_tokens: int = 64
    max_response_tokens: int = 64
    top_k_context_tokens: int = 16
    top_k_response_tokens: int = 16
    max_parameter_refs: int = 32
    max_parameter_values_for_fingerprint: int = 16
    compression_norm_limit: float = 32.0
    default_memory_type: str = "ltm"
    default_geometry_map: str = "episodic"
    default_task_mode: str = "context_episode"
    require_explicit_store_permission: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.max_context_tokens <= 0 or self.max_response_tokens <= 0:
            raise ValueError("max token windows must be positive")
        if self.top_k_context_tokens <= 0 or self.top_k_response_tokens <= 0:
            raise ValueError("top-k token counts must be positive")
        if self.max_parameter_refs < 0:
            raise ValueError("max_parameter_refs must be non-negative")
        if self.compression_norm_limit <= 0:
            raise ValueError("compression_norm_limit must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class ParameterWeightReference:
    """Reference to a model parameter/weight related to a context episode.

    This stores metadata and a bounded fingerprint only. It does not store full
    weight tensors and never mutates model parameters.
    """

    parameter_name: str
    module_path: str
    reference_kind: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    requires_grad: bool
    relevance_score: float
    fingerprint: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "module_path": self.module_path,
            "reference_kind": self.reference_kind,
            "shape": list(self.shape),
            "numel": int(self.numel),
            "dtype": self.dtype,
            "requires_grad": bool(self.requires_grad),
            "relevance_score": float(self.relevance_score),
            "fingerprint": self.fingerprint,
            "reason": self.reason,
            "paamax_metadata": {
                "trace_type": "context_parameter_weight_reference",
                "full_weight_tensor_stored": False,
                "parameter_mutation": False,
                "reference_only": True,
            },
        }


@dataclass
class ContextCompressionOutput:
    compressed_context: torch.Tensor
    compressed_response: Optional[torch.Tensor]
    episode_vector: torch.Tensor
    context_salience: torch.Tensor
    response_salience: Optional[torch.Tensor]
    context_top_indices: torch.Tensor
    response_top_indices: Optional[torch.Tensor]
    content_signature_id: str
    context_fingerprint: str
    response_fingerprint: Optional[str]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_signature_id": self.content_signature_id,
            "context_fingerprint": self.context_fingerprint,
            "response_fingerprint": self.response_fingerprint,
            "compressed_context_shape": list(self.compressed_context.shape),
            "compressed_response_shape": None if self.compressed_response is None else list(self.compressed_response.shape),
            "episode_vector_shape": list(self.episode_vector.shape),
            "context_top_indices": self.context_top_indices.detach().cpu().tolist(),
            "response_top_indices": None if self.response_top_indices is None else self.response_top_indices.detach().cpu().tolist(),
            "trace": safe_jsonable(self.trace),
        }


@dataclass
class ContextEpisodicMemoryCandidate:
    """Consolidation candidate linking context, response, and weight refs.

    This is the bridge object that can later become an episodic project/chat
    memory. By default it is a proposal only; storage requires explicit opt-in.
    """

    candidate_id: str
    project_id: str
    chat_id: str
    episode_id: str
    source_kind: str
    context_map: str
    task_mode: str
    context_fingerprint: str
    response_fingerprint: Optional[str]
    episode_vector_fingerprint: str
    episode_vector_norm: float
    parameter_refs: List[ParameterWeightReference] = field(default_factory=list)
    weight_refs: List[ParameterWeightReference] = field(default_factory=list)
    compression_trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "project_id": self.project_id,
            "chat_id": self.chat_id,
            "episode_id": self.episode_id,
            "source_kind": self.source_kind,
            "context_map": self.context_map,
            "task_mode": self.task_mode,
            "context_fingerprint": self.context_fingerprint,
            "response_fingerprint": self.response_fingerprint,
            "episode_vector_fingerprint": self.episode_vector_fingerprint,
            "episode_vector_norm": self.episode_vector_norm,
            "parameter_refs": [p.to_dict() for p in self.parameter_refs],
            "weight_refs": [p.to_dict() for p in self.weight_refs],
            "compression_trace": safe_jsonable(self.compression_trace),
            "metadata": safe_jsonable(self.metadata),
            "created_at": self.created_at,
            "paamax_metadata": {
                "trace_type": "context_episodic_memory_candidate",
                "write_permission_required": True,
                "write_permission_granted": False,
                "consolidation_candidate": True,
                "project_or_chat_episode": True,
                "parameter_reference_only": True,
                "weight_mutation": False,
                "memory_store_mutation_by_default": False,
            },
        }


@dataclass
class ContextMemoryStageResult:
    """Result of optionally staging a candidate into shared/QH stores."""

    candidate: ContextEpisodicMemoryCandidate
    stored: bool
    shared_slot_result: Optional[Dict[str, Any]] = None
    qh_record: Optional[Dict[str, Any]] = None
    reason: str = "proposal_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "stored": bool(self.stored),
            "shared_slot_result": self.shared_slot_result,
            "qh_record": self.qh_record,
            "reason": self.reason,
            "paamax_metadata": {
                "trace_type": "context_memory_stage_result",
                "stored": bool(self.stored),
                "write_permission_required": True,
                "write_permission_granted": bool(self.stored),
                "consolidation_candidate": True,
            },
        }


class ContextCompressor(nn.Module):
    """Compress context/response tokens into a bounded episodic vector."""

    def __init__(self, config: ContextCompressionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.context_score = nn.Linear(config.dim, 1)
        self.response_score = nn.Linear(config.dim, 1)
        self.context_norm = nn.LayerNorm(config.dim)
        self.response_norm = nn.LayerNorm(config.dim)
        self.episode_fusion = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.GELU(),
            nn.LayerNorm(config.dim),
        )

    def _validate_tokens(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        ensure_finite_tensor(name, tensor)
        if tensor.dim() != 3 or tensor.size(-1) != self.config.dim:
            raise ValueError(f"{name} must be [B,T,{self.config.dim}]")
        return tensor

    def _bounded_window(self, tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if tensor.size(1) <= max_tokens:
            return tensor
        # Keep a deterministic prefix/suffix blend to avoid unbounded scans.
        half = max_tokens // 2
        tail = max_tokens - half
        return torch.cat([tensor[:, :half], tensor[:, -tail:]], dim=1)

    def _compress(self, tensor: torch.Tensor, scorer: nn.Linear, norm: nn.LayerNorm, top_k: int):
        h = norm(tensor)
        logits = scorer(h).squeeze(-1)
        salience = torch.softmax(logits, dim=-1)
        compressed = torch.sum(salience.unsqueeze(-1) * h, dim=1)
        compressed = clamp_norm(compressed, self.config.compression_norm_limit)
        k = min(top_k, tensor.size(1))
        _, top_idx = torch.topk(salience, k=k, dim=-1)
        return compressed, salience, top_idx

    def forward(self, context: torch.Tensor, response: Optional[torch.Tensor] = None) -> ContextCompressionOutput:
        context = self._validate_tokens("context", context)
        context_w = self._bounded_window(context, self.config.max_context_tokens)
        compressed_context, context_salience, context_top_idx = self._compress(
            context_w, self.context_score, self.context_norm, self.config.top_k_context_tokens
        )

        compressed_response = None
        response_salience = None
        response_top_idx = None
        response_fp = None
        if response is not None:
            response = self._validate_tokens("response", response)
            response_w = self._bounded_window(response, self.config.max_response_tokens)
            compressed_response, response_salience, response_top_idx = self._compress(
                response_w, self.response_score, self.response_norm, self.config.top_k_response_tokens
            )
            response_fp = tensor_fingerprint(response_w)
        else:
            compressed_response = torch.zeros_like(compressed_context)

        episode_vector = self.episode_fusion(torch.cat([compressed_context, compressed_response], dim=-1))
        episode_vector = clamp_norm(episode_vector, self.config.compression_norm_limit)

        context_fp = tensor_fingerprint(context_w)
        signature = _stable_id("ctxsig", {
            "context_fp": context_fp,
            "response_fp": response_fp,
            "context_shape": list(context.shape),
            "response_shape": None if response is None else list(response.shape),
        })
        trace = foundation_trace(
            trace_type="context_compression_trace",
            module=__name__,
            message="context and response compressed for episodic consolidation candidate",
            payload={
                "content_signature_id": signature,
                "context_shape": list(context.shape),
                "context_window_shape": list(context_w.shape),
                "response_shape": None if response is None else list(response.shape),
                "context_top_k": min(self.config.top_k_context_tokens, context_w.size(1)),
                "response_top_k": None if response is None else min(self.config.top_k_response_tokens, response_w.size(1)),
                "episode_vector_norm_mean": float(episode_vector.detach().float().norm(dim=-1).mean().cpu()),
            },
        )
        return ContextCompressionOutput(
            compressed_context=compressed_context,
            compressed_response=None if response is None else compressed_response,
            episode_vector=episode_vector,
            context_salience=context_salience,
            response_salience=response_salience,
            context_top_indices=context_top_idx,
            response_top_indices=response_top_idx,
            content_signature_id=signature,
            context_fingerprint=context_fp,
            response_fingerprint=response_fp,
            trace=trace,
        )


class ContextParameterReferenceExtractor:
    """Extract bounded parameter/weight references related to context episodes."""

    DEFAULT_KEYWORDS = (
        "context",
        "wm_context",
        "context_to_wm",
        "context_map",
        "context_geometry",
        "memory_augmented",
        "retrieval",
        "attention",
        "shared_slot",
        "quantum_holographic",
        "qh",
        "commit",
        "episodic",
        "ltm",
        "mann",
        "spcp",
        "curved",
        "depth",
    )

    def __init__(self, config: ContextCompressionConfig):
        config.validate()
        self.config = config

    def _kind(self, name: str, tensor: torch.Tensor) -> str:
        low = name.lower()
        if low.endswith(".weight") or "weight" in low:
            return "weight"
        if low.endswith(".bias") or "bias" in low:
            return "bias"
        if tensor.dim() >= 2:
            return "weight_like_parameter"
        return "parameter"

    def _module_path(self, name: str) -> str:
        return name.rsplit(".", 1)[0] if "." in name else ""

    def _score(self, name: str, tensor: torch.Tensor, parameter_hints: Sequence[str]) -> Tuple[float, str]:
        low = name.lower()
        score = 0.0
        reasons: List[str] = []
        for kw in self.DEFAULT_KEYWORDS:
            if kw in low:
                score += 1.0
                reasons.append(f"keyword:{kw}")
        for hint in parameter_hints:
            if hint and hint.lower() in low:
                score += 2.0
                reasons.append(f"hint:{hint}")
        if "weight" in low:
            score += 0.5
            reasons.append("weight_parameter")
        if tensor.dim() >= 2:
            score += 0.25
            reasons.append("matrix_or_higher_rank")
        if tensor.requires_grad:
            score += 0.25
            reasons.append("trainable")
        return float(score), ",".join(reasons) if reasons else "low_direct_name_match"

    def extract(
        self,
        model: Optional[nn.Module],
        *,
        parameter_hints: Optional[Iterable[str]] = None,
        min_relevance_score: float = 0.25,
    ) -> List[ParameterWeightReference]:
        if model is None:
            return []
        hints = list(parameter_hints or [])
        refs: List[ParameterWeightReference] = []
        for name, parameter in model.named_parameters(recurse=True):
            if len(refs) >= max(self.config.max_parameter_refs * 4, self.config.max_parameter_refs):
                break
            score, reason = self._score(name, parameter, hints)
            if score < min_relevance_score:
                continue
            refs.append(
                ParameterWeightReference(
                    parameter_name=name,
                    module_path=self._module_path(name),
                    reference_kind=self._kind(name, parameter),
                    shape=tuple(int(v) for v in parameter.shape),
                    numel=int(parameter.numel()),
                    dtype=str(parameter.dtype),
                    requires_grad=bool(parameter.requires_grad),
                    relevance_score=score,
                    fingerprint=tensor_fingerprint(parameter.detach(), self.config.max_parameter_values_for_fingerprint),
                    reason=reason,
                )
            )
        refs.sort(key=lambda r: (-r.relevance_score, r.parameter_name))
        return refs[: self.config.max_parameter_refs]


class ContextEpisodicMemoryBuilder(nn.Module):
    """Build context-linked episodic memory candidates and optional store records."""

    def __init__(self, config: ContextCompressionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.compressor = ContextCompressor(config)
        self.reference_extractor = ContextParameterReferenceExtractor(config)

    def build_candidate(
        self,
        *,
        context: torch.Tensor,
        response: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        project_id: str = "unknown_project",
        chat_id: str = "unknown_chat",
        episode_id: str = "unknown_episode",
        context_map: Optional[str] = None,
        task_mode: Optional[str] = None,
        task_hints: Optional[Iterable[str]] = None,
        parameter_hints: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ContextEpisodicMemoryCandidate, torch.Tensor, ContextCompressionOutput]:
        compression = self.compressor(context, response)
        parameter_refs = self.reference_extractor.extract(
            model,
            parameter_hints=list(parameter_hints or []) + list(task_hints or []) + [context_map or ""],
        )
        weight_refs = [p for p in parameter_refs if "weight" in p.reference_kind]
        vector = compression.episode_vector.mean(dim=0).detach()
        candidate_id = _stable_id("ctxmem", {
            "project_id": project_id,
            "chat_id": chat_id,
            "episode_id": episode_id,
            "content_signature_id": compression.content_signature_id,
            "context_map": context_map or self.config.default_geometry_map,
            "task_mode": task_mode or self.config.default_task_mode,
            "parameter_ref_names": [p.parameter_name for p in parameter_refs],
        })
        candidate = ContextEpisodicMemoryCandidate(
            candidate_id=candidate_id,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
            source_kind="context_buffer_response_episode",
            context_map=context_map or self.config.default_geometry_map,
            task_mode=task_mode or self.config.default_task_mode,
            context_fingerprint=compression.context_fingerprint,
            response_fingerprint=compression.response_fingerprint,
            episode_vector_fingerprint=tensor_fingerprint(vector),
            episode_vector_norm=_safe_norm(vector),
            parameter_refs=parameter_refs,
            weight_refs=weight_refs,
            compression_trace=compression.to_dict(),
            metadata={
                "task_hints": list(task_hints or []),
                "parameter_reference_count": len(parameter_refs),
                "weight_reference_count": len(weight_refs),
                "response_linked": response is not None,
                **(metadata or {}),
            },
        )
        return candidate, vector, compression

    def build_and_stage(
        self,
        *,
        context: torch.Tensor,
        response: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        shared_slot_store: Any = None,
        qh_storage: Any = None,
        allow_store: bool = False,
        write_permission: bool = False,
        project_id: str = "unknown_project",
        chat_id: str = "unknown_chat",
        episode_id: str = "unknown_episode",
        context_map: Optional[str] = None,
        task_mode: Optional[str] = None,
        task_hints: Optional[Iterable[str]] = None,
        parameter_hints: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMemoryStageResult:
        candidate, vector, _ = self.build_candidate(
            context=context,
            response=response,
            model=model,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
            context_map=context_map,
            task_mode=task_mode,
            task_hints=task_hints,
            parameter_hints=parameter_hints,
            metadata=metadata,
        )
        if not allow_store:
            return ContextMemoryStageResult(candidate=candidate, stored=False, reason="proposal_only_explicit_store_permission_not_granted")
        if shared_slot_store is None:
            return ContextMemoryStageResult(candidate=candidate, stored=False, reason="proposal_only_missing_shared_slot_store")
        if self.config.require_explicit_store_permission and not write_permission:
            return ContextMemoryStageResult(candidate=candidate, stored=False, reason="proposal_only_write_permission_false")

        slot_result = shared_slot_store.write_slot(
            memory_type=self.config.default_memory_type,
            local_slot_id=candidate.candidate_id,
            content=vector,
            owner="ltm",
            geometry_map=candidate.context_map,
            depth_index=0,
            confidence=1.0,
            write_permission=write_permission,
            metadata=candidate.to_dict(),
        )
        qh_record_dict = None
        if qh_storage is not None:
            qh_record = qh_storage.create_record(
                canonical_slot_id=slot_result.canonical_id,
                vector=vector,
                depth_index=0,
                bank_name="context_buffer_episode",
                geometry_name=candidate.context_map,
                triplet_index=0,
                memory_type=self.config.default_memory_type,
                task_mode=candidate.task_mode,
                confidence=1.0,
                write_permission=write_permission,
                metadata={"context_memory_candidate_id": candidate.candidate_id},
            )
            qh_record_dict = qh_record.to_dict()
        return ContextMemoryStageResult(
            candidate=candidate,
            stored=True,
            shared_slot_result=slot_result.to_dict(),
            qh_record=qh_record_dict,
            reason="stored_with_explicit_permission",
        )


# ---------------------------------------------------------------------------
# WM context-compression quality contract
# ---------------------------------------------------------------------------

def wm_context_compression_contract() -> dict:
    """Return serialization-safe metadata for context compression extension."""
    return foundation_trace(
        trace_type="wm_context_compression_contract",
        module=__name__,
        message="context compression, parameter/weight references, and episodic candidate creation are available",
        payload={
            "context_compression": True,
            "response_linking": True,
            "parameter_weight_references": "metadata_and_bounded_fingerprints_only",
            "episodic_consolidation_candidates": True,
            "default_store_mutation": False,
            "explicit_store_permission_required": True,
            "shared_slot_and_qh_compatible": True,
            "shape_checks": "context/response must be [B,T,D]",
            "finite_checks": True,
            "bounded_parameter_refs": True,
        },
        paamax={
            "write_permission_required": True,
            "write_permission_granted": False,
            "context_memory_extension": True,
        },
    )


__all__ = [
    "ContextCompressionConfig",
    "ParameterWeightReference",
    "ContextCompressionOutput",
    "ContextEpisodicMemoryCandidate",
    "ContextMemoryStageResult",
    "ContextCompressor",
    "ContextParameterReferenceExtractor",
    "ContextEpisodicMemoryBuilder",
    "wm_context_compression_contract",
]
