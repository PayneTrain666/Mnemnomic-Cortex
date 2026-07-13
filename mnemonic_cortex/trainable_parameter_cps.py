"""Trainable central parameter storage for ordinary PyTorch layers.

This module is deliberately independent of QSPIN, shared slots, and QH storage.
It only replaces eligible local modules after an explicit transaction commit.
"""

from __future__ import annotations

import fnmatch
import math
import threading
import uuid
import weakref
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class TrainableParameterCPSConfig:
    """Policy used while discovering and optionally compressing parameters."""

    include: Tuple[str, ...] = ("*",)
    exclude: Tuple[str, ...] = ()
    allowed_module_types: Tuple[str, ...] = (
        "Linear",
        "Embedding",
        "MultiheadAttention",
    )
    exact_dtype: Optional[str] = None
    max_rank: int = 8
    adaptive_rank: bool = True
    reconstruction_tolerance: float = 1.0e-6
    output_tolerance: float = 1.0e-6
    min_cohort_size: int = 2
    enable_compression: bool = False
    default_commit_policy: str = "manual"

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", tuple(self.include))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        object.__setattr__(self, "allowed_module_types", tuple(self.allowed_module_types))
        if self.max_rank < 0:
            raise ValueError("max_rank must be non-negative")
        if self.reconstruction_tolerance < 0 or self.output_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if self.min_cohort_size < 2:
            raise ValueError("min_cohort_size must be at least two")
        if self.default_commit_policy != "manual":
            raise ValueError("default_commit_policy must remain 'manual'")

    @property
    def compression_enabled(self) -> bool:
        return self.enable_compression


@dataclass(frozen=True)
class ParameterRefMetadata:
    handle: str
    module_path: str
    parameter_name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    numel: int
    aliases: Tuple[str, ...] = ()
    storage_kind: str = "pending"
    slab_index: int = -1
    offset: int = -1
    rank: int = 0


@dataclass(frozen=True)
class ProposalMetadata:
    proposal_id: str
    references: Tuple[ParameterRefMetadata, ...]
    replacements: Tuple[str, ...]
    rejected: Mapping[str, str] = field(default_factory=dict)
    state: str = "staged"


@dataclass(frozen=True)
class EvaluationMetadata:
    proposal_id: str
    original_scalars: int
    proposed_scalars: int
    scalar_savings: int
    compressed_handles: Tuple[str, ...] = ()
    exact_handles: Tuple[str, ...] = ()
    max_reconstruction_error: float = 0.0
    max_output_error: float = 0.0
    compression_applied: bool = False


@dataclass(frozen=True)
class CommitMetadata:
    transaction_id: str
    proposal_id: str
    replacements: Tuple[str, ...]
    parameter_handles: Tuple[str, ...]
    committed: bool = True
    rolled_back: bool = False


@dataclass(frozen=True)
class ParameterCohort:
    """Compatible handles considered together for optional compression."""

    cohort_id: str
    role: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    handles: Tuple[str, ...]


@dataclass(frozen=True)
class RollbackRecord:
    """Audit record for a completed reversible ownership transaction."""

    transaction_id: str
    restored_paths: Tuple[str, ...]
    optimizer_rebuild_required: bool = True


# Public names used by the architectural plan. Metadata suffixes remain as
# backwards-compatible spellings for early adopters of this module.
TrainableParameterRef = ParameterRefMetadata
ConsolidationProposal = ProposalMetadata
ConsolidationEvaluation = EvaluationMetadata
ConsolidationCommit = CommitMetadata


@dataclass
class _Candidate:
    path: str
    module: nn.Module
    kind: str
    parameters: Dict[str, nn.Parameter]


@dataclass
class _StagedPlan:
    root_ref: "weakref.ReferenceType[nn.Module]"
    proposal: ProposalMetadata
    candidates: Tuple[_Candidate, ...]
    handle_by_parameter_id: Dict[int, str]
    parameter_by_handle: Dict[str, nn.Parameter]


@dataclass
class _Representation:
    kind: str
    shape: Tuple[int, ...]
    slab_index: int = -1
    offset: int = -1
    numel: int = 0
    template_slab: int = -1
    template_offset: int = -1
    left_slab: int = -1
    left_offset: int = -1
    right_slab: int = -1
    right_offset: int = -1
    rank: int = 0


class _CPSConsumer:
    """Non-owning access shared by all functional wrappers."""

    def _bind_cps(self, cps: "TrainableParameterCPS") -> None:
        object.__setattr__(self, "_cps_ref", weakref.ref(cps))

    def _cps(self) -> "TrainableParameterCPS":
        cps = self.__dict__["_cps_ref"]()
        if cps is None:
            raise RuntimeError("the owning TrainableParameterCPS no longer exists")
        return cps


class CPSBackedLinear(_CPSConsumer, nn.Module):
    """Functional ``nn.Linear`` facade whose tensors are owned by a CPS."""

    def __init__(
        self,
        cps: "TrainableParameterCPS",
        weight_handle: str,
        bias_handle: Optional[str],
        source: nn.Linear,
    ) -> None:
        super().__init__()
        self._bind_cps(cps)
        object.__setattr__(self, "_weight_handle", weight_handle)
        object.__setattr__(self, "_bias_handle", bias_handle)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.train(source.training)

    @property
    def weight(self) -> torch.Tensor:
        return self._cps().materialize(self.__dict__["_weight_handle"])

    @property
    def bias(self) -> Optional[torch.Tensor]:
        handle = self.__dict__["_bias_handle"]
        return None if handle is None else self._cps().materialize(handle)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.__dict__['_bias_handle'] is not None}, cps_backed=True"
        )


class CPSBackedEmbedding(_CPSConsumer, nn.Module):
    """Functional ``nn.Embedding`` facade whose weight is owned by a CPS."""

    def __init__(
        self,
        cps: "TrainableParameterCPS",
        weight_handle: str,
        source: nn.Embedding,
    ) -> None:
        super().__init__()
        self._bind_cps(cps)
        object.__setattr__(self, "_weight_handle", weight_handle)
        for name in (
            "num_embeddings",
            "embedding_dim",
            "padding_idx",
            "max_norm",
            "norm_type",
            "scale_grad_by_freq",
            "sparse",
        ):
            setattr(self, name, getattr(source, name))
        self.train(source.training)

    @property
    def weight(self) -> torch.Tensor:
        return self._cps().materialize(self.__dict__["_weight_handle"])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, cps_backed=True"
        )


class CPSBackedMultiheadAttention(_CPSConsumer, nn.Module):
    """Functional facade for standard same-dimension MultiheadAttention."""

    def __init__(
        self,
        cps: "TrainableParameterCPS",
        handles: Mapping[str, Optional[str]],
        source: nn.MultiheadAttention,
    ) -> None:
        super().__init__()
        self._bind_cps(cps)
        object.__setattr__(self, "_handles", dict(handles))
        for name in (
            "embed_dim",
            "num_heads",
            "dropout",
            "batch_first",
            "head_dim",
            "kdim",
            "vdim",
            "add_zero_attn",
            "_qkv_same_embed_dim",
        ):
            setattr(self, name, getattr(source, name))
        self.out_proj = CPSBackedLinear(
            cps,
            handles["out_proj.weight"],  # type: ignore[arg-type]
            handles.get("out_proj.bias"),
            source.out_proj,
        )
        self.train(source.training)

    def _tensor(self, name: str) -> Optional[torch.Tensor]:
        handle = self.__dict__["_handles"].get(name)
        return None if handle is None else self._cps().materialize(handle)

    @property
    def in_proj_weight(self) -> Optional[torch.Tensor]:
        return self._tensor("in_proj_weight")

    @property
    def in_proj_bias(self) -> Optional[torch.Tensor]:
        return self._tensor("in_proj_bias")

    @property
    def bias_k(self) -> Optional[torch.Tensor]:
        return self._tensor("bias_k")

    @property
    def bias_v(self) -> Optional[torch.Tensor]:
        return self._tensor("bias_v")

    @property
    def q_proj_weight(self) -> None:
        return None

    @property
    def k_proj_weight(self) -> None:
        return None

    @property
    def v_proj_weight(self) -> None:
        return None

    def merge_masks(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ):
        # TransformerEncoderLayer's optimized path calls this public MHA helper.
        return nn.MultiheadAttention.merge_masks(
            self, attn_mask, key_padding_mask, query
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = query.transpose(1, 0), key.transpose(1, 0)
                    value = key
            else:
                query, key, value = (
                    query.transpose(1, 0),
                    key.transpose(1, 0),
                    value.transpose(1, 0),
                )
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            output = output.transpose(1, 0)
        return output, weights

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}, cps_backed=True"
        )


class TrainableParameterCPS(nn.Module):
    """Own flat trainable slabs and transactionally install functional facades."""

    _ATTACHMENT_NAME = "_trainable_parameter_cps"

    def __init__(self, config: Optional[TrainableParameterCPSConfig] = None) -> None:
        super().__init__()
        self.config = config or TrainableParameterCPSConfig()
        self.slabs = nn.ParameterList()
        self._registry: Dict[str, ParameterRefMetadata] = {}
        self._representations: Dict[str, _Representation] = {}
        self._staged: Optional[_StagedPlan] = None
        self._evaluation: Optional[EvaluationMetadata] = None
        self._commit: Optional[CommitMetadata] = None
        object.__setattr__(self, "_original_modules", {})
        object.__setattr__(self, "_root_ref", None)
        object.__setattr__(self, "_lock", threading.RLock())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Remain transparent if attached to a sequential-style root container."""

        return input

    @property
    def registry(self) -> Mapping[str, ParameterRefMetadata]:
        return dict(self._registry)

    @property
    def proposal(self) -> Optional[ProposalMetadata]:
        return None if self._staged is None else self._staged.proposal

    @property
    def evaluation(self) -> Optional[EvaluationMetadata]:
        return self._evaluation

    @property
    def commit_metadata(self) -> Optional[CommitMetadata]:
        return self._commit

    def _matches_policy(self, path: str, parameter_names: Iterable[str]) -> bool:
        names = [path] + [f"{path}.{name}" if path else name for name in parameter_names]
        included = any(
            fnmatch.fnmatchcase(name, pattern)
            for name in names
            for pattern in self.config.include
        )
        excluded = any(
            fnmatch.fnmatchcase(name, pattern)
            for name in names
            for pattern in self.config.exclude
        )
        return included and not excluded

    @staticmethod
    def _module_entries(root: nn.Module) -> List[Tuple[str, nn.Module]]:
        try:
            return list(root.named_modules(remove_duplicate=False))
        except TypeError:  # pragma: no cover - old PyTorch compatibility
            return list(root.named_modules())

    @staticmethod
    def _candidate_parameters(module: nn.Module, kind: str) -> Dict[str, nn.Parameter]:
        if kind == "linear":
            result = {"weight": module.weight}  # type: ignore[attr-defined]
            if module.bias is not None:  # type: ignore[attr-defined]
                result["bias"] = module.bias  # type: ignore[attr-defined]
            return result
        if kind == "embedding":
            return {"weight": module.weight}  # type: ignore[attr-defined]
        mha = module
        result = {"in_proj_weight": mha.in_proj_weight}  # type: ignore[attr-defined]
        for name in ("in_proj_bias", "bias_k", "bias_v"):
            parameter = getattr(mha, name)
            if parameter is not None:
                result[name] = parameter
        result["out_proj.weight"] = mha.out_proj.weight
        if mha.out_proj.bias is not None:
            result["out_proj.bias"] = mha.out_proj.bias
        return result

    def stage(self, root: nn.Module) -> ProposalMetadata:
        """Discover eligible modules without changing ``root``."""

        with self._lock:
            if self._commit is not None and self._commit.committed and not self._commit.rolled_back:
                raise RuntimeError("this CPS already has a committed transaction")
            entries = self._module_entries(root)
            mha_paths = {
                path
                for path, module in entries
                if isinstance(module, nn.MultiheadAttention)
            }
            candidates: List[_Candidate] = []
            rejected: Dict[str, str] = {}
            seen_candidate_modules: set[int] = set()
            replacement_paths: List[str] = []

            for path, module in entries:
                display = path or "<root>"
                lower = f"{path}.{module.__class__.__name__}".lower()
                if path == self._ATTACHMENT_NAME or path.startswith(self._ATTACHMENT_NAME + "."):
                    rejected[display] = "new CPS modules are never discoverable"
                    continue
                if isinstance(module, TrainableParameterCPS) or "cps" in lower:
                    rejected[display] = "new CPS modules are never discoverable"
                    continue
                if "qspin" in lower:
                    rejected[display] = "qspin modules are inert and excluded"
                    continue
                if any(path.startswith(parent + ".") for parent in mha_paths if parent != path):
                    continue

                if isinstance(module, nn.MultiheadAttention):
                    if not module._qkv_same_embed_dim or module.kdim != module.embed_dim or module.vdim != module.embed_dim:
                        rejected[display] = "unsupported MultiheadAttention dimensions"
                        continue
                    kind = "multihead_attention"
                elif isinstance(module, nn.Linear):
                    kind = "linear"
                elif isinstance(module, nn.Embedding):
                    kind = "embedding"
                else:
                    direct = list(module.named_parameters(recurse=False))
                    if direct:
                        rejected[display] = "unsupported parameterized module type"
                    continue
                if module.__class__.__name__ not in self.config.allowed_module_types:
                    rejected[display] = "module type disabled by configuration"
                    continue

                parameters = self._candidate_parameters(module, kind)
                if not self._matches_policy(path, parameters):
                    rejected[display] = "excluded by include/exclude policy"
                    continue
                if parametrize.is_parametrized(module) or (
                    kind == "multihead_attention" and parametrize.is_parametrized(module.out_proj)
                ):
                    rejected[display] = "parametrized modules are unsupported"
                    continue
                if any(isinstance(parameter, UninitializedParameter) for parameter in parameters.values()):
                    rejected[display] = "lazy/uninitialized parameters are unsupported"
                    continue
                if any(not parameter.requires_grad for parameter in parameters.values()):
                    rejected[display] = "frozen parameters are unsupported"
                    continue

                seen_candidate_modules.add(id(module))
                replacement_paths.append(path)
                candidates.append(_Candidate(path, module, kind, parameters))

            parameter_by_handle: Dict[str, nn.Parameter] = {}
            handle_by_parameter_id: Dict[int, str] = {}
            aliases: Dict[str, List[str]] = {}
            primary: Dict[str, Tuple[str, str]] = {}
            for candidate in candidates:
                for name, parameter in candidate.parameters.items():
                    full_name = f"{candidate.path}.{name}" if candidate.path else name
                    handle = handle_by_parameter_id.get(id(parameter))
                    if handle is None:
                        handle = f"p{len(handle_by_parameter_id):06d}"
                        handle_by_parameter_id[id(parameter)] = handle
                        parameter_by_handle[handle] = parameter
                        aliases[handle] = []
                        primary[handle] = (candidate.path, name)
                    aliases[handle].append(full_name)

            references = tuple(
                ParameterRefMetadata(
                    handle=handle,
                    module_path=primary[handle][0],
                    parameter_name=primary[handle][1],
                    shape=tuple(parameter.shape),
                    dtype=str(parameter.dtype),
                    device=str(parameter.device),
                    numel=parameter.numel(),
                    aliases=tuple(aliases[handle]),
                )
                for handle, parameter in parameter_by_handle.items()
            )
            proposal = ProposalMetadata(
                proposal_id=uuid.uuid4().hex,
                references=references,
                replacements=tuple(replacement_paths),
                rejected=rejected,
            )
            self._staged = _StagedPlan(
                root_ref=weakref.ref(root),
                proposal=proposal,
                candidates=tuple(candidates),
                handle_by_parameter_id=handle_by_parameter_id,
                parameter_by_handle=parameter_by_handle,
            )
            self._evaluation = self._analyze_plan(self._staged)
            return proposal

    discover = stage
    stage_exact = stage

    def group(
        self,
        proposal: Optional[ProposalMetadata] = None,
    ) -> Tuple[ParameterCohort, ...]:
        """Group discovered handles by role, shape, dtype, and device."""
        active = proposal or self.proposal
        if active is None:
            raise RuntimeError("discover/stage a model before grouping")
        grouped: Dict[
            Tuple[str, Tuple[int, ...], str, str],
            List[str],
        ] = {}
        for ref in active.references:
            key = (ref.parameter_name, ref.shape, ref.dtype, ref.device)
            grouped.setdefault(key, []).append(ref.handle)
        return tuple(
            ParameterCohort(
                cohort_id=f"cohort-{index:06d}",
                role=key[0],
                shape=key[1],
                dtype=key[2],
                device=key[3],
                handles=tuple(handles),
            )
            for index, (key, handles) in enumerate(
                sorted(grouped.items(), key=lambda item: str(item[0]))
            )
        )

    def _candidate_reconstruction_error(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        if original.numel() == 0:
            return 0.0
        return float((original.detach() - reconstructed.detach()).abs().max().item())

    def _factor_residual(
        self, residual: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, float]]:
        rows, cols = residual.shape
        maximum = min(self.config.max_rank, rows, cols)
        if maximum == 0:
            error = float(residual.abs().max().item()) if residual.numel() else 0.0
            return (residual.new_empty((rows, 0)), residual.new_empty((0, cols)), 0, error)
        work = residual.detach()
        try:
            u, s, vh = torch.linalg.svd(work, full_matrices=False)
        except RuntimeError:
            return None
        ranks = range(0, maximum + 1) if self.config.adaptive_rank else (maximum,)
        best: Optional[Tuple[torch.Tensor, torch.Tensor, int, float]] = None
        for rank in ranks:
            if rank == 0:
                left = work.new_empty((rows, 0))
                right = work.new_empty((0, cols))
                reconstructed = torch.zeros_like(work)
            else:
                left = u[:, :rank] * s[:rank]
                right = vh[:rank, :]
                reconstructed = left @ right
            error = float((work - reconstructed).abs().max().item())
            best = (left, right, rank, error)
            # The parameter max-error is a conservative bounded-input output proxy.
            tolerance = min(
                self.config.reconstruction_tolerance,
                self.config.output_tolerance,
            )
            if error <= tolerance:
                return best
        return best

    def _compression_groups(
        self, plan: _StagedPlan
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, torch.Tensor, int]],
        Dict[str, torch.Tensor],
        float,
    ]:
        factors: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        templates: Dict[str, torch.Tensor] = {}
        max_error = 0.0
        if not self.config.enable_compression:
            return factors, templates, max_error
        cohorts: Dict[Tuple[Any, ...], List[str]] = {}
        for handle, parameter in plan.parameter_by_handle.items():
            if parameter.dim() != 2 or not parameter.is_floating_point():
                continue
            key = (tuple(parameter.shape), parameter.dtype, parameter.device)
            cohorts.setdefault(key, []).append(handle)

        for handles in cohorts.values():
            if len(handles) < self.config.min_cohort_size:
                continue
            tensors = [plan.parameter_by_handle[handle].detach() for handle in handles]
            template = torch.stack(tensors).mean(dim=0)
            group_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]] = {}
            group_error = 0.0
            valid = True
            residual_scalars = 0
            for handle, tensor in zip(handles, tensors):
                result = self._factor_residual(tensor - template)
                if result is None:
                    valid = False
                    break
                left, right, rank, error = result
                group_error = max(group_error, error)
                if error > min(
                    self.config.reconstruction_tolerance,
                    self.config.output_tolerance,
                ):
                    valid = False
                    break
                residual_scalars += rank * (tensor.shape[0] + tensor.shape[1])
                group_factors[handle] = (left, right, rank)
            original_scalars = sum(tensor.numel() for tensor in tensors)
            proposed_scalars = template.numel() + residual_scalars
            if not valid or proposed_scalars >= original_scalars:
                continue
            template_key = min(handles)
            templates[template_key] = template
            for handle in handles:
                left, right, rank = group_factors[handle]
                factors[handle] = (left, right, rank)
                templates[handle] = template
            max_error = max(max_error, group_error)
        return factors, templates, max_error

    def _analyze_plan(self, plan: _StagedPlan) -> EvaluationMetadata:
        factors, templates, max_error = self._compression_groups(plan)
        original = sum(parameter.numel() for parameter in plan.parameter_by_handle.values())
        exact_handles = [handle for handle in plan.parameter_by_handle if handle not in factors]
        proposed = sum(plan.parameter_by_handle[handle].numel() for handle in exact_handles)
        seen_templates: set[int] = set()
        for handle, (left, right, _rank) in factors.items():
            template = templates[handle]
            if id(template) not in seen_templates:
                proposed += template.numel()
                seen_templates.add(id(template))
            proposed += left.numel() + right.numel()
        return EvaluationMetadata(
            proposal_id=plan.proposal.proposal_id,
            original_scalars=original,
            proposed_scalars=proposed,
            scalar_savings=original - proposed,
            compressed_handles=tuple(factors),
            exact_handles=tuple(exact_handles),
            max_reconstruction_error=max_error,
            max_output_error=max_error,
            compression_applied=bool(factors),
        )

    def evaluate(self, proposal: Optional[ProposalMetadata] = None) -> EvaluationMetadata:
        if self._staged is None or self._evaluation is None:
            raise RuntimeError("stage a model before evaluation")
        if proposal is not None and proposal.proposal_id != self._staged.proposal.proposal_id:
            raise ValueError("proposal does not belong to this CPS")
        return self._evaluation

    def evaluate_probe(self, probe) -> EvaluationMetadata:
        """Temporarily commit exact adapters and compare a pure probe output."""
        if not callable(probe):
            raise TypeError("probe must be callable and accept the staged root")
        if self._staged is None:
            raise RuntimeError("stage a model before probe evaluation")
        root = self._staged.root_ref()
        if root is None:
            raise RuntimeError("staged root no longer exists")

        def tensors(value):
            if torch.is_tensor(value):
                return [value.detach().clone()]
            if isinstance(value, (tuple, list)):
                out = []
                for item in value:
                    out.extend(tensors(item))
                return out
            if isinstance(value, dict):
                out = []
                for key in sorted(value):
                    out.extend(tensors(value[key]))
                return out
            return []

        before = tensors(probe(root))
        self.commit(self._staged.proposal, allow_compression=False)
        try:
            after = tensors(probe(root))
            error = 0.0
            if len(before) != len(after):
                error = float("inf")
            else:
                for lhs, rhs in zip(before, after):
                    if lhs.shape != rhs.shape:
                        error = float("inf")
                        break
                    if lhs.numel():
                        error = max(error, float((lhs - rhs).abs().max().item()))
        finally:
            self.rollback()
        self._evaluation = replace(
            self._evaluation or self._analyze_plan(self._staged),
            max_output_error=error,
        )
        return self._evaluation

    def _append_slab(self, value: torch.Tensor) -> int:
        self.slabs.append(nn.Parameter(value.detach().reshape(-1).clone()))
        return len(self.slabs) - 1

    def _build_storage(
        self,
        plan: _StagedPlan,
        *,
        allow_compression: bool = False,
    ) -> None:
        self.slabs = nn.ParameterList()
        self._registry.clear()
        self._representations.clear()
        if allow_compression and self.config.enable_compression:
            factors, templates, max_error = self._compression_groups(plan)
        else:
            factors, templates, max_error = {}, {}, 0.0

        exact_groups: Dict[Tuple[torch.dtype, torch.device], List[str]] = {}
        for handle, parameter in plan.parameter_by_handle.items():
            if handle not in factors:
                exact_groups.setdefault((parameter.dtype, parameter.device), []).append(handle)
        for handles in exact_groups.values():
            slab_index = self._append_slab(
                torch.cat([plan.parameter_by_handle[handle].detach().reshape(-1) for handle in handles])
            )
            offset = 0
            for handle in handles:
                parameter = plan.parameter_by_handle[handle]
                self._representations[handle] = _Representation(
                    kind="exact",
                    shape=tuple(parameter.shape),
                    slab_index=slab_index,
                    offset=offset,
                    numel=parameter.numel(),
                )
                offset += parameter.numel()

        template_locations: Dict[int, Tuple[int, int]] = {}
        for handle, (left, right, rank) in factors.items():
            template = templates[handle]
            identity = id(template)
            if identity not in template_locations:
                template_locations[identity] = (self._append_slab(template), 0)
            template_slab, template_offset = template_locations[identity]
            left_slab = self._append_slab(left) if rank else -1
            right_slab = self._append_slab(right) if rank else -1
            parameter = plan.parameter_by_handle[handle]
            self._representations[handle] = _Representation(
                kind="low_rank",
                shape=tuple(parameter.shape),
                template_slab=template_slab,
                template_offset=template_offset,
                left_slab=left_slab,
                left_offset=0,
                right_slab=right_slab,
                right_offset=0,
                rank=rank,
            )

        refs_by_handle = {ref.handle: ref for ref in plan.proposal.references}
        for handle, representation in self._representations.items():
            ref = refs_by_handle[handle]
            if representation.kind == "exact":
                updated = replace(
                    ref,
                    storage_kind="exact",
                    slab_index=representation.slab_index,
                    offset=representation.offset,
                )
            else:
                updated = replace(
                    ref,
                    storage_kind="shared_low_rank",
                    slab_index=representation.template_slab,
                    offset=representation.template_offset,
                    rank=representation.rank,
                )
            self._registry[handle] = updated

        stored = sum(parameter.numel() for parameter in self.slabs)
        original = sum(parameter.numel() for parameter in plan.parameter_by_handle.values())
        self._evaluation = replace(
            self._evaluation or self._analyze_plan(plan),
            proposed_scalars=stored,
            scalar_savings=original - stored,
            max_reconstruction_error=max_error,
            max_output_error=max_error,
            compression_applied=bool(factors),
        )

    def materialize(self, handle: str) -> torch.Tensor:
        """Return a differentiable view/reconstruction for a stable handle."""

        representation = self._representations.get(handle)
        if representation is None:
            raise KeyError(f"unknown CPS handle: {handle}")
        if representation.kind == "exact":
            return self.slabs[representation.slab_index].narrow(
                0, representation.offset, representation.numel
            ).view(representation.shape)
        rows, cols = representation.shape
        template = self.slabs[representation.template_slab].narrow(
            0, representation.template_offset, rows * cols
        ).view(rows, cols)
        if representation.rank == 0:
            return template
        left = self.slabs[representation.left_slab].narrow(
            0, representation.left_offset, rows * representation.rank
        ).view(rows, representation.rank)
        right = self.slabs[representation.right_slab].narrow(
            0, representation.right_offset, representation.rank * cols
        ).view(representation.rank, cols)
        return template + left @ right

    def _handle(self, plan: _StagedPlan, parameter: Optional[nn.Parameter]) -> Optional[str]:
        if parameter is None:
            return None
        return plan.handle_by_parameter_id[id(parameter)]

    def _make_wrapper(self, plan: _StagedPlan, candidate: _Candidate) -> nn.Module:
        if candidate.kind == "linear":
            source = candidate.module
            return CPSBackedLinear(
                self,
                self._handle(plan, source.weight),  # type: ignore[arg-type,union-attr]
                self._handle(plan, source.bias),  # type: ignore[union-attr]
                source,  # type: ignore[arg-type]
            )
        if candidate.kind == "embedding":
            source = candidate.module
            return CPSBackedEmbedding(
                self,
                self._handle(plan, source.weight),  # type: ignore[arg-type,union-attr]
                source,  # type: ignore[arg-type]
            )
        source = candidate.module
        handles = {
            "in_proj_weight": self._handle(plan, source.in_proj_weight),  # type: ignore[union-attr]
            "in_proj_bias": self._handle(plan, source.in_proj_bias),  # type: ignore[union-attr]
            "bias_k": self._handle(plan, source.bias_k),  # type: ignore[union-attr]
            "bias_v": self._handle(plan, source.bias_v),  # type: ignore[union-attr]
            "out_proj.weight": self._handle(plan, source.out_proj.weight),  # type: ignore[union-attr]
            "out_proj.bias": self._handle(plan, source.out_proj.bias),  # type: ignore[union-attr]
        }
        return CPSBackedMultiheadAttention(self, handles, source)  # type: ignore[arg-type]

    @staticmethod
    def _parent_and_name(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
        if not path:
            raise ValueError("the root module itself cannot be replaced in place")
        if "." not in path:
            return root, path
        parent_path, name = path.rsplit(".", 1)
        return root.get_submodule(parent_path), name

    def commit(
        self,
        proposal: Optional[ProposalMetadata | nn.Module] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        allow_compression: bool = False,
    ) -> CommitMetadata:
        """Install staged wrappers.

        Exact central slabs are the default first commit. Compression is a
        separate post-commit transaction unless ``allow_compression`` is
        explicitly requested by a low-level caller.
        """

        with self._lock:
            if optimizer is not None:
                raise ValueError(
                    "optimizer migration is not implemented; commit before optimizer construction"
                )
            if isinstance(proposal, nn.Module):
                proposal = self.stage(proposal)
            if self._commit is not None and self._commit.committed and not self._commit.rolled_back:
                return self._commit
            if self._staged is None:
                raise RuntimeError("stage a model before commit")
            plan = self._staged
            if proposal is not None and proposal.proposal_id != plan.proposal.proposal_id:
                raise ValueError("proposal does not belong to this CPS")
            root = plan.root_ref()
            if root is None:
                raise RuntimeError("staged root no longer exists")
            if any(not candidate.path for candidate in plan.candidates):
                raise ValueError("cannot replace an eligible root module in place")
            evaluation = self._evaluation or self._analyze_plan(plan)
            if evaluation.max_reconstruction_error > self.config.reconstruction_tolerance:
                raise RuntimeError("consolidation reconstruction tolerance failed")
            if evaluation.max_output_error > self.config.output_tolerance:
                raise RuntimeError("consolidation output tolerance failed")
            existing = getattr(root, self._ATTACHMENT_NAME, None)
            if existing is not None and existing is not self:
                raise RuntimeError("root already owns another TrainableParameterCPS")

            self._build_storage(plan, allow_compression=allow_compression)
            originals: Dict[str, nn.Module] = {}
            wrappers_by_module_id: Dict[int, nn.Module] = {}
            candidate_by_path = {item.path: item for item in plan.candidates}
            try:
                root.add_module(self._ATTACHMENT_NAME, self)
                for path in plan.proposal.replacements:
                    parent, name = self._parent_and_name(root, path)
                    original = getattr(parent, name)
                    candidate = candidate_by_path[path]
                    source_id = id(candidate.module)
                    wrapper = wrappers_by_module_id.get(source_id)
                    if wrapper is None:
                        wrapper = self._make_wrapper(plan, candidate)
                        wrappers_by_module_id[source_id] = wrapper
                    originals[path] = original
                    setattr(parent, name, wrapper)
            except Exception:
                for path, original in reversed(list(originals.items())):
                    parent, name = self._parent_and_name(root, path)
                    setattr(parent, name, original)
                if getattr(root, self._ATTACHMENT_NAME, None) is self:
                    delattr(root, self._ATTACHMENT_NAME)
                self.slabs = nn.ParameterList()
                self._registry.clear()
                self._representations.clear()
                raise

            object.__setattr__(self, "_original_modules", originals)
            object.__setattr__(self, "_root_ref", weakref.ref(root))
            self._commit = CommitMetadata(
                transaction_id=uuid.uuid4().hex,
                proposal_id=plan.proposal.proposal_id,
                replacements=plan.proposal.replacements,
                parameter_handles=tuple(self._registry),
            )
            return self._commit

    def compress_committed(
        self,
        *,
        probe=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> EvaluationMetadata:
        """Replace exact slabs with validated shared-template/low-rank cohorts.

        ``probe`` may be a callable accepting the committed root and returning a
        tensor (or nested tensors). If validation fails, exact slabs are rebuilt
        immediately and remain active.
        """
        with self._lock:
            if optimizer is not None:
                raise ValueError(
                    "compression changes parameter ownership; rebuild the optimizer"
                )
            if not self.config.enable_compression:
                raise RuntimeError("compression is disabled in TrainableParameterCPSConfig")
            if self._commit is None or not self._commit.committed or self._commit.rolled_back:
                raise RuntimeError("commit exact CPS ownership before compression")
            if self._staged is None:
                raise RuntimeError("committed CPS has no staged provenance")
            root_ref = self.__dict__.get("_root_ref")
            root = None if root_ref is None else root_ref()
            if root is None:
                raise RuntimeError("committed root no longer exists")

            def flatten_output(value):
                if torch.is_tensor(value):
                    return [value.detach().clone()]
                if isinstance(value, (tuple, list)):
                    out = []
                    for item in value:
                        out.extend(flatten_output(item))
                    return out
                if isinstance(value, dict):
                    out = []
                    for key in sorted(value):
                        out.extend(flatten_output(value[key]))
                    return out
                return []

            before = flatten_output(probe(root)) if callable(probe) else []
            values = {
                handle: nn.Parameter(self.materialize(handle).detach().clone())
                for handle in self._representations
            }
            original_plan = self._staged
            compression_plan = _StagedPlan(
                root_ref=original_plan.root_ref,
                proposal=original_plan.proposal,
                candidates=original_plan.candidates,
                handle_by_parameter_id=original_plan.handle_by_parameter_id,
                parameter_by_handle=values,
            )
            candidate_eval = self._analyze_plan(compression_plan)
            if not candidate_eval.compression_applied:
                self._evaluation = candidate_eval
                return candidate_eval

            self._build_storage(compression_plan, allow_compression=True)
            after = flatten_output(probe(root)) if callable(probe) else []
            output_error = 0.0
            if before or after:
                if len(before) != len(after):
                    output_error = float("inf")
                else:
                    for lhs, rhs in zip(before, after):
                        if lhs.shape != rhs.shape:
                            output_error = float("inf")
                            break
                        if lhs.numel():
                            output_error = max(
                                output_error,
                                float((lhs - rhs).abs().max().item()),
                            )
            if output_error > float(self.config.output_tolerance):
                self._build_storage(compression_plan, allow_compression=False)
                self._evaluation = replace(
                    candidate_eval,
                    proposed_scalars=sum(p.numel() for p in self.slabs),
                    scalar_savings=0,
                    max_output_error=output_error,
                    compression_applied=False,
                )
                return self._evaluation
            self._evaluation = replace(
                self._evaluation or candidate_eval,
                max_output_error=output_error,
                compression_applied=True,
            )
            return self._evaluation

    def propose_compression(self) -> EvaluationMetadata:
        """Analyze optional compression without changing committed storage."""
        if self._staged is None:
            raise RuntimeError("stage a model before proposing compression")
        values = (
            {
                handle: nn.Parameter(self.materialize(handle).detach().clone())
                for handle in self._representations
            }
            if self._representations
            else self._staged.parameter_by_handle
        )
        plan = _StagedPlan(
            root_ref=self._staged.root_ref,
            proposal=self._staged.proposal,
            candidates=self._staged.candidates,
            handle_by_parameter_id=self._staged.handle_by_parameter_id,
            parameter_by_handle=values,
        )
        return self._analyze_plan(plan)

    evaluate_compression = propose_compression

    def rollback(
        self,
        commit_id: Optional[str] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[CommitMetadata]:
        """Restore the exact original modules; repeated calls are harmless."""

        with self._lock:
            if optimizer is not None:
                raise ValueError(
                    "rollback changes parameter ownership; rebuild the optimizer"
                )
            if self._commit is None:
                return None
            if commit_id is not None and commit_id != self._commit.transaction_id:
                raise ValueError("rollback transaction id does not match active commit")
            if self._commit.rolled_back:
                return self._commit
            root_ref = self.__dict__.get("_root_ref")
            root = None if root_ref is None else root_ref()
            if root is None:
                raise RuntimeError("committed root no longer exists")
            originals: Dict[str, nn.Module] = self.__dict__["_original_modules"]
            # Preserve all training performed after consolidation by copying the
            # current canonical materializations back into original Parameters.
            if self._staged is not None:
                with torch.no_grad():
                    for candidate in self._staged.candidates:
                        for parameter in candidate.parameters.values():
                            handle = self._staged.handle_by_parameter_id.get(id(parameter))
                            if handle in self._representations:
                                parameter.copy_(
                                    self.materialize(handle).to(
                                        device=parameter.device,
                                        dtype=parameter.dtype,
                                    )
                                )
            for path, original in reversed(list(originals.items())):
                parent, name = self._parent_and_name(root, path)
                setattr(parent, name, original)
            if getattr(root, self._ATTACHMENT_NAME, None) is self:
                delattr(root, self._ATTACHMENT_NAME)
            self._commit = replace(self._commit, committed=False, rolled_back=True)
            object.__setattr__(self, "_original_modules", {})
            object.__setattr__(self, "_root_ref", None)
            self.slabs = nn.ParameterList()
            self._registry.clear()
            self._representations.clear()
            return self._commit

    def capacity_report(self) -> Dict[str, Any]:
        """Report literal tensor scalar counts; no theoretical multipliers."""

        if self._staged is None:
            original = 0
            aliases = 0
        else:
            original = sum(
                parameter.numel() for parameter in self._staged.parameter_by_handle.values()
            )
            aliases = sum(
                max(0, len(ref.aliases) - 1) for ref in self._staged.proposal.references
            )
        stored = sum(parameter.numel() for parameter in self.slabs)
        original_eligible = (
            sum(ref.numel * max(1, len(ref.aliases)) for ref in self._staged.proposal.references)
            if self._staged is not None
            else 0
        )
        if not self._representations and self._evaluation is not None:
            stored = self._evaluation.proposed_scalars
        literal_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.slabs
        )
        bytes_by_dtype: Dict[str, int] = {}
        for parameter in self.slabs:
            key = str(parameter.dtype)
            bytes_by_dtype[key] = bytes_by_dtype.get(key, 0) + (
                parameter.numel() * parameter.element_size()
            )
        return {
            "original_scalar_count": original,
            "stored_scalar_count": stored,
            "scalar_savings": original - stored,
            "compression_ratio": (stored / original) if original else 1.0,
            "original_eligible_scalars": original_eligible,
            "unique_scalars_after_sharing": original,
            "literal_trainable_scalars": stored,
            "compressed_scalars": stored,
            "real_compression_ratio": (
                original_eligible / stored if stored else 1.0
            ),
            "literal_bytes": literal_bytes,
            "bytes_by_dtype": bytes_by_dtype,
            "estimated_adam_training_bytes": literal_bytes * 4,
            "unique_parameter_handles": len(
                self._registry
                or (() if self._staged is None else self._staged.parameter_by_handle)
            ),
            "tied_alias_count": aliases,
            "slab_count": len(self.slabs),
            "rejected_modules": (
                dict(self._staged.proposal.rejected)
                if self._staged is not None
                else {}
            ),
            "ownership": {
                handle: {
                    "aliases": list(ref.aliases),
                    "storage_kind": ref.storage_kind,
                    "slab_index": ref.slab_index,
                    "offset": ref.offset,
                    "shape": list(ref.shape),
                    "rank": ref.rank,
                }
                for handle, ref in self._registry.items()
            },
            "cohorts": [
                asdict(cohort)
                for cohort in (self.group() if self._staged is not None else ())
            ],
            "literal": True,
        }

    def to_manifest(self) -> Dict[str, Any]:
        """Return JSON-serializable transaction metadata (tensor data is in state_dict)."""

        return {
            "version": 1,
            "config": asdict(self.config),
            "proposal": None if self.proposal is None else asdict(self.proposal),
            "evaluation": None if self._evaluation is None else asdict(self._evaluation),
            "commit": None if self._commit is None else asdict(self._commit),
            "registry": {handle: asdict(ref) for handle, ref in self._registry.items()},
            "representations": {
                handle: asdict(representation)
                for handle, representation in self._representations.items()
            },
            "slab_shapes": [list(parameter.shape) for parameter in self.slabs],
            "slab_dtypes": [str(parameter.dtype) for parameter in self.slabs],
            "capacity": self.capacity_report(),
        }

    manifest = to_manifest

    def load_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Validate manifest compatibility; tensor restoration uses ``load_state_dict``."""

        if manifest.get("version") != 1:
            raise ValueError("unsupported CPS manifest version")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("manifest is missing config metadata")
        normalized = dict(config)
        normalized["include"] = tuple(normalized.get("include", ("*",)))
        normalized["exclude"] = tuple(normalized.get("exclude", ()))
        if TrainableParameterCPSConfig(**normalized) != self.config:
            raise ValueError("manifest config does not match this CPS")

    def prepare_from_manifest(
        self,
        root: nn.Module,
        manifest: Mapping[str, Any],
    ) -> CommitMetadata:
        """Rebuild adapters/storage layout before loading checkpoint tensors."""
        self.load_manifest(manifest)
        proposal = self.stage(root)
        saved_proposal = manifest.get("proposal") or {}
        saved_replacements = tuple(saved_proposal.get("replacements", ()))
        if saved_replacements and tuple(proposal.replacements) != saved_replacements:
            raise ValueError("checkpoint CPS binding paths do not match this model")
        commit = self.commit(proposal)

        saved_shapes = manifest.get("slab_shapes") or []
        saved_dtypes = manifest.get("slab_dtypes") or []
        saved_representations = manifest.get("representations") or {}
        if saved_shapes and saved_representations:
            reference = next(self.parameters(), None)
            device = reference.device if reference is not None else torch.device("cpu")
            dtype = reference.dtype if reference is not None else torch.float32
            self.slabs = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            tuple(int(v) for v in shape),
                            device=device,
                            dtype=(
                                getattr(
                                    torch,
                                    str(saved_dtypes[index]).replace("torch.", ""),
                                    dtype,
                                )
                                if index < len(saved_dtypes)
                                else dtype
                            ),
                        )
                    )
                    for index, shape in enumerate(saved_shapes)
                ]
            )
            self._representations = {
                str(handle): _Representation(**dict(payload))
                for handle, payload in saved_representations.items()
            }
            registry = {}
            for handle, payload in (manifest.get("registry") or {}).items():
                data = dict(payload)
                data["shape"] = tuple(data.get("shape", ()))
                data["aliases"] = tuple(data.get("aliases", ()))
                registry[str(handle)] = ParameterRefMetadata(**data)
            self._registry = registry
        return commit


__all__ = [
    "CPSBackedEmbedding",
    "CPSBackedLinear",
    "CPSBackedMultiheadAttention",
    "CommitMetadata",
    "ConsolidationCommit",
    "ConsolidationEvaluation",
    "ConsolidationProposal",
    "EvaluationMetadata",
    "ParameterCohort",
    "ParameterRefMetadata",
    "ProposalMetadata",
    "RollbackRecord",
    "TrainableParameterCPS",
    "TrainableParameterCPSConfig",
    "TrainableParameterRef",
]
