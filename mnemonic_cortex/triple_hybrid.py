import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence, Tuple

from .hybrid_router_v2 import HybridRouterV2
from .lightbulb_event_logger import LightbulbEventLogger
from .ltm_aux_memory import ConsolidatedLTMBank, NeuralFieldMemory
from .memory_attention import MultiScaleAttention
from .memory_consolidation_manager_v2 import MemoryConsolidationManagerV2
from .memory_transformer_v2 import (
    EnhancedCGMNMemoryWithTransformerV2,
    EnhancedCurvedMemoryWithTransformerV2,
    EnhancedHyperGeometricMemoryWithTransformerV2,
)
from .topology_manager import TopologyManagerV3

logger = logging.getLogger(__name__)


class EnhancedTripleHybridMemory(nn.Module):
    """Wrapper combining HG, CGMN, and Curved memories with selectable fusion:
       - 'weighted' (default): learned static mix
       - 'cross_attn': cross-attention over {HG,CGMN,Curved} per timestep
    Also propagates temperature and energy-efficient mode; exposes consolidation.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hg_dim: int = 24,
        hg_slots: int = 1028,
        hg_qubits: int = 8,
        cgmn_dim: int = 16,
        cgmn_slots: int = 512,
        cgmn_slot_dim: int = 256,
        curved_hidden: int = 256,
        curved_curvature: int = 8,
        curved_slots: int = 128,
        n_transformer_layers: int = 3,
        n_heads: int = 8,
        attention_type: str = "multiscale",
        fusion: str = "weighted",
        enable_hg_bank: bool = True,
        hg_bank_size: int = 1024,
        hg_use_entanglement: bool = True,
        curved_use_tcn: bool = True,
        hg_ann_centroids: int = 256,
        hg_ann_top: int = 8,
        hg_transformer_layers: int = 0,
        cgmn_transformer_layers: int = 0,
        curved_transformer_layers: int = 0,
        fusion_transformer_layers: int = 0,
        cross_model_attention_layers: int = 0,
        prefusion_specialization_layers: int = 0,
        hg_transformer_heads: int = 0,
        cgmn_transformer_heads: int = 0,
        curved_transformer_heads: int = 0,
        fusion_transformer_heads: int = 0,
        cross_model_attention_heads: int = 0,
        enable_hns_fusion: bool = True,
        topological_dim: int = 128,
        consolidated_slots: int = 512,
        nfm_slots: int = 64,
        consolidation_interval: int = 10,
        dynamic_consolidation: bool = True,
        consolidation_threshold: float = 0.6,
        hns_fusion_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim % int(n_heads) != 0:
            raise AssertionError(
                f"d_model {self.input_dim} must be divisible by num_heads {int(n_heads)}"
            )
        self.hg_dim = int(hg_dim)
        self.hg_slots = int(hg_slots)
        self.hg_qubits = int(hg_qubits)
        self.cgmn_dim = int(cgmn_dim)
        self.cgmn_slots = int(cgmn_slots)
        self.cgmn_slot_dim = int(cgmn_slot_dim)
        self.curved_hidden_dim = int(curved_hidden)
        self.curved_curvature_dim = int(curved_curvature)
        self.curved_slots = int(curved_slots)
        self.n_transformer_layers = int(max(1, n_transformer_layers))
        self.n_heads = int(max(1, n_heads))
        self.attention_type = str(attention_type).strip().lower()
        if self.attention_type not in {"multiscale", "standard"}:
            raise ValueError("attention_type must be 'multiscale' or 'standard'")

        bank_layers = self.n_transformer_layers
        hg_layers = int(hg_transformer_layers) if int(hg_transformer_layers) > 0 else bank_layers
        cg_layers = int(cgmn_transformer_layers) if int(cgmn_transformer_layers) > 0 else bank_layers
        cv_layers = int(curved_transformer_layers) if int(curved_transformer_layers) > 0 else bank_layers
        fusion_layers = int(fusion_transformer_layers) if int(fusion_transformer_layers) > 0 else max(4, bank_layers + 1)
        cross_layers = int(cross_model_attention_layers) if int(cross_model_attention_layers) > 0 else max(4, bank_layers + 1)
        prefusion_layers = int(prefusion_specialization_layers) if int(prefusion_specialization_layers) > 0 else max(2, bank_layers - 1)

        fusion_heads = self._resolve_heads(self.input_dim, self.n_heads, fusion_transformer_heads)
        cross_model_heads = self._resolve_heads(self.input_dim, self.n_heads, cross_model_attention_heads)
        hg_heads = self._resolve_heads(self.input_dim, self.n_heads, hg_transformer_heads)
        cg_heads = self._resolve_heads(self.input_dim, self.n_heads, cgmn_transformer_heads)
        cv_heads = self._resolve_heads(self.input_dim, self.n_heads, curved_transformer_heads)

        self.hyper_geometric = EnhancedHyperGeometricMemoryWithTransformerV2(
            input_dim,
            self.hg_dim,
            self.hg_slots,
            self.hg_qubits,
            hg_layers,
            hg_heads,
            self.attention_type,
            enable_bank=bool(enable_hg_bank),
            bank_size=int(hg_bank_size),
            use_entanglement=bool(hg_use_entanglement),
            ann_centroids=hg_ann_centroids,
            ann_top=hg_ann_top,
        )
        self.cgmn = EnhancedCGMNMemoryWithTransformerV2(
            input_dim,
            self.cgmn_dim,
            self.cgmn_slots,
            self.cgmn_slot_dim,
            cg_layers,
            cg_heads,
            self.attention_type,
        )
        self.curved = EnhancedCurvedMemoryWithTransformerV2(
            input_dim,
            self.curved_hidden_dim,
            self.curved_curvature_dim,
            self.curved_slots,
            cv_layers,
            cv_heads,
            self.attention_type,
            use_tcn=bool(curved_use_tcn),
        )
        self.hg = self.hyper_geometric
        self.topology_manager = TopologyManagerV3(subsystems=("hg", "cgmn", "curved"))
        self.mix = nn.Parameter(torch.tensor([0.34, 0.33, 0.33]))  # [hg, cgmn, curved]
        self.enable_hns_fusion = bool(enable_hns_fusion)
        self.n_routed_tokens = 5
        self.n_total_tokens_for_fusion = 6
        self.fusion_in_dim = self.input_dim * self.n_total_tokens_for_fusion
        self.consolidation_interval = int(max(1, consolidation_interval))
        self.dynamic_consolidation = bool(dynamic_consolidation)
        self.consolidation_threshold = float(consolidation_threshold)
        self.consolidation_counter = 0
        self.lightbulb_intensity = 0.0
        self.cons_novelty = 0.0
        self.lightbulb_decay = 0.9
        self.lightbulb_logger = LightbulbEventLogger()
        self.register_buffer("consolidated_usage", torch.zeros(int(consolidated_slots)))
        self.router = nn.Sequential(
            nn.Linear(input_dim + 12, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )
        self.hns_router = HybridRouterV2(
            input_dim=self.input_dim,
            n_subsystems=self.n_routed_tokens,
            n_heads=self.n_heads,
        )
        self.cross_memory_attention = nn.MultiheadAttention(
            self.input_dim, num_heads=fusion_heads, batch_first=True
        )
        fusion_stack_heads = self._resolve_heads(self.fusion_in_dim, self.n_heads, fusion_transformer_heads)
        self.memory_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.fusion_in_dim,
                fusion_stack_heads,
                dim_feedforward=max(128, self.fusion_in_dim),
                batch_first=True,
                activation="gelu",
            ),
            num_layers=max(1, int(hns_fusion_layers)),
        )
        self.fusion_projection = nn.Linear(self.fusion_in_dim, self.output_dim)
        self.hns_to_seq = (
            nn.Linear(self.output_dim, self.input_dim)
            if self.output_dim != self.input_dim
            else nn.Identity()
        )
        self.hns_blend_gate = nn.Parameter(torch.tensor(0.35))
        self.hns_blend_norm = nn.LayerNorm(self.input_dim)
        self.importance_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.input_dim * 2,
                fusion_heads,
                dim_feedforward=max(128, self.input_dim * 2),
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.importance_head = nn.Sequential(
            nn.Linear(self.input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.consolidation_gates = nn.Sequential(
            nn.Linear(self.input_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        self.consolidation_network = nn.Sequential(
            nn.Linear(self.input_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
        )
        self.consolidated_memory = ConsolidatedLTMBank(
            self.input_dim, slots=int(consolidated_slots), n_heads=self.n_heads
        )
        self.nfm = NeuralFieldMemory(self.input_dim, slots=int(nfm_slots))
        self.topo_consolidator = MemoryConsolidationManagerV2(
            embedding_dim=self.input_dim,
            topological_dim=int(topological_dim),
        )
        self.topo_to_input = nn.Linear(int(topological_dim), self.input_dim)
        self.fusion_mode = fusion
        self.cross_fuser = self._make_attention(fusion_heads)
        self.hg_to_cg_attn = self._make_attention(fusion_heads)
        self.cg_to_spatial_attn = self._make_attention(fusion_heads)
        self.spatial_to_hg_attn = self._make_attention(fusion_heads)
        self.inter_norm_hg = nn.LayerNorm(input_dim)
        self.inter_norm_cg = nn.LayerNorm(input_dim)
        self.inter_norm_spatial = nn.LayerNorm(input_dim)
        self.global_inter_attn = self._make_attention(fusion_heads)
        self.inter_exchange_gate = nn.Parameter(torch.tensor(0.20))
        self.last_inter_memory_stats = {}
        self.last_router_weights = None
        self.last_router_stats = {}
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=fusion_heads,
                dim_feedforward=max(128, input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, fusion_layers),
        )
        self.refiner_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))
        self.external_attention_context = None
        self.qdt_to_hg_attn = self._make_attention(cross_model_heads)
        self.qdt_to_cg_attn = self._make_attention(cross_model_heads)
        self.qdt_to_cv_attn = self._make_attention(cross_model_heads)
        self.cross_model_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=cross_model_heads,
                dim_feedforward=max(128, input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, cross_layers),
        )
        self.cross_model_norm = nn.LayerNorm(input_dim)
        self.prefusion_compare_attn = self._make_attention(fusion_heads)
        self.prefusion_compare_norm = nn.LayerNorm(input_dim)
        self.prefusion_compare_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=fusion_heads,
                dim_feedforward=max(128, input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, prefusion_layers),
        )
        self.prefusion_specialization_head = nn.Linear(input_dim, 1)
        self.prefusion_peer_attn = self._make_attention(fusion_heads)
        self.prefusion_propagation_norm = nn.LayerNorm(input_dim)
        self.prefusion_propagation_gate = nn.Parameter(torch.tensor(0.24))
        self.last_prefusion_specialization_stats = {}

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    @classmethod
    def _resolve_heads(cls, dim: int, n_heads: int, explicit: int = 0) -> int:
        if int(explicit) > 0:
            heads = int(explicit)
        elif int(n_heads) > 0 and dim % int(n_heads) == 0:
            heads = int(n_heads)
        else:
            heads = cls._pick_num_heads(dim)
        if dim % heads != 0:
            raise ValueError(f"input_dim={dim} must be divisible by attention heads={heads}")
        return heads

    def _make_attention(self, num_heads: int) -> nn.Module:
        if self.attention_type == "multiscale":
            return MultiScaleAttention(self.input_dim, num_heads)
        return nn.MultiheadAttention(self.input_dim, num_heads=num_heads, batch_first=True)

    @staticmethod
    def _attn_weight_mean(weights) -> float:
        if weights is None:
            return 0.0
        return float(weights.detach().mean().item())

    def set_temperature(self, t: torch.Tensor):
        self.hg.set_temperature(t)
        self.cgmn.set_temperature(t)
        self.curved.set_temperature(t)

    def set_external_attention_context(self, context: torch.Tensor) -> None:
        if context is None:
            self.external_attention_context = None
            self.hg.clear_external_attention_context()
            self.cgmn.clear_external_attention_context()
            self.curved.clear_external_attention_context()
            return
        ctx = torch.as_tensor(context, device=self.mix.device, dtype=self.mix.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        elif ctx.dim() != 3:
            raise ValueError("external attention context must be [T,D], [B,D], or [B,S,D]")
        if ctx.size(-1) != self.hg.input_dim:
            raise ValueError(f"external attention context dim must be {self.hg.input_dim}")
        self.external_attention_context = ctx.detach()
        self.hg.set_external_attention_context(self.external_attention_context)
        self.cgmn.set_external_attention_context(self.external_attention_context)
        self.curved.set_external_attention_context(self.external_attention_context)

    def clear_external_attention_context(self) -> None:
        self.set_external_attention_context(None)

    def enable_energy_efficient_mode(self, enable: bool = True):
        self.hg.enable_energy_efficient_mode(enable)
        self.cgmn.enable_energy_efficient_mode(enable)
        self.curved.enable_energy_efficient_mode(enable)

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1):
        self.hg.consolidate_unused(threshold)
        self.cgmn.consolidate_unused(threshold)
        self.curved.consolidate_unused(threshold)

    @torch.no_grad()
    def evolve_topologies(self, fitness_by_subsystem):
        """Update topology states using subsystem fitness values.
        fitness_by_subsystem keys: 'hg' | 'cgmn' | 'curved'
        """
        out = {}
        for name in ("hg", "cgmn", "curved"):
            if name in fitness_by_subsystem:
                out[name] = self.topology_manager.evolve_topology(
                    fitness=float(fitness_by_subsystem[name]),
                    subsystem=name,
                )
                # Apply a small channel-prior nudge to each merger.
                if name == "hg":
                    self.topology_manager.steer_merger(self.hg.geometry_merger, subsystem="hg")
                elif name == "cgmn":
                    self.topology_manager.steer_merger(self.cgmn.geometry_merger, subsystem="cgmn")
                else:
                    self.topology_manager.steer_merger(self.curved.geometry_merger, subsystem="curved")
        return out

    @torch.no_grad()
    def mutate_curvatures(self, loss_value: float):
        """Apply loss-aware curvature mutation via TopologyManagerV3."""
        self.hg.memory_curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.hg.memory_curvature, loss_value=loss_value, subsystem="hg"
            )
        )
        self.cgmn.curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.cgmn.curvature, loss_value=loss_value, subsystem="cgmn"
            )
        )
        self.curved.memory_curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.curved.memory_curvature,
                loss_value=loss_value,
                subsystem="curved",
            )
        )
        # Mutate SPD factors with the same topology mode, gently.
        self.hg.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.hg.spd_L, subsystem="hg"))
        self.cgmn.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.cgmn.spd_L, subsystem="cgmn"))
        self.curved.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.curved.spd_L, subsystem="curved"))

    @torch.no_grad()
    def step_topology(self, loss_value: float):
        """
        Per-step topology hook for memory-local geometry controllers.
        """
        if hasattr(self.hg, "step_topology"):
            self.hg.step_topology(loss_value)
        if hasattr(self.cgmn, "step_topology"):
            self.cgmn.step_topology(loss_value)
        if hasattr(self.curved, "step_topology"):
            self.curved.step_topology(loss_value)

    @staticmethod
    def _seq_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _estimate_importance(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand_as(x)
        contextual = torch.cat([x, context], dim=-1)
        if contextual.size(-1) != self.input_dim * 2:
            contextual = torch.cat([x, x], dim=-1)
        try:
            imp_feat = self.importance_predictor(contextual).mean(dim=1)
        except Exception:
            imp_feat = contextual.mean(dim=1)
        if imp_feat.size(-1) == self.input_dim * 2:
            return self.importance_head(imp_feat)
        return torch.sigmoid(imp_feat.mean(dim=-1, keepdim=True))

    def detect_cross_memory_lightbulb(
        self,
        hg_nov: torch.Tensor,
        cg_sim: torch.Tensor,
        cu_spread: torch.Tensor,
    ) -> Tuple[bool, torch.Tensor]:
        hg_norm = torch.sigmoid(hg_nov * 5)
        cg_norm = torch.sigmoid(cg_sim * 5)
        cu_norm = torch.sigmoid(cu_spread * 5)
        combined = (hg_norm + (1.0 - cg_norm) + cu_norm) / 3.0
        return bool((combined.mean() > 0.6).item()), combined.mean()

    def coordinate_lightbulb_moments(
        self,
        hg_query: torch.Tensor,
        cgmn_positions: torch.Tensor,
        curved_activation: torch.Tensor,
    ) -> float:
        hg_light, hg_nov = self.hyper_geometric.detect_lightbulb_moment(hg_query)
        cg_ins, cg_sim = self.cgmn.detect_geometric_insight(cgmn_positions)
        cu_chain, cu_spread = self.curved.detect_associative_chain(curved_activation)
        is_cross, intensity = self.detect_cross_memory_lightbulb(
            hg_nov.mean(), cg_sim.mean(), cu_spread.mean()
        )
        if is_cross or hg_light or cg_ins or cu_chain:
            new_intensity = max(float(self.lightbulb_intensity), float(intensity.item()))
            self.lightbulb_intensity = 0.9 * self.lightbulb_intensity + 0.1 * new_intensity
            self.lightbulb_logger.log(
                source="cross-memory",
                intensity=float(self.lightbulb_intensity),
            )
        else:
            self.lightbulb_intensity *= self.lightbulb_decay
        return float(self.lightbulb_intensity)

    def _detect_consolidated_lightbulb(
        self,
        query: torch.Tensor,
        consolidated_output: torch.Tensor,
    ) -> Tuple[bool, float]:
        if consolidated_output is None or (
            isinstance(consolidated_output, torch.Tensor) and consolidated_output.norm() == 0
        ):
            return False, 0.0
        q = query.mean(dim=1) if query.dim() == 3 else query
        dif = consolidated_output.mean(dim=1) - q
        consolidated_novelty = torch.sigmoid(dif.norm(dim=-1) * 5).mean().item()
        trig = consolidated_novelty > 0.7
        if trig:
            self.lightbulb_logger.log(
                source="consolidated",
                intensity=float(consolidated_novelty),
            )
        return trig, float(consolidated_novelty)

    def _cross_memory_integrate(
        self,
        hg_out: torch.Tensor,
        cgmn_out: torch.Tensor,
        curved_out: torch.Tensor,
        consolidated_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outs = [hg_out, cgmn_out, curved_out]
        if consolidated_out is not None:
            outs.append(consolidated_out)
        mem = torch.stack(outs, dim=1)
        pooled = mem.mean(dim=2)
        attn, _ = self.cross_memory_attention(pooled, pooled, pooled, need_weights=False)
        refined = []
        for i in range(attn.size(1)):
            token = attn[:, i : i + 1, :]
            sa, _ = self.cross_memory_attention(token, token, token, need_weights=False)
            refined.append(sa.squeeze(1))
        return torch.stack(
            [attn[:, i, :] + refined[i] for i in range(attn.size(1))],
            dim=1,
        )

    def _adaptive_consolidation(self, pooled: torch.Tensor, importance: torch.Tensor) -> None:
        bsz = pooled.size(0)
        gates = self.consolidation_gates(pooled.reshape(bsz, -1))
        consolidated = self.consolidation_network(
            (gates.unsqueeze(-1) * pooled).reshape(bsz, -1)
        )
        trigger = False
        if self.dynamic_consolidation and float(self.lightbulb_intensity) >= self.consolidation_threshold:
            trigger = True
        elif (self.consolidation_counter % self.consolidation_interval) == 0:
            trigger = True
        if trigger:
            _ = self.consolidated_memory(
                consolidated.unsqueeze(1),
                operation="write",
                lightbulb_intensity=float(self.lightbulb_intensity),
            )
            self.lightbulb_logger.log(
                source="consolidation",
                intensity=float(self.lightbulb_intensity),
                step=self.consolidation_counter,
            )
        self.consolidation_counter += 1

    def _retrieve_consolidated(
        self,
        query: torch.Tensor,
        lightbulb_intensity: float = 0.0,
    ) -> torch.Tensor:
        if self.consolidation_counter == 0:
            self.cons_novelty = 0.0
            return torch.zeros_like(query)
        out = self.consolidated_memory(
            query,
            operation="read",
            lightbulb_intensity=lightbulb_intensity,
        )
        _, nov = self._detect_consolidated_lightbulb(query, out)
        self.cons_novelty = nov
        return out

    def _apply_hns_fusion(
        self,
        x: torch.Tensor,
        rhg: torch.Tensor,
        rcg: torch.Tensor,
        rcv: torch.Tensor,
        base_fused: torch.Tensor,
        *,
        cross_intensity: float,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enable_hns_fusion:
            return base_fused

        bsz, seq, dim = x.shape
        consolidated_out = self._retrieve_consolidated(x, cross_intensity)
        integrated = self._cross_memory_integrate(rhg, rcg, rcv, consolidated_out)
        topo_recalled = self.topo_consolidator.recall(x, topk=1)
        if topo_recalled.size(1) == 0:
            topo_token = x.new_zeros(bsz, 1, dim)
        else:
            topo_token = self.topo_to_input(topo_recalled.squeeze(1)).unsqueeze(1)

        if integrated.size(1) == 4:
            tokens5 = torch.cat([integrated, topo_token], dim=1)
        else:
            tokens5 = torch.cat([integrated[:, :3, :], topo_token], dim=1)

        seq_ctx = x.mean(dim=1)
        weights, refined = self.hns_router(tokens5, context=seq_ctx)
        fused_tokens = refined * weights.unsqueeze(-1)
        routed_vec = fused_tokens.reshape(bsz, self.n_routed_tokens * dim)
        nfm_token = self.nfm.read_token(x)
        fused_vec = torch.cat([routed_vec, nfm_token], dim=-1)
        if fused_vec.shape[-1] != self.fusion_in_dim:
            if not self.training:
                logger.warning(
                    "Fusion dim mismatch: got %d expected %d",
                    fused_vec.shape[-1],
                    self.fusion_in_dim,
                )
            pad = self.fusion_in_dim - fused_vec.shape[-1]
            if pad > 0:
                fused_vec = F.pad(fused_vec, (0, pad))
            else:
                fused_vec = fused_vec[..., : self.fusion_in_dim]

        fused = self.memory_fusion(fused_vec.unsqueeze(1)).squeeze(1)
        hns_vec = self.fusion_projection(fused)
        hns_seq = self.hns_to_seq(hns_vec).unsqueeze(1).expand(-1, seq, -1)
        gate = torch.sigmoid(self.hns_blend_gate)
        self.lightbulb_logger.log(source="read-output", intensity=float(cross_intensity))
        return self.hns_blend_norm(base_fused + gate * hns_seq)

    def _fuse(self, rhg, rcg, rcv, routing_weights=None):
        if self.fusion_mode == 'cross_attn':
            B,S,D = rhg.shape
            tokens = torch.stack([rhg, rcg, rcv], dim=2)     # (B,S,3,D)
            tokens = tokens.view(B*S, 3, D)                  # (B*S,3,D)
            fused, _ = self.cross_fuser(tokens, tokens, tokens)  # (B*S,3,D)
            fused = fused.mean(dim=1).view(B,S,D)            # (B,S,D)
            refined = self.refiner(fused)
            return self.refiner_norm(fused + refined)
        else:
            if routing_weights is None:
                w = torch.softmax(self.mix, dim=0)
                fused = w[0] * rhg + w[1] * rcg + w[2] * rcv
            else:
                w = routing_weights.unsqueeze(1).unsqueeze(-1)  # (B,1,3,1)
                stacked = torch.stack([rhg, rcg, rcv], dim=2)   # (B,S,3,D)
                fused = (w * stacked).sum(dim=2)
            refined = self.refiner(fused)
            return self.refiner_norm(fused + refined)

    @staticmethod
    def _grab_router_features(mem, bsz: int, device, dtype):
        f = getattr(mem, "last_router_features", None)
        if not isinstance(f, dict):
            z = torch.zeros(bsz, device=device, dtype=dtype)
            return z, z, z, z
        vals = []
        for key in ("omega_mean", "curv_mean", "dist_mean", "entropy"):
            v = f.get(key, None)
            if isinstance(v, torch.Tensor):
                t = v.detach().to(device=device, dtype=dtype).reshape(-1)
                if t.numel() == 1:
                    t = t.expand(bsz)
                elif t.numel() != bsz:
                    t = t.mean().expand(bsz)
            else:
                t = torch.zeros(bsz, device=device, dtype=dtype)
            vals.append(t)
        return tuple(vals)

    def _inter_memory_exchange(self, rhg, rcg, rcv):
        """
        Cross-memory attention cycle:
          HG -> CGMN, CGMN -> Spatial, Spatial -> HG, then global tri-memory pass.
        """
        gate = torch.sigmoid(self.inter_exchange_gate)

        cg_from_hg, w_hg_to_cg = self.hg_to_cg_attn(rcg, rhg, rhg, need_weights=True)
        sp_from_cg, w_cg_to_sp = self.cg_to_spatial_attn(rcv, rcg, rcg, need_weights=True)
        hg_from_sp, w_sp_to_hg = self.spatial_to_hg_attn(rhg, rcv, rcv, need_weights=True)

        rhg_e = self.inter_norm_hg(rhg + gate * hg_from_sp)
        rcg_e = self.inter_norm_cg(rcg + gate * cg_from_hg)
        rcv_e = self.inter_norm_spatial(rcv + gate * sp_from_cg)

        bsz, seq, dim = rhg.shape
        tri = torch.stack([rhg_e, rcg_e, rcv_e], dim=2).reshape(bsz * seq, 3, dim)
        tri_global, w_global = self.global_inter_attn(tri, tri, tri, need_weights=True)
        tri = tri + gate * tri_global
        tri = tri.reshape(bsz, seq, 3, dim)

        rhg_o = tri[:, :, 0, :]
        rcg_o = tri[:, :, 1, :]
        rcv_o = tri[:, :, 2, :]
        self.last_inter_memory_stats = {
            "gate": float(gate.detach().item()),
            "hg_to_cg_mean": self._attn_weight_mean(w_hg_to_cg),
            "cg_to_spatial_mean": self._attn_weight_mean(w_cg_to_sp),
            "spatial_to_hg_mean": self._attn_weight_mean(w_sp_to_hg),
            "global_mean": self._attn_weight_mean(w_global),
        }
        return rhg_o, rcg_o, rcv_o

    def _prefusion_specialization_exchange(self, rhg, rcg, rcv, routing_weights=None):
        """
        Observe pre-fusion bank readouts with multi-head attention, score per-system
        specialization, and propagate cooperative advantages back into each system
        prior to inter-memory exchange and final fusion.
        """
        bsz, seq, dim = rhg.shape
        systems = torch.stack([rhg, rcg, rcv], dim=2)  # (B,S,3,D)
        flat = systems.reshape(bsz * seq, 3, dim)

        compared, compare_w = self.prefusion_compare_attn(flat, flat, flat, need_weights=True)
        cooperative = self.prefusion_compare_norm(flat + compared)
        cooperative = self.prefusion_compare_norm(
            cooperative + self.prefusion_compare_encoder(cooperative)
        )

        spec_logits = self.prefusion_specialization_head(cooperative).squeeze(-1)  # (B*S,3)
        spec_weights = torch.softmax(spec_logits, dim=-1)

        if routing_weights is not None:
            route = routing_weights.unsqueeze(1).expand(-1, seq, -1).reshape(bsz * seq, 3)
            blend = 0.5 * spec_weights + 0.5 * route
            spec_weights = blend / blend.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        gate = torch.sigmoid(self.prefusion_propagation_gate)
        specialized_pool = (spec_weights.unsqueeze(-1) * cooperative).sum(dim=1, keepdim=True)
        peer_views = []
        for idx in range(3):
            mask = torch.ones(3, device=flat.device, dtype=torch.bool)
            mask[idx] = False
            peer = cooperative[:, mask, :]
            peer_w = spec_weights[:, mask]
            peer_w = peer_w / peer_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            peer_mix = (peer_w.unsqueeze(-1) * peer).sum(dim=1, keepdim=True)
            peer_attn, _ = self.prefusion_peer_attn(
                flat[:, idx : idx + 1, :],
                peer,
                peer,
                need_weights=False,
            )
            peer_views.append(peer_attn)
        peer_stack = torch.cat(peer_views, dim=1)

        enhanced = flat + gate * (cooperative - flat)
        enhanced = enhanced + gate * specialized_pool
        enhanced = enhanced + gate * peer_stack
        enhanced = self.prefusion_propagation_norm(enhanced)

        self.last_prefusion_specialization_stats = {
            "gate": float(gate.detach().item()),
            "compare_attn_mean": self._attn_weight_mean(compare_w),
            "hg_specialization_mean": float(spec_weights[:, 0].detach().mean().item()),
            "cgmn_specialization_mean": float(spec_weights[:, 1].detach().mean().item()),
            "curved_specialization_mean": float(spec_weights[:, 2].detach().mean().item()),
        }

        rhg_o = enhanced[:, 0, :].reshape(bsz, seq, dim)
        rcg_o = enhanced[:, 1, :].reshape(bsz, seq, dim)
        rcv_o = enhanced[:, 2, :].reshape(bsz, seq, dim)
        return rhg_o, rcg_o, rcv_o

    def _apply_external_attention_stack(self, rhg, rcg, rcv):
        if self.external_attention_context is None:
            return rhg, rcg, rcv
        ctx = self.external_attention_context.to(device=rhg.device, dtype=rhg.dtype)
        if ctx.size(0) == 1 and rhg.size(0) > 1:
            ctx = ctx.expand(rhg.size(0), -1, -1)
        elif ctx.size(0) != rhg.size(0):
            ctx = ctx.mean(dim=0, keepdim=True).expand(rhg.size(0), -1, -1)
        hg_ext, _ = self.qdt_to_hg_attn(rhg, ctx, ctx, need_weights=False)
        cg_ext, _ = self.qdt_to_cg_attn(rcg, ctx, ctx, need_weights=False)
        cv_ext, _ = self.qdt_to_cv_attn(rcv, ctx, ctx, need_weights=False)
        rhg = self.cross_model_norm(rhg + hg_ext)
        rcg = self.cross_model_norm(rcg + cg_ext)
        rcv = self.cross_model_norm(rcv + cv_ext)
        joined = torch.cat([rhg, rcg, rcv], dim=1)
        joined = self.cross_model_norm(joined + self.cross_model_stack(joined))
        s = rhg.size(1)
        return joined[:, :s, :], joined[:, s:2 * s, :], joined[:, 2 * s:, :]

    def _run_read_pipeline(
        self,
        x: torch.Tensor,
        *,
        fire_mask=None,
        recall_boost: float = 0.3,
        context: Optional[torch.Tensor] = None,
        cross_intensity: float = 0.0,
        apply_hns: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand_as(x)
        rhg = self.hg(x, operation="read", fire_mask=fire_mask, recall_boost=recall_boost)
        rcg = self.cgmn(x, operation="read", fire_mask=fire_mask, recall_boost=recall_boost)
        rcv = self.curved(x, operation="read")
        rhg, rcg, rcv = self._apply_external_attention_stack(rhg, rcg, rcv)
        bsz = x.size(0)
        device = x.device
        dtype = x.dtype
        hg_om, hg_cv, hg_dm, hg_en = self._grab_router_features(self.hg, bsz, device, dtype)
        cg_om, cg_cv, cg_dm, cg_en = self._grab_router_features(self.cgmn, bsz, device, dtype)
        cv_om, cv_cv, cv_dm, cv_en = self._grab_router_features(self.curved, bsz, device, dtype)
        router_feats = torch.stack(
            [
                hg_om, hg_cv, hg_dm, hg_en,
                cg_om, cg_cv, cg_dm, cg_en,
                cv_om, cv_cv, cv_dm, cv_en,
            ],
            dim=-1,
        )
        router_in = torch.cat([x.mean(dim=1), router_feats], dim=-1)
        routing_weights = self.router(router_in)
        self.last_router_weights = routing_weights.detach()
        self.last_router_stats = {
            "hg": float(routing_weights[:, 0].detach().mean().item()),
            "cgmn": float(routing_weights[:, 1].detach().mean().item()),
            "curved": float(routing_weights[:, 2].detach().mean().item()),
            "lightbulb_intensity": float(cross_intensity),
            "cons_novelty": float(self.cons_novelty),
        }
        rhg, rcg, rcv = self._prefusion_specialization_exchange(
            rhg, rcg, rcv, routing_weights=routing_weights
        )
        rhg, rcg, rcv = self._inter_memory_exchange(rhg, rcg, rcv)
        base_fused = self._fuse(rhg, rcg, rcv, routing_weights=routing_weights)
        fused = (
            self._apply_hns_fusion(
                x,
                rhg,
                rcg,
                rcv,
                base_fused,
                cross_intensity=cross_intensity,
                context=context,
            )
            if apply_hns
            else base_fused
        )
        result = {"hg": rhg, "cgmn": rcg, "curved": rcv, "fused": fused}
        self._read_pipeline_cache = {
            "x_id": id(x),
            "fire_mask_id": None if fire_mask is None else id(fire_mask),
            "recall_boost": float(recall_boost),
            "reads": result,
        }
        return result

    def read_banks(
        self,
        x: torch.Tensor,
        *,
        fire_mask=None,
        recall_boost: float = 0.3,
        context: Optional[torch.Tensor] = None,
        include_fused: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Read HG/CGMN/Curved through the same fusion pipeline as forward(read)."""
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand_as(x)
        cross_intensity = 0.0
        hg_query = self.hyper_geometric.encode_to_manifold(x).mean(dim=2)
        cgmn_proj_flat = self.cgmn.manifold_projection(x)
        if cgmn_proj_flat.numel() == 0:
            cgmn_positions = x.new_zeros(x.size(0), x.size(1), 1, 3)
        else:
            bc, tc, k3 = cgmn_proj_flat.shape
            m = max(1, k3 // 3)
            usable = m * 3
            if usable != k3:
                cgmn_proj_flat = cgmn_proj_flat[..., :usable]
            cgmn_positions = cgmn_proj_flat.view(bc, tc, m, 3)
        curved_encoded = self.curved.encoder(x)
        curved_activation = F.relu(curved_encoded.mean(dim=1))
        cross_intensity = self.coordinate_lightbulb_moments(
            hg_query, cgmn_positions, curved_activation
        )
        out = self._run_read_pipeline(
            x,
            fire_mask=fire_mask,
            recall_boost=recall_boost,
            context=context,
            cross_intensity=cross_intensity,
            apply_hns=self.enable_hns_fusion and include_fused,
        )
        if not include_fused:
            out.pop("fused", None)
        return out

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        fire_mask=None,
        recall_boost: float = 0.3,
        context: Optional[torch.Tensor] = None,
    ):
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand_as(x)

        try:
            self._last_seq = x + self.hyper_geometric.pos_enc(x)
        except Exception:
            self._last_seq = x

        importance = self._estimate_importance(x, context)
        cross_intensity = 0.0
        if operation == "read":
            hg_query = self.hyper_geometric.encode_to_manifold(x).mean(dim=2)
            cgmn_proj_flat = self.cgmn.manifold_projection(x)
            if cgmn_proj_flat.numel() == 0:
                cgmn_positions = x.new_zeros(x.size(0), x.size(1), 1, 3)
            else:
                bc, tc, k3 = cgmn_proj_flat.shape
                m = max(1, k3 // 3)
                usable = m * 3
                if usable != k3:
                    cgmn_proj_flat = cgmn_proj_flat[..., :usable]
                cgmn_positions = cgmn_proj_flat.view(bc, tc, m, 3)
            curved_encoded = self.curved.encoder(x)
            curved_activation = F.relu(curved_encoded.mean(dim=1))
            cross_intensity = self.coordinate_lightbulb_moments(
                hg_query, cgmn_positions, curved_activation
            )

        if operation == "write":
            with torch.no_grad():
                self.nfm.ingest(x)
            self._last_seq = x
            hg_seq = self.hyper_geometric(
                x,
                operation="write",
                fire_mask=fire_mask,
                recall_boost=recall_boost,
            )
            try:
                self._last_seq = hg_seq
            except Exception:
                pass
            _ = self.cgmn(
                x,
                operation="write",
                fire_mask=fire_mask,
                recall_boost=recall_boost,
            )
            _ = self.curved(x, operation="write", importance=importance)
            hg_out_w = self._seq_pool(hg_seq)
            cg_out_w = self._seq_pool(
                self.cgmn(x, operation="read", fire_mask=fire_mask, recall_boost=recall_boost)
            )
            cv_out_w = self._seq_pool(self.curved(x, operation="read"))
            pooled = torch.stack([hg_out_w, cg_out_w, cv_out_w], dim=1)
            self._adaptive_consolidation(pooled, importance)
            with torch.no_grad():
                _ = self.topo_consolidator.consolidate(pooled, importance)
            return x

        reads = self._run_read_pipeline(
            x,
            fire_mask=fire_mask,
            recall_boost=recall_boost,
            context=context,
            cross_intensity=cross_intensity,
            apply_hns=self.enable_hns_fusion,
        )
        try:
            self._last_seq = reads["hg"]
        except Exception:
            pass
        return reads["fused"]

    # --------- Episodic bridge adapters ----------
    @staticmethod
    def _norm_bank_name(bank_name: str) -> str:
        name = str(bank_name).strip().lower()
        alias = {
            "hg": "hg",
            "episodic": "hg",
            "cgmn": "cgmn",
            "semantic": "cgmn",
            "curved": "curved",
            "spatial": "curved",
        }
        if name not in alias:
            raise ValueError(f"unknown bank_name={bank_name}")
        return alias[name]

    def write_bank(
        self,
        bank_name: str,
        x: torch.Tensor,
        *,
        fire_mask=None,
        recall_boost: float = 0.3,
        context: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Write directly into one specific bank using a normalized bank name.
        """
        name = self._norm_bank_name(bank_name)
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand_as(x)
        importance = self._estimate_importance(x, context)
        if name == "hg":
            _ = self.hg(x, operation="write", fire_mask=fire_mask, recall_boost=recall_boost)
        elif name == "cgmn":
            _ = self.cgmn(x, operation="write", fire_mask=fire_mask, recall_boost=recall_boost)
        else:
            _ = self.curved(x, operation="write", importance=importance)

    def _cached_read_banks(
        self,
        x: torch.Tensor,
        *,
        fire_mask=None,
        recall_boost: float = 0.3,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cache = getattr(self, "_read_pipeline_cache", None)
        if (
            cache is not None
            and cache.get("x_id") == id(x)
            and cache.get("fire_mask_id") == (None if fire_mask is None else id(fire_mask))
            and float(cache.get("recall_boost", -1.0)) == float(recall_boost)
        ):
            return cache["reads"]
        return self.read_banks(
            x,
            fire_mask=fire_mask,
            recall_boost=recall_boost,
            context=context,
            include_fused=True,
        )

    def read_bank(self, bank_name: str, x: torch.Tensor, *, fire_mask=None, recall_boost: float = 0.3) -> torch.Tensor:
        """
        Read one bank through the full triple-hybrid fusion pipeline.
        """
        raw = str(bank_name).strip().lower()
        reads = self._cached_read_banks(x, fire_mask=fire_mask, recall_boost=recall_boost)
        if raw == "fused":
            return reads["fused"]
        name = self._norm_bank_name(bank_name)
        if name == "hg":
            return reads["hg"]
        if name == "cgmn":
            return reads["cgmn"]
        if name == "curved":
            return reads["curved"]
        return reads["fused"]

    @torch.no_grad()
    def ingest_episodic_vectors(
        self,
        vectors: torch.Tensor,
        *,
        target_banks=("hg", "cgmn", "curved"),
        write_scale: float = 1.0,
    ) -> Dict[str, int]:
        """
        Ingest external episodic vectors into selected banks.
        Returns per-bank token counts ingested.
        """
        x = torch.as_tensor(vectors, device=self.mix.device, dtype=self.mix.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError("vectors must be [T,D], [B,D], or [B,S,D]")
        if x.size(-1) != self.hg.input_dim:
            raise ValueError(f"vectors last dim must be input_dim={self.hg.input_dim}")
        if not torch.isfinite(x).all():
            raise ValueError("vectors contains NaN/Inf")
        out = {}
        for name_raw in tuple(target_banks):
            name = self._norm_bank_name(name_raw)
            if name == "hg":
                self.hg.ingest_external_vectors(x, write_scale=write_scale)
            elif name == "cgmn":
                self.cgmn.ingest_external_vectors(x, write_scale=write_scale)
            else:
                self.curved.ingest_external_vectors(x, write_scale=write_scale)
            out[name] = int(x.size(0) * x.size(1))
        with torch.no_grad():
            self.nfm.ingest(x)
        return out
