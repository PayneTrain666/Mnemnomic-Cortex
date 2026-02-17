import torch
import torch.nn as nn
from typing import Dict
from .sensory_buffer import EnhancedSensoryBuffer
from .memory_curved import EnhancedCurvedMemory
from .triple_hybrid import EnhancedTripleHybridMemory
from .lightbulb import LightbulbDetector, ExplosiveRecallScaler
from .topology_manager_v2 import TopologyManagerV2
from .consolidated_lexicon import ConsolidatedLexicon
from .cms_ops import (
    CMSRecordLogger,
    dump_cpg_shards,
    load_cpg_shards_into_lexicon,
    run_cms_consolidation_ema,
)
from .consolidation_broker import ConsolidationBroker
from .ahg import AHGConfig, AntiHallucinationGuard
from .config_loader import apply_config_to_broker, load_yaml_config
from .cps import ConsolidatedParamStore, UnifiedParamCfg
from .cps_fuser import CPSFuser, FuserCfg
from .diagnostics import ModelDiagnostics

class EnhancedMnemonicCortex(nn.Module):
    """Top-level controller that routes inputs through buffer → WM → LTM with
    lightbulb-triggered 'explosive recall' (temperature modulation).
    Adds:
      • enable_energy_mode()
      • forgetting-style consolidation via consolidate_memories(threshold)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 sensory_buffer_size: int = 5,
                 wm_slots: int = 7, wm_slot_dim: int = 256,
                 ltm_hg_slots: int = 2048, ltm_cgmn_slots: int = 1024, ltm_curved_slots: int = 512,
                 fusion: str = 'weighted',
                 cms_vocab_size: int = 0,
                 cms_senses: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._ctx_heads = self._pick_num_heads(input_dim)

        self.sensory_buffer = EnhancedSensoryBuffer(sensory_buffer_size, input_dim)
        self.working_memory = EnhancedCurvedMemory(input_dim, hidden_dim=wm_slot_dim, mem_slots=wm_slots)
        self.long_term_memory = EnhancedTripleHybridMemory(input_dim, output_dim,
                                                           hg_slots=ltm_hg_slots, cgmn_slots=ltm_cgmn_slots, curved_slots=ltm_curved_slots,
                                                           fusion=fusion)

        # Context projection (kept simple: same dim by default)
        self.ctx_proj = nn.Linear(input_dim, input_dim)
        # Transformer-style context processors for stronger sequence conditioning.
        self.ctx_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.ctx_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=self._ctx_heads,
                dim_feedforward=max(128, input_dim * 4),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )
        self.ctx_norm = nn.LayerNorm(input_dim)
        self.query_ctx_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(input_dim)

        # Encoding and retrieval heads
        self.hippocampal_encoder = nn.Sequential(nn.Linear(input_dim*2, 512), nn.ReLU(), nn.Linear(512, 256))
        self.r_proj = nn.Linear(input_dim, 256)
        self.cue_to_input = nn.Linear(256, input_dim)
        self.retrieval = nn.Sequential(nn.Linear(256 + input_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

        # Lightbulb + temperature scaler
        self.lightbulb = LightbulbDetector(input_dim, thresh=2.0)
        self.temp_scaler = ExplosiveRecallScaler(base_temp=1.0, min_temp=0.5, boost=0.3)

        # Importance predictor for consolidation condition
        self.importance_predictor = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

        # Forgetting threshold for consolidation
        self.forgetting_threshold = 0.3
        self.energy_mode = False

        # Contrastive recall head (InfoNCE)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.recall_temp = 0.07  # temperature for InfoNCE

        # Learned write gate with STE
        self.write_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.topology = TopologyManagerV2(default_policy="default")
        self.topology.apply_to_model(self)
        self.consolidated_lexicon = None
        self.consolidation_broker = None
        self.ahg = AntiHallucinationGuard(AHGConfig())
        self.last_ahg_decision = None
        self.last_cms_aux = None
        self.last_cps_aux = None
        self.cms_logger = None
        self.cps = ConsolidatedParamStore(
            default_cfg=UnifiedParamCfg(d_euclid=input_dim, d_hyp=64, d_spher=64, d_fisher=64, d_phase=32)
        )
        self.cps_fuser = CPSFuser(FuserCfg(d_model=input_dim))
        self.diagnostics = ModelDiagnostics(enabled=False)
        if int(cms_vocab_size) > 0:
            self.enable_consolidated_lexicon(vocab_size=cms_vocab_size, senses=cms_senses)

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    # ---------------- Helpers ----------------
    def enable_energy_mode(self, enable: bool = True):
        self.energy_mode = enable
        self.long_term_memory.enable_energy_efficient_mode(enable)
        self.working_memory.enable_energy_efficient_mode(enable)

    @torch.no_grad()
    def apply_topology_policy(self, name: str):
        """Switch active topology policy and apply it immediately."""
        self.topology.activate_policy(name, model=self)

    @torch.no_grad()
    def topology_step(self, loss_value: float):
        """Call once per optimizer step with scalar loss."""
        self.topology.step(self, loss_value)

    def enable_consolidated_lexicon(
        self,
        vocab_size: int,
        senses: int = 3,
        d_hyper: int = 32,
        q_complex: int = 8,
        d_euclid: int = 16,
        d_pron: int = 12,
        d_char: int = 16,
        conformal_b: float = 0.02,
    ):
        """Attach a sense-aware consolidated lexicon for token-level fusion."""
        self.consolidated_lexicon = ConsolidatedLexicon(
            vocab_size=vocab_size,
            model_dim=self.input_dim,
            senses=senses,
            d_hyper=d_hyper,
            q_complex=q_complex,
            d_euclid=d_euclid,
            d_pron=d_pron,
            d_char=d_char,
            conformal_b=conformal_b,
            context_dim=self.input_dim,
        )
        return self

    def enable_consolidation_broker(
        self,
        vocab_size: int,
        store_configs: dict = None,
        config_path: str = "cm_config.yaml",
    ):
        """
        Attach a multi-store CMS broker (CRS/SKS/CAS/PSS) for intent routing.
        """
        self.consolidation_broker = ConsolidationBroker(
            vocab_size=vocab_size,
            model_dim=self.input_dim,
            store_configs=store_configs,
        )
        try:
            gcfg = load_yaml_config(config_path)
            apply_config_to_broker(self.consolidation_broker, gcfg)
            self.ahg = AntiHallucinationGuard(gcfg.ahg)
        except Exception:
            # Keep defaults if config is not present or invalid.
            pass
        return self

    def enable_ahg(self, cfg: AHGConfig = None):
        self.ahg = AntiHallucinationGuard(cfg or AHGConfig())
        return self

    def enable_diagnostics(
        self,
        enabled: bool = True,
        log_path: str = None,
        flush_every: int = 100,
    ):
        self.diagnostics.configure(enabled=enabled, log_path=log_path, flush_every=flush_every)
        return self

    def flush_diagnostics(self):
        return self.diagnostics.flush()

    def enable_cms_logger(
        self,
        out_dir: str = "cms_logs",
        max_buffer: int = 100000,
        sample_rate: float = 0.1,
    ):
        self.cms_logger = CMSRecordLogger(
            out_dir=out_dir, max_buffer=max_buffer, sample_rate=sample_rate
        )
        return self

    def flush_cms_logger(self):
        if self.cms_logger is None:
            return None
        return self.cms_logger.flush()

    @torch.no_grad()
    def run_cms_consolidation_ema(self, records, ema: float = 0.9):
        if self.consolidated_lexicon is None:
            return
        run_cms_consolidation_ema(self.consolidated_lexicon, records, ema=ema)

    def save_cms_shards(
        self,
        out_dir: str,
        shard_size: int = 4096,
        quantize: bool = False,
    ):
        if self.consolidated_lexicon is None:
            return []
        return dump_cpg_shards(
            self.consolidated_lexicon,
            out_dir=out_dir,
            shard_size=shard_size,
            quantize=quantize,
        )

    @torch.no_grad()
    def load_cms_shards(self, shard_dir: str, keys=None):
        if self.consolidated_lexicon is None:
            return
        load_cpg_shards_into_lexicon(self.consolidated_lexicon, shard_dir=shard_dir, keys=keys)

    def get_consolidated_parameter_groups(
        self,
        base_lr: float = 1e-3,
        assoc_lr_scale: float = 1.0,
        routing_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ):
        """Return logically grouped CMS params for optimizers."""
        if self.consolidated_lexicon is None:
            return []
        return self.consolidated_lexicon.build_optimizer_param_groups(
            base_lr=base_lr,
            assoc_lr_scale=assoc_lr_scale,
            routing_lr_scale=routing_lr_scale,
            weight_decay=weight_decay,
        )

    def _apply_consolidated_memory(
        self,
        sensory_input,
        token_ids,
        context_features=None,
        consolidation_intent: str = "auto",
    ):
        if (self.consolidated_lexicon is None and self.consolidation_broker is None) or token_ids is None:
            return sensory_input
        if token_ids.shape[:2] != sensory_input.shape[:2]:
            raise ValueError(
                f"token_ids shape {list(token_ids.shape)} must match sensory_input batch/seq "
                f"{list(sensory_input.shape[:2])}"
            )
        bsz, seq, dim = sensory_input.shape
        flat_ids = token_ids.reshape(-1).long()
        flat_base = sensory_input.reshape(-1, dim)
        if context_features is None:
            ctx = sensory_input.mean(dim=1, keepdim=True).expand(-1, seq, -1)
            flat_ctx = ctx.reshape(-1, dim)
        else:
            if context_features.shape[:2] != sensory_input.shape[:2]:
                raise ValueError(
                    f"context_features shape {list(context_features.shape)} must match sensory_input "
                    f"batch/seq {list(sensory_input.shape[:2])}"
                )
            flat_ctx = context_features.reshape(-1, context_features.size(-1))
        if self.consolidation_broker is not None:
            fused, baux = self.consolidation_broker.route_fuse(
                flat_ids,
                flat_base,
                flat_ctx,
                intent=consolidation_intent,
            )
            primary = baux["selected_store"]
            aux = baux["stores"][primary]
            aux["broker"] = baux
            self.last_cms_aux = aux
            self.diagnostics.log(
                "cms_broker_route",
                {
                    "intent": consolidation_intent,
                    "selected_store": primary,
                },
            )
        else:
            fused, _, aux = self.consolidated_lexicon(flat_ids, flat_base, flat_ctx)
            self.last_cms_aux = aux
            self.diagnostics.log("cms_single_store", {"intent": consolidation_intent})

        # Optional CPS fusion: token-keyed polymorphic unified parameter overlay.
        cps_fused = []
        cps_loss = fused.new_tensor(0.0)
        for tid in flat_ids.tolist():
            up = self.cps.ensure(f"token:{int(tid)}")
            v, loss = self.cps_fuser.fuse(up.view())
            cps_fused.append(v)
            cps_loss = cps_loss + loss
        if cps_fused:
            cps_fused = torch.stack(cps_fused, dim=0).to(fused.device, fused.dtype)
            fused = 0.85 * fused + 0.15 * cps_fused
            cps_loss = cps_loss / max(1, len(cps_fused))
            self.last_cps_aux = {"agree_loss": cps_loss}
            aux["cps"] = self.last_cps_aux
            self.diagnostics.record_scalar("cps_agree_loss", float(cps_loss.detach().item()))
            self.diagnostics.log("cps_fusion", {"count": int(len(cps_fused))})

        if self.cms_logger is not None:
            self.cms_logger.log_from_aux(flat_ids, aux)
        return fused.view(bsz, seq, dim)

    def _ste_write_gate(self, x):
        """Compute write gate with straight-through estimator.
        x: (B,S,d) -> gate_prob: (B,1), gate_hard: (B,1)
        """
        pooled = x.mean(dim=1)  # (B,d)
        gate_prob = self.write_gate(pooled)  # (B,1) continuous in [0,1]
        
        if self.training:
            # Sample binary decision
            gate_hard = (torch.rand_like(gate_prob) < gate_prob).float()
            # STE: forward uses hard, backward uses soft
            gate_ste = gate_hard - gate_prob.detach() + gate_prob
        else:
            # At inference, threshold at 0.5
            gate_ste = (gate_prob > 0.5).float()
        
        return gate_ste, gate_prob

    def _tile_context(self, context, S):
        # context: (B,c) -> project to (B,S,input_dim)
        cproj = self.ctx_proj(context)                    # (B,input_dim)
        return cproj.unsqueeze(1).expand(-1, S, -1)

    def process_sensory_input(self, sensory_input):
        self.sensory_buffer.update(sensory_input)
        base = self.sensory_buffer.attention_filter(sensory_input)
        attn_out, _ = self.ctx_attn(base, base, base, need_weights=False)
        enc_out = self.ctx_encoder(base)
        out = self.ctx_norm(base + 0.5 * attn_out + 0.5 * enc_out)
        self.diagnostics.record_scalar("sensory_norm", float(out.norm(dim=-1).mean().item()))
        return out

    def encode_memory(self, info, context, mtype):
        B,S,d = info.shape
        ctx = self._tile_context(context, S)
        pooled = torch.cat([info, ctx], dim=-1).mean(dim=1)      # (B, 2d)
        idx = self.hippocampal_encoder(pooled)                   # (B,256)

        cue = idx.unsqueeze(1).expand(-1, S, -1)                 # (B,S,256)
        # Project cue to input_dim if needed
        if cue.size(-1) != self.input_dim:
            cue = self.cue_to_input(cue)

        if mtype == 'episodic':
            self.long_term_memory.hg(cue, operation='write')
        elif mtype == 'semantic':
            self.long_term_memory.cgmn(cue, operation='write')
        else:
            self.long_term_memory.curved(cue, operation='write')
        return idx

    def retrieve_memory(self, cue, context, strategy='associative', fire_mask=None, recall_boost: float = 0.3):
        """Retrieve memories with optional explosive recall (fire_mask)."""
        B,S,d = cue.shape
        ctx = self._tile_context(context, S)
        c = (cue + ctx) * 0.5
        qctx, _ = self.query_ctx_attn(c, ctx, ctx, need_weights=False)
        c = self.query_norm(c + qctx)

        if self.consolidation_broker is not None and self.ahg is not None:
            query = c.mean(dim=1)
            bdiag = self.consolidation_broker.route_read(query, intent="auto", k=8)
            xdiag = self.consolidation_broker.cross_store_diagnostics(query, k=8)
            decision = self.ahg.decide(bdiag, xdiag)
            self.last_ahg_decision = {
                "action": decision.action,
                "reason": decision.reason,
                "scores": decision.scores,
            }
            self.diagnostics.log(
                "ahg_decision",
                {
                    "action": decision.action,
                    "strategy_in": strategy,
                    "proto": decision.scores.get("proto", -1.0),
                    "fisher": decision.scores.get("fisher", -1.0),
                    "agree": decision.scores.get("agree", -1.0),
                },
            )
            if decision.action == "explosive":
                recall_boost = max(recall_boost, 0.65)
            elif decision.action == "refine":
                strategy = "direct"
            elif decision.action == "ask":
                # conservative fallback answer vector from context only
                z = torch.zeros(context.size(0), 256, device=context.device, dtype=context.dtype)
                return self.retrieval(torch.cat([z, context], dim=-1))

        if strategy == 'direct':
            r = self.long_term_memory(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)
        elif strategy == 'associative':
            # Use curved memory path (kept simple; doesn't use fire)
            r = self.long_term_memory.curved(c, operation='read')
        else:
            # Reconstructive via HG
            r = self.long_term_memory.hg(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)

        cue_vec = self.r_proj(r.mean(dim=1))
        self.diagnostics.record_scalar("recall_boost", float(recall_boost))
        self.diagnostics.log("retrieve_path", {"strategy": strategy})
        return self.retrieval(torch.cat([cue_vec, context], dim=-1))

    @torch.no_grad()
    def consolidate_memories(self, threshold: float = None):
        """Forget rarely used slots (usage-based) and gently decay working-memory importance."""
        th = self.forgetting_threshold if threshold is None else float(threshold)
        self.long_term_memory.consolidate_unused(th)
        self.working_memory.memory_importance.mul_(0.999)
        self.diagnostics.log("consolidate_memories", {"threshold": th})
        if self.consolidation_broker is not None:
            try:
                hg = self.long_term_memory.hg.output_projection(self.long_term_memory.hg.values.detach())
                cg = self.long_term_memory.cgmn.output_projection(self.long_term_memory.cgmn.memory_slots.detach())
                cv = self.long_term_memory.curved.decoder(self.long_term_memory.curved.memory_slots.detach())
                items = torch.cat([hg, cg, cv], dim=0)
                metas = (
                    [{"domain": "reasoning", "tags": ["hg"]}] * hg.size(0)
                    + [{"domain": "reasoning", "tags": ["cgmn"]}] * cg.size(0)
                    + [{"domain": "reasoning", "tags": ["curved"]}] * cv.size(0)
                )
                self.consolidation_broker.unify_and_route_write(items, metas, domain="reasoning")
                self.diagnostics.log(
                    "unify_and_route_write",
                    {"items": int(items.size(0)), "domain": "reasoning"},
                )
            except Exception:
                pass

    def get_metrics(self):
        """Collect diagnostic metrics from all components."""
        metrics = {}
        # Lightbulb metrics
        metrics.update(self.lightbulb.get_metrics())
        # Memory module metrics
        metrics.update(self.long_term_memory.hg.get_metrics())
        metrics.update(self.long_term_memory.cgmn.get_metrics())
        metrics.update(self.long_term_memory.curved.get_metrics())
        metrics.update(self.working_memory.get_metrics())
        if self.consolidation_broker is not None:
            b = self.consolidation_broker.get_metrics()
            for k, v in b.items():
                for kk, vv in v.items():
                    metrics[f"broker_{k}_{kk}"] = vv
        if self.last_ahg_decision is not None:
            metrics["ahg_last_action"] = self.last_ahg_decision.get("action", "none")
        dsum = self.diagnostics.summary()
        metrics["diag_enabled"] = float(1.0 if dsum.get("enabled") else 0.0)
        metrics["diag_events_buffered"] = float(dsum.get("events_buffered", 0))
        for k, v in dsum.get("ema", {}).items():
            metrics[f"diag_ema_{k}"] = float(v)
        # Cortex-level
        metrics['energy_mode'] = self.energy_mode
        metrics['forgetting_threshold'] = self.forgetting_threshold
        return metrics

    @torch.no_grad()
    def get_holonomy_stats(self, x: torch.Tensor):
        """Diagnostics for latent spin holonomy in HG/CGMN paths."""
        bsz, seq, _ = x.shape
        hg_man = self.long_term_memory.hg.encode_to_manifold(x)
        cg_man = self.long_term_memory.cgmn.manifold_projection(x).view(
            bsz, seq, self.long_term_memory.cgmn.D, 3
        )
        cg_man = self.long_term_memory.cgmn._evolve(cg_man)
        return {
            "hg": self.long_term_memory.hg.holonomy_stats(hg_man),
            "cgmn": self.long_term_memory.cgmn.holonomy_stats(cg_man),
        }

    def save_checkpoint(self, path: str, version: str = '1.0'):
        """Save versioned checkpoint with explicit schema."""
        import torch
        checkpoint = {
            'version': version,
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'forgetting_threshold': self.forgetting_threshold,
                'energy_mode': self.energy_mode,
            },
            'lightbulb_state': {
                'trigger_rate_ema': self.lightbulb.trigger_rate_ema.item(),
                'threshold': self.lightbulb.thresh,
                'total_fires': self.lightbulb.total_fires.item(),
                'total_samples': self.lightbulb.total_samples.item(),
            },
            'memory_state': {
                'hg_usage': self.long_term_memory.hg.usage_counts.clone(),
                'cgmn_usage': self.long_term_memory.cgmn.usage_counts.clone(),
                'curved_usage': self.long_term_memory.curved.usage_counts.clone(),
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load checkpoint with version validation."""
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        
        # Version check
        version = checkpoint.get('version', 'unknown')
        if version != '1.0' and strict:
            raise ValueError(f"Checkpoint version {version} does not match expected '1.0'")
        elif version != '1.0':
            print(f"Warning: Loading checkpoint version {version}, expected '1.0'. Compatibility not guaranteed.")
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Restore config
        if 'config' in checkpoint:
            self.forgetting_threshold = checkpoint['config'].get('forgetting_threshold', self.forgetting_threshold)
            self.energy_mode = checkpoint['config'].get('energy_mode', self.energy_mode)
        
        # Restore lightbulb state
        if 'lightbulb_state' in checkpoint:
            lb = checkpoint['lightbulb_state']
            self.lightbulb.trigger_rate_ema.fill_(lb.get('trigger_rate_ema', 0.0))
            self.lightbulb.thresh = lb.get('threshold', 2.0)
            self.lightbulb.total_fires.fill_(lb.get('total_fires', 0))
            self.lightbulb.total_samples.fill_(lb.get('total_samples', 0))
        
        # Restore memory usage
        if 'memory_state' in checkpoint:
            mem = checkpoint['memory_state']
            if 'hg_usage' in mem:
                self.long_term_memory.hg.usage_counts.copy_(mem['hg_usage'])
            if 'cgmn_usage' in mem:
                self.long_term_memory.cgmn.usage_counts.copy_(mem['cgmn_usage'])
            if 'curved_usage' in mem:
                self.long_term_memory.curved.usage_counts.copy_(mem['curved_usage'])

    def compute_recall_loss(self, cue, context):
        """InfoNCE contrastive loss: cue vs. retrieved memory.
        cue: (B,S,d), context: (B,d)
        Returns scalar loss.
        """
        B, S, d = cue.shape
        # Retrieve from LTM
        retrieved = self.retrieve_memory(cue, context, strategy='direct')  # (B,d)
        
        # Project both to contrastive space
        cue_pooled = cue.mean(dim=1)  # (B,d)
        z_cue = torch.nn.functional.normalize(self.contrastive_proj(cue_pooled), dim=-1)  # (B,128)
        z_ret = torch.nn.functional.normalize(self.contrastive_proj(retrieved), dim=-1)   # (B,128)
        
        # InfoNCE: positive = same batch index, negatives = others
        logits = torch.matmul(z_cue, z_ret.T) / self.recall_temp  # (B,B)
        labels = torch.arange(B, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    # ---------------- Forward ----------------
    def forward(
        self,
        sensory_input,
        context,
        operation='process',
        return_aux_losses=False,
        token_ids=None,
        use_consolidated_memory=False,
        context_features=None,
        consolidation_intent="auto",
    ):
        if use_consolidated_memory and token_ids is not None:
            sensory_input = self._apply_consolidated_memory(
                sensory_input,
                token_ids,
                context_features=context_features,
                consolidation_intent=consolidation_intent,
            )
        fire = self.lightbulb(sensory_input)                # (B,)
        temp = self.temp_scaler(fire)                       # (B,) scalars
        self.working_memory.set_temperature(temp)
        self.long_term_memory.set_temperature(temp)

        if operation == 'process':
            filtered = self.process_sensory_input(sensory_input)          # (B,S,d)
            self.diagnostics.record_scalar("fire_rate", float(fire.float().mean().item()))

            # --- Working-memory write phase ----------------------------------
            # Store filtered sensory input into WM slots with importance gating.
            imp = self.importance_predictor(filtered.mean(dim=1))         # (B,1)
            self.working_memory(filtered, operation='write', importance=imp)
            self.diagnostics.record_scalar("importance_mean", float(imp.mean().item()))

            # --- Working-memory read phase -----------------------------------
            wm_out = self.working_memory(filtered, operation='read')      # (B,S,d)

            # --- Consolidation into long-term memory ------------------------
            if self.training:  # consolidate only during training
                # Learned write gate with STE
                gate, gate_prob = self._ste_write_gate(filtered)  # (B,1)
                scaled = wm_out * imp.unsqueeze(-1)  # (B,S,d)
                self.diagnostics.record_scalar("write_gate_prob", float(gate_prob.mean().item()))
                
                # Only encode if gate=1 (batched conditional write)
                if gate.sum() > 0:  # at least one sample wants to write
                    # Mask the batch
                    write_mask = gate.squeeze(-1) > 0.5  # (B,)
                    if write_mask.any():
                        self.encode_memory(scaled[write_mask], context[write_mask], mtype='episodic')
            
            # --- Compute auxiliary losses if requested ----------------------
            if return_aux_losses and self.training:
                recall_loss = self.compute_recall_loss(filtered, context)
                gate, gate_prob = self._ste_write_gate(filtered)
                self.diagnostics.record_scalar("recall_loss", float(recall_loss.detach().item()))
                return wm_out, {'recall_loss': recall_loss, 'write_gate_prob': gate_prob.mean()}
            
            return wm_out

        elif operation == 'retrieve':
            # Route fire to LTM for sharper readout on the cue as well
            fire_ret = self.lightbulb(sensory_input)  # (B,)
            return self.retrieve_memory(sensory_input, context, strategy='direct', fire_mask=fire_ret, recall_boost=0.3)

        else:
            self.consolidate_memories()
            return None
