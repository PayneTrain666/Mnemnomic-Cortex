import inspect
import torch
import torch.nn as nn
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.anti_hallucination import AHGThresholds, HallucinationGuard
from benchmark.tasks import TOK2IDX  # for SOS token when auto-shifting target inputs
import math

_CORTEX_PARAM_NAMES = set(inspect.signature(EnhancedMnemonicCortex.__init__).parameters) - {"self"}

class CortexSeqModel(nn.Module):
    """Wrap EnhancedMnemonicCortex for token-level seq tasks."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 144,
        fusion: str = "weighted",
        sensory_buffer_size: int = 5,
        wm_slots: int = 7,
        wm_slot_dim: int = 256,
        ltm_hg_dim: int = 24,
        ltm_hg_slots: int = 1028,
        ltm_hg_qubits: int = 8,
        ltm_cgmn_dim: int = 16,
        ltm_cgmn_slots: int = 512,
        ltm_cgmn_slot_dim: int = 256,
        ltm_curved_slots: int = 128,
        ltm_curved_curvature: int = 8,
        ltm_n_transformer_layers: int = 3,
        ltm_n_heads: int = 8,
        ltm_attention_type: str = "multiscale",
        recall_loss_weight: float = 0.1,
        cms_enabled: bool = True,
        cms_senses: int = 3,
        cms_aux_weight: float = 0.02,
        cms_log_dir: str = None,
        cms_log_sample_rate: float = 0.1,
        cms_log_max_buffer: int = 100000,
        cms_multi_store: bool = False,
        cms_consolidation_intent: str = "auto",
        ahg_enabled: bool = False,
        hgm_enabled: bool = False,
        ltm_curved_hidden_dim: int = 0,
        ltm_curved_hidden_mult: float = 1.5,
        task_decoder_enabled: bool = False,
        task_decoder_layers: int = 4,
        task_decoder_heads: int = 8,
        task_decoder_dropout: float = 0.1,
        task_decoder_use_sinusoidal: bool = True,
        **cortex_kwargs,
    ):
        super().__init__()
        self.hgm_enabled = bool(hgm_enabled)
        self.ltm_curved_hidden_dim = int(ltm_curved_hidden_dim) if int(ltm_curved_hidden_dim) > 0 else int(d_model * ltm_curved_hidden_mult)
        self.enable_full_fusion_stack = bool(cortex_kwargs.pop("enable_full_fusion_stack", False))
        cortex_only_kwargs = {
            k: v for k, v in cortex_kwargs.items() if k in _CORTEX_PARAM_NAMES
        }
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cortex = EnhancedMnemonicCortex(
            input_dim=d_model,
            output_dim=d_model,
            sensory_buffer_size=sensory_buffer_size,
            wm_slots=wm_slots,
            wm_slot_dim=wm_slot_dim,
            ltm_hg_dim=ltm_hg_dim,
            ltm_hg_slots=ltm_hg_slots,
            ltm_hg_qubits=ltm_hg_qubits,
            ltm_cgmn_dim=ltm_cgmn_dim,
            ltm_cgmn_slots=ltm_cgmn_slots,
            ltm_cgmn_slot_dim=ltm_cgmn_slot_dim,
            ltm_curved_hidden=cortex_only_kwargs.pop("ltm_curved_hidden", self.ltm_curved_hidden_dim),
            ltm_curved_curvature=ltm_curved_curvature,
            ltm_curved_slots=ltm_curved_slots,
            ltm_n_transformer_layers=ltm_n_transformer_layers,
            ltm_n_heads=ltm_n_heads,
            ltm_attention_type=ltm_attention_type,
            fusion=fusion,
            **cortex_only_kwargs,
        )
        if self.hgm_enabled:
            if self.cortex.shared_memory_subsystem is None:
                self.cortex.enable_shared_memory_subsystem(num_slots=max(256, ltm_hg_slots))
            if self.cortex.hg_episodic_ltm is None:
                self.cortex.enable_hg_episodic_ltm()
        if self.enable_full_fusion_stack:
            self.cortex.enable_cps_cms_full_stack(
                vocab_size=vocab_size,
                enable_broker=cms_enabled and cms_multi_store,
                enable_advanced=False,
                enable_reasoning_bridge=True,
                enable_qdt_wm_bridge=True,
            )
        self.cms_enabled = bool(cms_enabled)
        if cms_enabled:
            self.cortex.enable_consolidated_lexicon(vocab_size=vocab_size, senses=cms_senses)
            if cms_multi_store:
                self.cortex.enable_consolidation_broker(vocab_size=vocab_size)
            if cms_log_dir:
                self.cortex.enable_cms_logger(
                    out_dir=cms_log_dir,
                    max_buffer=cms_log_max_buffer,
                    sample_rate=cms_log_sample_rate,
                )
        self.proj = nn.Linear(d_model, vocab_size)
        self.recall_loss_weight = recall_loss_weight
        self.cms_aux_weight = float(cms_aux_weight)
        self.cms_consolidation_intent = str(cms_consolidation_intent)
        self.ahg_enabled = bool(ahg_enabled)
        self.hallucination_guard = None
        self.task_decoder_enabled = bool(task_decoder_enabled)
        self.task_decoder_use_sinusoidal = bool(task_decoder_use_sinusoidal)
        self.task_decoder_layers = int(task_decoder_layers)
        self.task_decoder_heads = int(task_decoder_heads)
        if self.task_decoder_enabled:
            nhead = self._resolve_heads(d_model, max(1, int(task_decoder_heads)))
            self.pre_fusion_stack_attn = nn.MultiheadAttention(
                d_model,
                num_heads=nhead,
                dropout=float(task_decoder_dropout),
                batch_first=True,
            )
            self.memory_compare_attn = nn.MultiheadAttention(
                d_model,
                num_heads=nhead,
                dropout=float(task_decoder_dropout),
                batch_first=True,
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(256, d_model * 4),
                dropout=float(task_decoder_dropout),
                activation="gelu",
                batch_first=True,
            )
            self.memory_compare_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.memory_compare_norm = nn.LayerNorm(d_model)
            self.system_specialization_head = nn.Linear(d_model, 1)
            self.mann_proxy = nn.Linear(d_model, d_model)
            self.final_system_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Linear(d_model, 3),
            )
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(256, d_model * 4),
                dropout=float(task_decoder_dropout),
                activation="gelu",
                batch_first=True,
            )
            self.task_decoder = nn.TransformerDecoder(dec_layer, num_layers=max(1, int(task_decoder_layers)))
            self.task_decoder_norm = nn.LayerNorm(d_model)
            self.last_task_decoder_stats = {}

    def topology_step(self, loss_value: float):
        if hasattr(self.cortex, "topology_step"):
            self.cortex.topology_step(float(loss_value))

    def flush_cms_logger(self):
        if hasattr(self.cortex, "flush_cms_logger"):
            return self.cortex.flush_cms_logger()
        return None

    def enable_hallucination_guard(
        self,
        thresholds: AHGThresholds = None,
        retriever_fn=None,
        cite_align_fn=None,
        cms_signals_fn=None,
    ):
        if thresholds is None:
            thresholds = AHGThresholds()
        self.hallucination_guard = HallucinationGuard(
            thresholds=thresholds,
            generate_fn=lambda prompt, **kw: {"out_text": "", "logits": None, "tokens": None},
            retriever_fn=retriever_fn,
            cite_align_fn=cite_align_fn,
            cms_signals_fn=cms_signals_fn,
        )
        self.ahg_enabled = True
        return self

    def _generate_for_guard(self, src: torch.Tensor, temperature: float = 0.2, do_sample: bool = False):
        self.eval()
        with torch.no_grad():
            logits = self.forward(src, return_aux_losses=False)
            if do_sample:
                probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1)
                tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), num_samples=1
                ).view(src.size(0), src.size(1))
            else:
                tokens = logits.argmax(dim=-1)
            out_text = " ".join(str(int(t)) for t in tokens[0].tolist())
            return {"out_text": out_text, "logits": logits, "tokens": tokens}

    def guarded_infer(
        self,
        src: torch.Tensor,
        query_text: str = None,
        retriever_fn=None,
        cite_align_fn=None,
        cms_signals_fn=None,
        thresholds: AHGThresholds = None,
    ):
        if not self.ahg_enabled and thresholds is None and retriever_fn is None and cite_align_fn is None and cms_signals_fn is None:
            # guard not enabled and no ad-hoc guard config: just return logits
            return {"mode": "answer", "logits": self.forward(src), "text": None, "signals": {}}

        guard = self.hallucination_guard
        if guard is None:
            guard = HallucinationGuard(
                thresholds=thresholds or AHGThresholds(),
                generate_fn=lambda prompt, **kw: self._generate_for_guard(
                    src, temperature=kw.get("temperature", 0.2), do_sample=kw.get("do_sample", False)
                ),
                retriever_fn=retriever_fn,
                cite_align_fn=cite_align_fn,
                cms_signals_fn=cms_signals_fn,
            )
        else:
            guard.generate = lambda prompt, **kw: self._generate_for_guard(
                src, temperature=kw.get("temperature", 0.2), do_sample=kw.get("do_sample", False)
            )
            if retriever_fn is not None:
                guard.retrieve = retriever_fn
            if cite_align_fn is not None:
                guard.cite_align = cite_align_fn
            if cms_signals_fn is not None:
                guard.cms_signals = cms_signals_fn
            if thresholds is not None:
                guard.th = thresholds

        prompt = query_text if query_text is not None else " ".join(str(int(t)) for t in src[0].tolist())
        decision = guard.answer(prompt)
        decision["logits"] = self.forward(src, return_aux_losses=False)
        return decision

    def _cms_aux_loss(self):
        aux = getattr(self.cortex, "last_cms_aux", None)
        cps_aux = getattr(self.cortex, "last_cps_aux", None)
        cps_loss = None
        if cps_aux is not None and isinstance(cps_aux.get("agree_loss", None), torch.Tensor):
            cps_loss = cps_aux["agree_loss"]
        if aux is None:
            return cps_loss
        w = aux.get("weights", None)
        warp = aux.get("warp", None)
        if w is None or warp is None:
            return cps_loss
        # Encourage peaked-but-not-collapsed sense routing + small conformal deviations.
        entropy = -(w * (w.clamp_min(1e-9)).log()).sum(dim=-1).mean()
        warp_reg = ((warp - 1.0) ** 2).mean()
        loss = self.cms_aux_weight * (0.1 * entropy + warp_reg)
        if cps_loss is not None:
            loss = loss + cps_loss
        return loss

    @staticmethod
    def _sinusoidal_pos_encoding(length: int, dim: int, device, dtype):
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        half = max(1, dim // 2)
        div = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe.to(dtype=dtype)

    @staticmethod
    def _resolve_heads(dim: int, target_heads: int) -> int:
        for h in range(max(1, int(target_heads)), 0, -1):
            if dim % h == 0:
                return h
        return 1

    def _read_mann_memory(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Read MANN-facing state from reasoning controller when available.
        Falls back to a learned proxy so the path stays differentiable.
        """
        rc = getattr(self.cortex, "reasoning_controller_api", None)
        if rc is None:
            return self.mann_proxy(emb)
        query = emb.mean(dim=1)
        with torch.no_grad():
            try:
                api_result = rc.run_reasoning_pass(
                    query,
                    content="seq_model_prefusion_mann",
                    write_permission=False,
                )
                out = api_result.result.mann_output
                if out is None:
                    out = api_result.result.output
                mann_state = out
            except Exception:
                mann_state = query
        return mann_state.unsqueeze(1).expand(-1, emb.size(1), -1).to(emb.device, emb.dtype)

    def _task_specific_decode(self, emb: torch.Tensor, fused_out: torch.Tensor):
        """
        Build a memory-attended decoder path using:
        - WM output
        - per-bank LTM outputs (HG/CGMN/Curved)
        - fused cortex output
        """
        if not self.task_decoder_enabled:
            return fused_out
        fire = self.cortex.lightbulb(emb)
        if hasattr(self.cortex, "_sync_attention_stacks"):
            self.cortex._sync_attention_stacks(emb)
        wm_op = self.cortex._wm_read_operation() if hasattr(self.cortex, "_wm_read_operation") else "read"
        wm = self.cortex.working_memory(emb, operation=wm_op)
        ltm = self.cortex.long_term_memory
        if hasattr(ltm, "read_banks"):
            bank_reads = ltm.read_banks(emb, fire_mask=fire, recall_boost=0.2, include_fused=False)
            hg, cg, cv = bank_reads["hg"], bank_reads["cgmn"], bank_reads["curved"]
        else:
            hg = ltm.read_bank("hg", emb, fire_mask=fire, recall_boost=0.2)
            cg = ltm.read_bank("cgmn", emb, fire_mask=fire, recall_boost=0.2)
            cv = ltm.read_bank("curved", emb, fire_mask=fire, recall_boost=0.2)
        mann = self._read_mann_memory(emb)
        mem_tokens = torch.stack([wm, hg, cg, cv, mann], dim=2)  # (B,T,5,d)
        bsz, seq, nsys, dim = mem_tokens.shape
        mem_flat = mem_tokens.reshape(bsz * seq, nsys, dim)
        compared, _ = self.memory_compare_attn(mem_flat, mem_flat, mem_flat, need_weights=False)
        cooperative = self.memory_compare_norm(mem_flat + compared)
        cooperative = self.memory_compare_encoder(cooperative)
        spec_logits = self.system_specialization_head(cooperative).squeeze(-1)  # (B*T,5)
        spec_weights = torch.softmax(spec_logits, dim=-1)
        cooperative_weighted = (spec_weights.unsqueeze(-1) * cooperative).sum(dim=1).reshape(bsz, seq, dim)

        wm_coop = cooperative[:, 0, :].reshape(bsz, seq, dim)
        ltm_coop = cooperative[:, 1:4, :]
        ltm_spec = spec_weights[:, 1:4]
        ltm_wsum = ltm_spec.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ltm_fused = ((ltm_spec.unsqueeze(-1) * ltm_coop).sum(dim=1) / ltm_wsum).reshape(bsz, seq, dim)
        mann_fused = cooperative[:, 4, :].reshape(bsz, seq, dim)

        triplet_logits = self.final_system_gate(torch.cat([wm_coop, ltm_fused, mann_fused], dim=-1))
        triplet_weights = torch.softmax(triplet_logits, dim=-1)
        tri = torch.stack([wm_coop, ltm_fused, mann_fused], dim=2)
        hierarchical_fused = (triplet_weights.unsqueeze(-1) * tri).sum(dim=2)

        stack = torch.cat([wm_coop, ltm_fused, mann_fused, cooperative_weighted, fused_out], dim=1)  # (B, 5T, d)
        q = fused_out
        if self.task_decoder_use_sinusoidal:
            pe_q = self._sinusoidal_pos_encoding(q.size(1), q.size(2), q.device, q.dtype).unsqueeze(0)
            pe_s = self._sinusoidal_pos_encoding(stack.size(1), stack.size(2), stack.device, stack.dtype).unsqueeze(0)
            q = q + pe_q
            stack = stack + pe_s
        attended, _ = self.pre_fusion_stack_attn(q, stack, stack, need_weights=False)
        decoded = self.task_decoder(tgt=q, memory=stack)
        self.last_task_decoder_stats = {
            "wm_weight_mean": float(triplet_weights[:, :, 0].detach().mean().item()),
            "ltm_weight_mean": float(triplet_weights[:, :, 1].detach().mean().item()),
            "mann_weight_mean": float(triplet_weights[:, :, 2].detach().mean().item()),
            "spec_wm_mean": float(spec_weights[:, 0].detach().mean().item()),
            "spec_hg_mean": float(spec_weights[:, 1].detach().mean().item()),
            "spec_cgmn_mean": float(spec_weights[:, 2].detach().mean().item()),
            "spec_curved_mean": float(spec_weights[:, 3].detach().mean().item()),
            "spec_mann_mean": float(spec_weights[:, 4].detach().mean().item()),
        }
        return self.task_decoder_norm(
            fused_out + 0.20 * cooperative_weighted + 0.25 * hierarchical_fused + 0.20 * attended + 0.55 * decoded
        )
        
    def forward(self, src: torch.Tensor, return_aux_losses=False):
        # src: (B,T)
        emb = self.embedding(src)          # (B,T,d)
        ctx = emb.mean(dim=1)              # simple context
        
        if return_aux_losses and self.training:
            out, aux = self.cortex(
                emb,
                ctx,
                operation='process',
                return_aux_losses=True,
                token_ids=src,
                use_consolidated_memory=self.cms_enabled,
                context_features=emb,
                consolidation_intent=self.cms_consolidation_intent,
            )
            cms_loss = self._cms_aux_loss()
            if cms_loss is not None:
                aux["cms_loss"] = cms_loss
            out = self._task_specific_decode(emb, out)
            if self.task_decoder_enabled and isinstance(self.last_task_decoder_stats, dict):
                for k, v in self.last_task_decoder_stats.items():
                    aux[k] = torch.tensor(float(v), device=out.device, dtype=out.dtype)
            logits = self.proj(out)        # (B,T,V)
            return logits, aux
        else:
            out = self.cortex(
                emb,
                ctx,
                operation='process',
                token_ids=src,
                use_consolidated_memory=self.cms_enabled,
                context_features=emb,
                consolidation_intent=self.cms_consolidation_intent,
            )  # (B,T,d)
            out = self._task_specific_decode(emb, out)
            logits = self.proj(out)        # (B,T,V)
            return logits


class LSTMSeq2Seq(nn.Module):
    """Simple encoder–decoder LSTM with shared embedding and linear output."""
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor=None):
        # src: (B,T); if tgt_in is omitted, use shifted src with SOS at t=0
        if tgt_in is None:
            sos = torch.full((src.size(0),1), TOK2IDX['<s>'], device=src.device, dtype=src.dtype)
            tgt_in = torch.cat([sos, src[:,:-1]], dim=1)
        enc_out, (h,c) = self.encoder(self.embedding(src))
        dec_out, _ = self.decoder(self.embedding(tgt_in), (h,c))
        return self.proj(dec_out)


class TinyTransformer(nn.Module):
    """2-layer Transformer encoder-decoder with shared embedding."""
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def _add_pos(self, x):
        return x + self.pos_emb[:x.size(1)]

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor=None):
        # src: (B,T); auto-generate tgt_in if not provided
        if tgt_in is None:
            sos = torch.full((src.size(0),1), TOK2IDX['<s>'], device=src.device, dtype=src.dtype)
            tgt_in = torch.cat([sos, src[:,:-1]], dim=1)
        src_e = self._add_pos(self.embedding(src))
        memory = self.encoder(src_e)
        tgt_e = self._add_pos(self.embedding(tgt_in))
        dec_out = self.decoder(tgt_e, memory)
        return self.proj(dec_out)


def get_model(name: str, vocab_size: int, **kwargs):
    name = name.lower()
    if name == 'cortex':
        return CortexSeqModel(vocab_size, **kwargs)
    if name == 'lstm':
        filtered = {k: v for k, v in kwargs.items() if k in {"d_model", "num_layers"}}
        return LSTMSeq2Seq(vocab_size, **filtered)
    if name == 'transformer':
        filtered = {k: v for k, v in kwargs.items() if k in {"d_model", "nhead", "num_layers"}}
        return TinyTransformer(vocab_size, **filtered)
    raise ValueError(f"Unknown model {name}")
