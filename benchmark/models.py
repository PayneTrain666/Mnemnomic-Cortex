import torch
import torch.nn as nn
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.anti_hallucination import AHGThresholds, HallucinationGuard
from benchmark.tasks import TOK2IDX  # for SOS token when auto-shifting target inputs

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
        ltm_hg_slots: int = 2048,
        ltm_cgmn_slots: int = 1024,
        ltm_curved_slots: int = 512,
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
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cortex = EnhancedMnemonicCortex(
            input_dim=d_model,
            output_dim=d_model,
            sensory_buffer_size=sensory_buffer_size,
            wm_slots=wm_slots,
            wm_slot_dim=wm_slot_dim,
            ltm_hg_slots=ltm_hg_slots,
            ltm_cgmn_slots=ltm_cgmn_slots,
            ltm_curved_slots=ltm_curved_slots,
            fusion=fusion,
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
