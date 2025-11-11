import torch
import torch.nn as nn
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from benchmark.tasks import TOK2IDX  # for SOS token when auto-shifting target inputs

class CortexSeqModel(nn.Module):
    """Wrap EnhancedMnemonicCortex for token-level seq tasks."""
    def __init__(self, vocab_size: int, d_model: int = 128, recall_loss_weight: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cortex = EnhancedMnemonicCortex(input_dim=d_model, output_dim=d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.recall_loss_weight = recall_loss_weight
        
    def forward(self, src: torch.Tensor, return_aux_losses=False):
        # src: (B,T)
        emb = self.embedding(src)          # (B,T,d)
        ctx = emb.mean(dim=1)              # simple context
        
        if return_aux_losses and self.training:
            out, aux = self.cortex(emb, ctx, operation='process', return_aux_losses=True)
            logits = self.proj(out)        # (B,T,V)
            return logits, aux
        else:
            out = self.cortex(emb, ctx, operation='process')  # (B,T,d)
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


def get_model(name: str, vocab_size: int):
    name = name.lower()
    if name == 'cortex':
        return CortexSeqModel(vocab_size)
    if name == 'lstm':
        return LSTMSeq2Seq(vocab_size)
    if name == 'transformer':
        return TinyTransformer(vocab_size)
    raise ValueError(f"Unknown model {name}")
