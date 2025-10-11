import torch
import torch.nn as nn
from mnemonic_cortex.cortex import EnhancedMnemonicCortex

class CortexSeqModel(nn.Module):
    """Wrap EnhancedMnemonicCortex for token-level seq tasks."""
    def __init__(self, vocab_size: int, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cortex = EnhancedMnemonicCortex(input_dim=d_model, output_dim=d_model)
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, src: torch.Tensor):
        # src: (B,T)
        emb = self.embedding(src)          # (B,T,d)
        ctx = emb.mean(dim=1)              # simple context
        out = self.cortex(emb, ctx, operation='process')  # (B,T,d)
        logits = self.proj(out)            # (B,T,V)
        return logits
