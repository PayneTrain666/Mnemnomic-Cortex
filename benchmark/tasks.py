from typing import Tuple
import torch
from torch.utils.data import Dataset

SYMBOLS = list("abcdefghijklmnopqrstuvwxyz0123456789")
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + SYMBOLS
VOCAB_SIZE = len(VOCAB)
IDX2TOK = {i:t for i,t in enumerate(VOCAB)}
TOK2IDX = {t:i for i,t in IDX2TOK.items()}

def random_sequence(max_len: int) -> torch.Tensor:
    L = torch.randint(1, max_len+1, (1,)).item()
    idx = torch.randint(3, VOCAB_SIZE, (L,))  # skip special tokens
    return idx

class CopyTask(Dataset):
    """Return (src, tgt) where tgt is identical to src with SOS/EOS."""
    def __init__(self, n_samples: int = 10000, max_len: int = 20):
        self.data = []
        for _ in range(n_samples):
            seq = random_sequence(max_len)
            src = torch.cat([seq, torch.tensor([TOK2IDX[EOS_TOKEN]])])
            tgt = torch.cat([torch.tensor([TOK2IDX[SOS_TOKEN]]), seq, torch.tensor([TOK2IDX[EOS_TOKEN]])])
            self.data.append((src, tgt))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class RecallTask(Dataset):
    """Provide sequence followed by a query marker; target is last item before marker."""
    QUERY_IDX = TOK2IDX["?"] if "?" in TOK2IDX else None
    def __init__(self, n_samples: int = 10000, max_len: int = 20):
        self.data = []
        for _ in range(n_samples):
            seq = random_sequence(max_len)
            marker = torch.tensor([TOK2IDX[EOS_TOKEN]])
            src = torch.cat([seq, marker])
            tgt_token = seq[-1]
            tgt = torch.tensor([tgt_token])
            self.data.append((src, tgt))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)
    pad_idx = TOK2IDX[PAD_TOKEN]
    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        src_padded[i,:s.size(0)] = s
        tgt_padded[i,:t.size(0)] = t
    return src_padded, tgt_padded


# ---------------- Additional tasks -----------------

class ReverseTask(Dataset):
    """Return sequence to be reversed."""
    def __init__(self, n_samples: int=10000, max_len:int=20):
        self.data=[]
        for _ in range(n_samples):
            seq = random_sequence(max_len)
            src = torch.cat([seq, torch.tensor([TOK2IDX[EOS_TOKEN]])])
            tgt = torch.cat([torch.tensor([TOK2IDX[SOS_TOKEN]]), seq.flip(0), torch.tensor([TOK2IDX[EOS_TOKEN]])])
            self.data.append((src,tgt))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]


class SortDigitsTask(Dataset):
    """Digits only; target is digits sorted ascending."""
    def __init__(self, n_samples:int=10000, max_len:int=10):
        digits = [TOK2IDX[c] for c in "0123456789"]
        self.data=[]
        for _ in range(n_samples):
            L=torch.randint(1,max_len+1,(1,)).item()
            idx=torch.tensor(digits)[torch.randperm(10)[:L]]
            src=torch.cat([idx, torch.tensor([TOK2IDX[EOS_TOKEN]])])
            sorted_idx = torch.sort(idx)[0]
            tgt=torch.cat([torch.tensor([TOK2IDX[SOS_TOKEN]]), sorted_idx, torch.tensor([TOK2IDX[EOS_TOKEN]])])
            self.data.append((src,tgt))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]


class ArithmeticTask(Dataset):
    """Add two random 3-digit numbers presented as a sequence 'abc+def'. Target is their sum digits."""
    def __init__(self, n_samples:int=10000):
        self.data=[]
        for _ in range(n_samples):
            a=torch.randint(100,1000,(1,)).item()
            b=torch.randint(100,1000,(1,)).item()
            expr=f"{a:03d}+{b:03d}="
            src=torch.tensor([TOK2IDX[c] for c in expr]+[TOK2IDX[EOS_TOKEN]])
            result=str(a+b)
            tgt=torch.tensor([TOK2IDX[SOS_TOKEN]]+[TOK2IDX[c] for c in result]+[TOK2IDX[EOS_TOKEN]])
            self.data.append((src,tgt))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]
