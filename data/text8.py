import os, gzip, urllib.request, hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset

URL = "http://mattmahoney.net/dc/text8.zip"
SHA1 = "3de703932151f5a2c68fc41c5ccf5c76e7d6c366"

class Text8(Dataset):
    """Character-level Text8 dataset. Splits into contiguous sequences."""
    def __init__(self, root: str = 'data', seq_len: int = 256, train: bool = True):
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        zip_path = root_path/"text8.gz"
        if not zip_path.exists():
            print("Downloading text8...")
            urllib.request.urlretrieve(URL, zip_path)
        if self._sha1(zip_path) != SHA1:
            raise RuntimeError("text8 checksum mismatch")
        with gzip.open(zip_path, 'rb') as f:
            raw = f.read().decode('utf-8')
        # integer encode
        vocab = sorted(list(set(raw)))
        self.stoi = {c:i for i,c in enumerate(vocab)}
        ids = torch.tensor([self.stoi[c] for c in raw], dtype=torch.long)
        split = int(0.9*len(ids))
        ids = ids[:split] if train else ids[split:]
        n = (len(ids)-1)//seq_len
        self.data = ids[:n*seq_len+1].view(n, seq_len+1)
        self.seq_len = seq_len
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]
    @staticmethod
    def _sha1(path):
        h=hashlib.sha1()
        with open(path,'rb') as f:
            while True:
                b=f.read(8192)
                if not b: break
                h.update(b)
        return h.hexdigest()

