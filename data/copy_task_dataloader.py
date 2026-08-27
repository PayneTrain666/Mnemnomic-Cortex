"""
Copy-task dataloader with extensive hyperparameters and sinusoidal encoder/decoder
for positional mastery on sequence-copy benchmarks.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from benchmark.tasks import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, TOK2IDX, VOCAB_SIZE


def sinusoidal_positional_encoding(
    length: int,
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
) -> torch.Tensor:
    """Standard sinusoidal PE [length, dim]."""
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = max(1, dim // 2)
    div = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-math.log(float(base)) / max(1, half - 1))
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


@dataclass
class CopyTaskDataConfig:
    """Extensive hyperparameters for copy-task data generation and loading."""

    # Dataset size / splits
    n_samples: int = 10000
    val_ratio: float = 0.1
    train: bool = True
    seed: int = 42

    # Sequence geometry
    min_len: int = 1
    max_len: int = 20
    fixed_len: int = 0
    curriculum_enabled: bool = False
    curriculum_start_len: int = 4
    curriculum_end_len: int = 20
    curriculum_ramp_epochs: int = 10
    curriculum_hold_epochs: int = 0
    # When current curriculum max > mix anchor, this fraction of samples are drawn
    # at the short/anchor length so long-ramp epochs do not starve short reverse.
    curriculum_short_mix_prob: float = 0.0
    curriculum_mix_anchor_len: int = 0

    # Task variant: forward copy, reverse copy, or mixed per sample
    task_mode: str = "copy"  # copy | reverse | mixed
    reverse_task_prob: float = 0.5

    # Vocabulary / tokenization
    vocab_mode: str = "full"  # full | alnum | digits | custom
    custom_symbols: Tuple[str, ...] = ()
    include_sos_in_target: bool = True
    include_eos_in_source: bool = True
    pad_token_id: int = field(default_factory=lambda: TOK2IDX[PAD_TOKEN])

    # Augmentation
    noise_prob: float = 0.0
    replace_prob: float = 0.0
    repeat_factor: int = 1
    delayed_copy_gap: int = 0
    delimiter_token: str = ""
    reverse_source: bool = False

    # DataLoader
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # Sinusoidal mastery encoder
    enable_sinusoidal_encoder: bool = True
    encoder_d_model: int = 128
    encoder_layers: int = 2
    encoder_heads: int = 4
    encoder_dropout: float = 0.1
    encoder_max_positions: int = 512
    encoder_learned_scale: bool = True
    encoder_base: float = 10000.0

    # Sinusoidal mastery decoder
    enable_sinusoidal_decoder: bool = True
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_dropout: float = 0.1
    decoder_ff_mult: float = 4.0
    mastery_loss_weight: float = 0.25
    teacher_forcing: bool = True

    # Collate extras
    return_position_ids: bool = True
    return_sinusoidal_features: bool = False
    align_target_with_logits: bool = True

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.min_len <= 0 or self.max_len <= 0:
            raise ValueError("min_len and max_len must be positive")
        if self.min_len > self.max_len:
            raise ValueError("min_len cannot exceed max_len")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.vocab_mode not in {"full", "alnum", "digits", "custom"}:
            raise ValueError("vocab_mode must be full|alnum|digits|custom")
        if str(self.task_mode).strip().lower() not in {"copy", "reverse", "mixed"}:
            raise ValueError("task_mode must be copy|reverse|mixed")
        if not 0.0 <= float(self.reverse_task_prob) <= 1.0:
            raise ValueError("reverse_task_prob must be in [0, 1]")
        if self.encoder_d_model % max(1, self.encoder_heads) != 0:
            raise ValueError("encoder_d_model must be divisible by encoder_heads")
        if self.delayed_copy_gap < 0:
            raise ValueError("delayed_copy_gap must be non-negative")
        if not 0.0 <= float(self.curriculum_short_mix_prob) <= 1.0:
            raise ValueError("curriculum_short_mix_prob must be in [0, 1]")

    def curriculum_length(self, epoch: int = 1) -> int:
        if int(self.fixed_len) > 0:
            return int(self.fixed_len)
        if not self.curriculum_enabled:
            return int(self.max_len)
        ramp = max(1, int(self.curriculum_ramp_epochs))
        hold = max(0, int(self.curriculum_hold_epochs))
        ep = max(1, int(epoch))
        if ep <= hold:
            cur = int(self.curriculum_start_len)
        else:
            t = min(1.0, (ep - hold - 1) / float(ramp))
            cur = int(
                round(
                    self.curriculum_start_len
                    + t * (self.curriculum_end_len - self.curriculum_start_len)
                )
            )
        return int(max(self.min_len, min(self.max_len, cur)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_symbol_indices(cfg: CopyTaskDataConfig) -> List[int]:
    if cfg.vocab_mode == "digits":
        symbols = [c for c in "0123456789" if c in TOK2IDX]
    elif cfg.vocab_mode == "alnum":
        symbols = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789" if c in TOK2IDX]
    elif cfg.vocab_mode == "custom":
        symbols = [s for s in cfg.custom_symbols if s in TOK2IDX]
    else:
        symbols = [k for k in TOK2IDX if k not in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}]
    if not symbols:
        raise ValueError("resolved symbol set is empty")
    return [TOK2IDX[s] for s in symbols]


def resolve_copy_task_variant(cfg: CopyTaskDataConfig, rng: random.Random) -> str:
    """Resolve per-sample task variant from config (copy, reverse, or mixed draw)."""
    mode = str(cfg.task_mode).strip().lower()
    if mode == "mixed":
        return "reverse" if rng.random() < float(cfg.reverse_task_prob) else "copy"
    if mode == "reverse":
        return "reverse"
    return "copy"


def _target_content_for_task(seq: torch.Tensor, task_variant: str) -> torch.Tensor:
    if str(task_variant).strip().lower() == "reverse":
        return torch.flip(seq, dims=[0])
    return seq


def _sample_sequence(cfg: CopyTaskDataConfig, symbol_ids: Sequence[int], rng: random.Random) -> torch.Tensor:
    if int(cfg.fixed_len) > 0:
        length = int(cfg.fixed_len)
    else:
        length = rng.randint(int(cfg.min_len), int(cfg.max_len))
    base = [symbol_ids[rng.randrange(len(symbol_ids))] for _ in range(length)]
    if cfg.reverse_source:
        base = list(reversed(base))
    if int(cfg.repeat_factor) > 1:
        base = base * int(cfg.repeat_factor)
    seq = torch.tensor(base, dtype=torch.long)
    if cfg.noise_prob > 0.0:
        mask = torch.rand(seq.numel()) < float(cfg.noise_prob)
        if mask.any():
            noise = torch.tensor([symbol_ids[rng.randrange(len(symbol_ids))] for _ in range(int(mask.sum()))])
            seq[mask] = noise
    if cfg.replace_prob > 0.0:
        for i in range(seq.numel()):
            if rng.random() < float(cfg.replace_prob):
                seq[i] = symbol_ids[rng.randrange(len(symbol_ids))]
    return seq


class CopyTaskDataset(Dataset):
    """Configurable copy-task dataset with optional delay/delimiter augmentation."""

    def __init__(self, cfg: CopyTaskDataConfig, *, epoch: int = 1, regenerate: bool = True):
        cfg.validate()
        self.cfg = cfg
        self.epoch = int(epoch)
        self.symbol_ids = _resolve_symbol_indices(cfg)
        self._rng = random.Random(int(cfg.seed) + int(epoch) * 10007)
        self.data: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = []
        if regenerate:
            self.regenerate(epoch=epoch)

    def regenerate(self, *, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.epoch = int(epoch)
            self._rng = random.Random(int(self.cfg.seed) + self.epoch * 10007)
        cur_max = self.cfg.curriculum_length(self.epoch)
        mix_prob = float(self.cfg.curriculum_short_mix_prob)
        anchor = int(self.cfg.curriculum_mix_anchor_len) or int(self.cfg.curriculum_start_len)
        anchor = max(int(self.cfg.min_len), min(int(cur_max), int(anchor)))
        self.data = []
        delim_id = TOK2IDX.get(self.cfg.delimiter_token) if self.cfg.delimiter_token else None
        gap = int(self.cfg.delayed_copy_gap)
        for _ in range(int(self.cfg.n_samples)):
            sample_max = int(cur_max)
            if mix_prob > 0.0 and int(cur_max) > int(anchor) and self._rng.random() < mix_prob:
                # Prefer exact short mastery replay; occasionally mid lengths below cur_max.
                if self._rng.random() < 0.75:
                    sample_max = int(anchor)
                else:
                    sample_max = int(self._rng.randint(int(anchor), max(int(anchor), int(cur_max) - 1)))
            local_cfg = CopyTaskDataConfig(
                **{**self.cfg.to_dict(), "max_len": sample_max, "min_len": min(int(self.cfg.min_len), sample_max)}
            )
            seq = _sample_sequence(local_cfg, self.symbol_ids, self._rng)
            task_variant = resolve_copy_task_variant(self.cfg, self._rng)
            tgt_content = _target_content_for_task(seq, task_variant)
            src_parts = [seq]
            if delim_id is not None:
                src_parts.append(torch.tensor([delim_id], dtype=torch.long))
            if gap > 0:
                src_parts.append(torch.full((gap,), int(self.cfg.pad_token_id), dtype=torch.long))
            src = torch.cat(src_parts, dim=0)
            if self.cfg.include_eos_in_source:
                src = torch.cat([src, torch.tensor([TOK2IDX[EOS_TOKEN]], dtype=torch.long)])
            tgt_parts = []
            if self.cfg.include_sos_in_target:
                tgt_parts.append(torch.tensor([TOK2IDX[SOS_TOKEN]], dtype=torch.long))
            tgt_parts.append(tgt_content)
            tgt_parts.append(torch.tensor([TOK2IDX[EOS_TOKEN]], dtype=torch.long))
            tgt = torch.cat(tgt_parts, dim=0)
            meta = {
                "content_len": int(seq.numel()),
                "tgt_content_len": int(tgt_content.numel()),
                "src_len": int(src.numel()),
                "tgt_len": int(tgt.numel()),
                "epoch": int(self.epoch),
                "curriculum_len": int(cur_max),
                "task_mode": str(task_variant),
                "task_variant": str(task_variant),
            }
            self.data.append((src, tgt, meta))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        src, tgt, meta = self.data[idx]
        return src, tgt, meta


class SinusoidalSequenceEncoder(nn.Module):
    """Sinusoidal positional encoder stack for copy-task mastery."""

    def __init__(self, cfg: CopyTaskDataConfig, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = int(cfg.encoder_d_model)
        heads = int(cfg.encoder_heads)
        self.token_embed = nn.Embedding(vocab_size, d)
        self.pos_scale = nn.Parameter(torch.ones(1)) if cfg.encoder_learned_scale else None
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=max(d, int(d * 2)),
            dropout=float(cfg.encoder_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(cfg.encoder_layers)))
        self.out_norm = nn.LayerNorm(d)

    def positional_encoding(self, length: int, device, dtype) -> torch.Tensor:
        pe = sinusoidal_positional_encoding(
            length,
            self.cfg.encoder_d_model,
            device=device,
            dtype=dtype,
            base=float(self.cfg.encoder_base),
        )
        if self.pos_scale is not None:
            pe = pe * self.pos_scale
        return pe

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)
        pe = self.positional_encoding(token_ids.size(1), x.device, x.dtype).unsqueeze(0)
        x = x + pe
        x = self.encoder(x)
        return self.out_norm(x)


class SinusoidalMasteryDecoder(nn.Module):
    """Sinusoidal decoder head for copy-target mastery."""

    def __init__(self, cfg: CopyTaskDataConfig, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = int(cfg.encoder_d_model)
        heads = int(cfg.decoder_heads)
        ff = max(d, int(d * float(cfg.decoder_ff_mult)))
        self.token_embed = nn.Embedding(vocab_size, d)
        self.pos_scale = nn.Parameter(torch.ones(1)) if cfg.encoder_learned_scale else None
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=ff,
            dropout=float(cfg.decoder_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=max(1, int(cfg.decoder_layers)))
        self.head = nn.Linear(d, vocab_size)
        self.out_norm = nn.LayerNorm(d)

    def positional_encoding(self, length: int, device, dtype) -> torch.Tensor:
        pe = sinusoidal_positional_encoding(
            length,
            self.cfg.encoder_d_model,
            device=device,
            dtype=dtype,
            base=float(self.cfg.encoder_base),
        )
        if self.pos_scale is not None:
            pe = pe * self.pos_scale
        return pe

    def forward(
        self,
        memory: torch.Tensor,
        tgt_ids: torch.Tensor,
        *,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.token_embed(tgt_ids)
        pe = self.positional_encoding(tgt_ids.size(1), tgt.device, tgt.dtype).unsqueeze(0)
        tgt = tgt + pe
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=tgt.device)
        dec = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.head(self.out_norm(dec))


class CopyTaskMasteryModel(nn.Module):
    """Standalone sinusoidal encoder/decoder for copy-task mastery training."""

    def __init__(self, cfg: CopyTaskDataConfig, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.cfg = cfg
        self.encoder = SinusoidalSequenceEncoder(cfg, vocab_size=vocab_size)
        self.decoder = SinusoidalMasteryDecoder(cfg, vocab_size=vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        *,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder(src_ids)
        if self.cfg.teacher_forcing:
            dec_in = tgt_ids
        else:
            dec_in = torch.roll(tgt_ids, shifts=1, dims=1)
            dec_in[:, 0] = TOK2IDX[SOS_TOKEN]
        return self.decoder(memory, dec_in, memory_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)

    def mastery_loss(
        self,
        logits: torch.Tensor,
        tgt_ids: torch.Tensor,
        *,
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        ignore = TOK2IDX[PAD_TOKEN] if ignore_index is None else int(ignore_index)
        if self.cfg.align_target_with_logits and tgt_ids.size(1) == logits.size(1) + 1:
            logits = logits[:, : tgt_ids.size(1) - 1, :]
            tgt_ids = tgt_ids[:, 1:]
        elif tgt_ids.size(1) != logits.size(1):
            n = min(tgt_ids.size(1), logits.size(1))
            logits = logits[:, :n, :]
            tgt_ids = tgt_ids[:, :n]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_ids.reshape(-1),
            ignore_index=ignore,
            label_smoothing=float(label_smoothing),
        )


def _pad_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return ids.eq(int(pad_id))


def copy_task_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]], cfg: CopyTaskDataConfig):
    srcs, tgts, metas = zip(*batch)
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)
    pad_id = int(cfg.pad_token_id)
    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    src_lens = torch.zeros(len(batch), dtype=torch.long)
    tgt_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : s.size(0)] = s
        tgt[i, : t.size(0)] = t
        src_lens[i] = int(s.size(0))
        tgt_lens[i] = int(t.size(0))

    out: Dict[str, Any] = {
        "src": src,
        "tgt": tgt,
        "src_lens": src_lens,
        "tgt_lens": tgt_lens,
        "meta": list(metas),
    }
    if cfg.return_position_ids:
        out["src_pos"] = torch.arange(max_src, dtype=torch.long).unsqueeze(0).repeat(len(batch), 1)
        out["tgt_pos"] = torch.arange(max_tgt, dtype=torch.long).unsqueeze(0).repeat(len(batch), 1)
    if cfg.return_sinusoidal_features:
        d = int(cfg.encoder_d_model)
        out["src_sin"] = sinusoidal_positional_encoding(max_src, d).unsqueeze(0).repeat(len(batch), 1, 1)
        out["tgt_sin"] = sinusoidal_positional_encoding(max_tgt, d).unsqueeze(0).repeat(len(batch), 1, 1)
    return out


def build_copy_task_dataset(
    cfg: CopyTaskDataConfig,
    *,
    epoch: int = 1,
    split: str = "train",
) -> Dataset:
    cfg.validate()
    ds = CopyTaskDataset(cfg, epoch=epoch)
    if float(cfg.val_ratio) <= 0.0 or split == "all":
        return ds
    n_val = max(1, int(len(ds) * float(cfg.val_ratio)))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(int(cfg.seed))
    perm = torch.randperm(len(ds), generator=gen).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if split == "train":
        return Subset(ds, train_idx)
    if split == "val":
        return Subset(ds, val_idx)
    raise ValueError("split must be train|val|all")


def build_copy_task_dataloader(
    cfg: CopyTaskDataConfig,
    *,
    epoch: int = 1,
    split: str = "train",
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    cfg.validate()
    ds = build_copy_task_dataset(cfg, epoch=epoch, split=split)
    shuffle = bool(cfg.shuffle) if split == "train" else False
    collate = collate_fn or (lambda batch: copy_task_collate_fn(batch, cfg))
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "shuffle": shuffle,
        "drop_last": bool(cfg.drop_last) if split == "train" else False,
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
        "collate_fn": collate,
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    return DataLoader(ds, **loader_kwargs)


def align_logits_targets(
    logits: torch.Tensor,
    tgt: torch.Tensor,
    *,
    align: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not align:
        n = min(logits.size(1), tgt.size(1))
        return logits[:, :n, :], tgt[:, :n]
    if tgt.size(1) == logits.size(1) + 1:
        n = min(logits.size(1), tgt.size(1) - 1)
        return logits[:, :n, :], tgt[:, 1 : 1 + n]
    n = min(logits.size(1), tgt.size(1))
    return logits[:, :n, :], tgt[:, :n]


def config_from_namespace(ns: Any) -> CopyTaskDataConfig:
    """Build config from argparse namespace or similar mapping object."""
    fields = CopyTaskDataConfig.__dataclass_fields__.keys()
    data = {k: getattr(ns, k) for k in fields if hasattr(ns, k)}
    return CopyTaskDataConfig(**data)
