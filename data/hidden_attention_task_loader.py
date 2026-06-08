from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from .copy_task_dataloader import CopyTaskDataConfig, build_copy_task_dataloader


@dataclass
class HiddenAttentionTaskLoaderConfig:
    """
    Curated task config for stressing hidden-layer attention interfaces.

    The task blends copy + reverse sequence reconstruction with delay and mild
    noise so WM/LTM fusion and hidden attention pathways are exercised.
    """

    train_samples: int = 2048
    val_samples: int = 512
    batch_size: int = 32
    min_len: int = 4
    max_len: int = 24
    delayed_copy_gap: int = 3
    task_mode: str = "mixed"
    reverse_task_prob: float = 0.45
    noise_prob: float = 0.02
    replace_prob: float = 0.01
    repeat_factor: int = 1
    seed: int = 1337
    num_workers: int = 0
    pin_memory: bool = True
    vocab_mode: str = "full"
    encoder_d_model: int = 96
    encoder_layers: int = 2
    encoder_heads: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 8

    def to_copy_cfg(self, *, n_samples: int, shuffle: bool) -> CopyTaskDataConfig:
        return CopyTaskDataConfig(
            n_samples=int(n_samples),
            min_len=int(self.min_len),
            max_len=int(self.max_len),
            batch_size=int(self.batch_size),
            seed=int(self.seed),
            delayed_copy_gap=int(self.delayed_copy_gap),
            task_mode=str(self.task_mode),
            reverse_task_prob=float(self.reverse_task_prob),
            noise_prob=float(self.noise_prob),
            replace_prob=float(self.replace_prob),
            repeat_factor=int(self.repeat_factor),
            num_workers=int(self.num_workers),
            pin_memory=bool(self.pin_memory),
            shuffle=bool(shuffle),
            vocab_mode=str(self.vocab_mode),
            val_ratio=0.0,
            enable_sinusoidal_encoder=True,
            enable_sinusoidal_decoder=True,
            encoder_d_model=int(self.encoder_d_model),
            encoder_layers=int(self.encoder_layers),
            encoder_heads=int(self.encoder_heads),
            decoder_layers=int(self.decoder_layers),
            decoder_heads=int(self.decoder_heads),
            return_position_ids=True,
            return_sinusoidal_features=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_hidden_attention_task_loaders(
    cfg: HiddenAttentionTaskLoaderConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    train_cfg = cfg.to_copy_cfg(n_samples=int(cfg.train_samples), shuffle=True)
    val_cfg = cfg.to_copy_cfg(n_samples=int(cfg.val_samples), shuffle=False)
    train_loader = build_copy_task_dataloader(train_cfg, split="all", epoch=1)
    val_loader = build_copy_task_dataloader(val_cfg, split="all", epoch=2)
    meta = {
        "task": "hidden_attention_copy_reverse_delayed",
        "train_cfg": train_cfg.to_dict(),
        "val_cfg": val_cfg.to_dict(),
    }
    return train_loader, val_loader, meta
