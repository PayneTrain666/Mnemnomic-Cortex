import os
import torch
import torch.nn as nn
from typing import Optional


def seed_everything(seed: int = 42, deterministic: bool = False):
    import random, numpy as np

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def enable_tensor_cores(model: Optional[nn.Module] = None):
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def optimize_memory_access(model: Optional[nn.Module] = None, deterministic: bool = False):
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = not bool(deterministic)


def distributed_setup(model: nn.Module):
    if not torch.cuda.is_available():
        return model
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model


def fast_pairwise_l2(a: torch.Tensor, b: torch.Tensor):
    """Compute pairwise L2 distances between a and b efficiently.
    a: (B, N, D) or (N, D) ; b: (M, D) -> returns (B, N, M) or (N, M)
    """
    if b.dim() != 2:
        raise ValueError("b must be 2D [M,D]")
    if a.size(-1) != b.size(-1):
        raise ValueError(f"dimension mismatch: a has D={a.size(-1)} vs b has D={b.size(-1)}")
    if b.device != a.device or b.dtype != a.dtype:
        b = b.to(device=a.device, dtype=a.dtype)
    if a.dim() == 2:
        a_sq = (a*a).sum(-1, keepdim=True)            # (N,1)
        b_sq = (b*b).sum(-1).unsqueeze(0)             # (1,M)
        prod = a @ b.t()                               # (N,M)
        d2 = a_sq + b_sq - 2*prod
        return torch.clamp(d2, min=0.0).sqrt()
    elif a.dim() == 3:
        B, N, D = a.shape
        a_sq = (a*a).sum(-1, keepdim=True)            # (B,N,1)
        b_sq = (b*b).sum(-1).view(1,1,-1)             # (1,1,M)
        prod = torch.einsum("bnd,md->bnm", a, b)      # (B,N,M)
        d2 = a_sq + b_sq - 2*prod
        return torch.clamp(d2, min=0.0).sqrt()
    else:
        raise ValueError("a must be 2D or 3D")
