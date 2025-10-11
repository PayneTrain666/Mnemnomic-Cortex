import os
import torch
import torch.nn as nn
from typing import Optional

def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enable_tensor_cores(model: Optional[nn.Module] = None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def optimize_memory_access(model: Optional[nn.Module] = None):
    torch.backends.cudnn.benchmark = True

def distributed_setup(model: nn.Module):
    if not torch.cuda.is_available():
        return model
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        from torch.distributed import init_process_group
        init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model

def fast_pairwise_l2(a: torch.Tensor, b: torch.Tensor):
    """Compute pairwise L2 distances between a and b efficiently.
    a: (B, N, D) or (N, D) ; b: (M, D) -> returns (B, N, M) or (N, M)
    """
    if a.dim() == 2:
        # (N,D) vs (M,D) -> (N,M)
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
