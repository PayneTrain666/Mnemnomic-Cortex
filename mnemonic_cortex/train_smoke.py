import torch
import torch.nn as nn
import torch.optim as optim
from .cortex import EnhancedMnemonicCortex
from .utils import enable_tensor_cores, optimize_memory_access, seed_everything

def smoke_run(device=None):
    seed_everything(42)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    B,S,d_in,d_out = 8, 5, 128, 128
    cortex = EnhancedMnemonicCortex(input_dim=d_in, output_dim=d_out, fusion='cross_attn').to(device)
    enable_tensor_cores(cortex); optimize_memory_access(cortex)

    x = torch.randn(B, S, d_in, device=device)
    ctx = torch.randn(B, d_in, device=device)

    y_proc = cortex(x, ctx, operation='process')    # (B,S,d_in)
    y_ret  = cortex(x, ctx, operation='retrieve')   # (B,d_in)
    cortex(x, ctx, operation='consolidate')         # None
    assert y_proc.shape == (B,S,d_in)
    assert y_ret.shape  == (B,d_in)
    return y_proc.detach().cpu().shape, y_ret.detach().cpu().shape

def tiny_train_step(steps=5, device=None):
    seed_everything(123)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    B,S,d_in,d_out = 16, 7, 128, 128
    model = EnhancedMnemonicCortex(input_dim=d_in, output_dim=d_out).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for t in range(steps):
        x = torch.randn(B, S, d_in, device=device)
        ctx = torch.randn(B, d_in, device=device)
        target = ctx  # pretend target for retrieval

        model.train()
        _ = model(x, ctx, operation='process')
        y_ret = model(x, ctx, operation='retrieve')

        loss = loss_fn(y_ret, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return float(loss.detach().cpu())

if __name__ == "__main__":
    proc_shape, ret_shape = smoke_run()
    print("Smoke OK:", proc_shape, ret_shape)
    final_loss = tiny_train_step(steps=5)
    print("Tiny train final loss:", final_loss)
