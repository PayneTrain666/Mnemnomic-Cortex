import argparse, time
import torch, contextlib
from benchmark.models import get_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cortex')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    model = get_model(args.model, vocab_size=256).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    inp = torch.randint(0,256,(args.batch,args.seq), device=args.device)

    torch.cuda.synchronize() if args.device.startswith('cuda') else None
    t0=time.time()
    for _ in range(args.steps):
        out = model(inp)
        loss = out.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.cuda.synchronize() if args.device.startswith('cuda') else None
    dt=time.time()-t0
    print(f"{args.model} - {args.steps} steps: {dt:.2f}s  => {(dt/args.steps):.4f}s/step")

if __name__=='__main__':
    main()
