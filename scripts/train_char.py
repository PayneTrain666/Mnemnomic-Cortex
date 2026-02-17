import argparse, time, math, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from data.text8 import Text8
from benchmark.models import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cortex', choices=['cortex','lstm','transformer'])
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cms_log_dir', default=None)
    p.add_argument('--cms_log_sample_rate', type=float, default=0.1)
    p.add_argument('--cms_log_max_buffer', type=int, default=100000)
    args = p.parse_args()

    train_ds = Text8(seq_len=args.seq_len, train=True)
    test_ds  = Text8(seq_len=args.seq_len, train=False)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  args.batch, shuffle=False)

    vocab_size = len(train_ds.stoi)
    model_kwargs = {}
    if args.model == "cortex" and args.cms_log_dir:
        model_kwargs.update(
            dict(
                cms_log_dir=args.cms_log_dir,
                cms_log_sample_rate=args.cms_log_sample_rate,
                cms_log_max_buffer=args.cms_log_max_buffer,
            )
        )
    model = get_model(args.model, vocab_size, **model_kwargs).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        m = model.train() if train else model.eval()
        total_loss, ntok = 0,0
        with torch.set_grad_enabled(train):
            for src, tgt in loader:
                src, tgt = src.to(args.device), tgt.to(args.device)
                output = model(src, return_aux_losses=True) if train and hasattr(model, "recall_loss_weight") else model(src)
                if isinstance(output, tuple):
                    logits, aux = output
                    recall_loss = aux.get("recall_loss", 0.0)
                    cms_loss = aux.get("cms_loss", 0.0)
                else:
                    logits = output
                    recall_loss = 0.0
                    cms_loss = 0.0
                loss = crit(logits.view(-1, vocab_size), tgt.view(-1))
                if train:
                    if isinstance(recall_loss, torch.Tensor):
                        loss = loss + model.recall_loss_weight * recall_loss
                    if isinstance(cms_loss, torch.Tensor):
                        loss = loss + cms_loss
                    opt.zero_grad(); loss.backward(); opt.step()
                    if hasattr(model, "topology_step"):
                        model.topology_step(loss.item())
                total_loss += loss.item()*tgt.numel()
                ntok += tgt.numel()
        if train and hasattr(model, "flush_cms_logger"):
            model.flush_cms_logger()
        return math.exp(total_loss/ntok)

    for ep in range(1,args.epochs+1):
        train_ppl = run_epoch(train_loader, True)
        test_ppl  = run_epoch(test_loader, False)
        print(f"epoch {ep}: train ppl {train_ppl:.2f}  test ppl {test_ppl:.2f}")

if __name__=='__main__':
    main()

