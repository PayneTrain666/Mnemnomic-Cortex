import torch, time, csv
from torch.utils.data import DataLoader
from benchmark.tasks import CopyTask, RecallTask, collate_fn, VOCAB_SIZE, TOK2IDX
from benchmark.models import CortexSeqModel


def train_epoch(model, loader, criterion, opt, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        L = min(logits.size(1), tgt.size(1))
        logits = logits[:,:L,:].contiguous().view(-1, logits.size(-1))
        tgt_flat = tgt[:,:L].contiguous().view(-1)
        loss = criterion(logits, tgt_flat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src)
            L = min(logits.size(1), tgt.size(1))
            loss = criterion(logits[:,:L,:].contiguous().view(-1, logits.size(-1)), tgt[:,:L].contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


def run(task_cls, n_epochs=1, batch_size=32, device='cpu'):
    train_ds = task_cls(2000)
    test_ds  = task_cls(400)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    model = CortexSeqModel(VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOK2IDX['<pad>'])
    results = {}
    for ep in range(n_epochs):
        tr_loss = train_epoch(model, train_loader, criterion, opt, device)
        val_loss = evaluate(model, test_loader, criterion, device)
    results['loss'] = val_loss
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmarks = {
        'copy': CopyTask,
        'recall': RecallTask,
    }
    for name, task in benchmarks.items():
        res = run(task_cls=task, n_epochs=1, device=device)
        print(name, res)

if __name__ == '__main__':
    main()
