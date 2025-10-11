import torch, csv, os
from torch.utils.data import DataLoader
from benchmark.tasks import CopyTask, RecallTask, collate_fn, VOCAB_SIZE, TOK2IDX
from benchmark.models import get_model


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
    total_tok, correct_tok = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src)
            L = min(logits.size(1), tgt.size(1))
            logits_trunc = logits[:,:L,:].contiguous()
            tgt_trunc = tgt[:,:L]
            loss = criterion(logits_trunc.view(-1, logits.size(-1)), tgt_trunc.view(-1))
            # token accuracy (ignore PAD)
            preds = logits_trunc.argmax(-1)
            mask = tgt_trunc != TOK2IDX['<pad>']
            correct_tok += (preds == tgt_trunc).masked_select(mask).sum().item()
            total_tok += mask.sum().item()
            total_loss += loss.item()
    acc = correct_tok / max(1,total_tok)
    return total_loss / len(loader), acc


def run(model_name, task_cls, n_epochs=1, batch_size=32, device='cpu'):
    train_ds = task_cls(2000)
    test_ds  = task_cls(400)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    model = get_model(model_name, VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOK2IDX['<pad>'])
    results = {}
    for ep in range(n_epochs):
        tr_loss = train_epoch(model, train_loader, criterion, opt, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    results['loss'] = val_loss
    results['acc'] = val_acc
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = {
        'copy': CopyTask,
        'recall': RecallTask,
    }
    models = ['cortex', 'lstm', 'transformer']
    results = []
    for m in models:
        for tname, tcls in tasks.items():
            res = run(model_name=m, task_cls=tcls, n_epochs=1, device=device)
            res.update({'model': m, 'task': tname})
            results.append(res)
            print(f"{m}/{tname}: loss={res['loss']:.3f} acc={res['acc']:.3f}")

    # save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','task','loss','acc'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
