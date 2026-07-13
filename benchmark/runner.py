import torch, csv, os
from torch.utils.data import DataLoader
from benchmark.tasks import CopyTask, RecallTask, collate_fn, VOCAB_SIZE, TOK2IDX
from benchmark.models import get_model


def _align_logits_targets(logits, tgt):
    # Copy-task targets include an extra SOS token in some settings.
    if tgt.size(1) == logits.size(1) + 1:
        L = min(logits.size(1), tgt.size(1) - 1)
        return logits[:, :L, :], tgt[:, 1 : 1 + L]
    L = min(logits.size(1), tgt.size(1))
    return logits[:, :L, :], tgt[:, :L]


def train_epoch(model, loader, criterion, opt, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Check if model returns aux losses
        output = model(src, return_aux_losses=True) if hasattr(model, 'recall_loss_weight') else model(src)
        
        if isinstance(output, tuple):
            logits, aux = output
            recall_loss = aux.get('recall_loss', 0.0)
            cms_loss = aux.get('cms_loss', 0.0)
        else:
            logits = output
            recall_loss = 0.0
            cms_loss = 0.0
        
        logits_aligned, tgt_aligned = _align_logits_targets(logits, tgt)
        logits = logits_aligned.contiguous().view(-1, logits_aligned.size(-1))
        tgt_flat = tgt_aligned.contiguous().view(-1)
        seq_loss = criterion(logits, tgt_flat)
        
        # Combine losses
        extra_loss = 0.0
        if isinstance(recall_loss, torch.Tensor):
            extra_loss = extra_loss + model.recall_loss_weight * recall_loss
        if isinstance(cms_loss, torch.Tensor):
            extra_loss = extra_loss + cms_loss
        loss = seq_loss + extra_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if hasattr(model, "topology_step"):
            model.topology_step(loss.item())
        total_loss += loss.item()
    if hasattr(model, "flush_cms_logger"):
        model.flush_cms_logger()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tok, correct_tok = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src)
            logits_trunc, tgt_trunc = _align_logits_targets(logits, tgt)
            loss = criterion(logits_trunc.reshape(-1, logits.size(-1)), tgt_trunc.reshape(-1))
            # token accuracy (ignore PAD)
            preds = logits_trunc.argmax(-1)
            mask = tgt_trunc != TOK2IDX['<pad>']
            correct_tok += (preds == tgt_trunc).masked_select(mask).sum().item()
            total_tok += mask.sum().item()
            total_loss += loss.item()
    acc = correct_tok / max(1,total_tok)
    return total_loss / len(loader), acc


def run(
    model_name,
    task_cls,
    n_epochs=1,
    batch_size=32,
    device='cpu',
    cms_log_dir=None,
    accept_min_acc: float = -1.0,
    accept_max_loss: float = -1.0,
):
    train_ds = task_cls(2000)
    test_ds  = task_cls(400)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    model_kwargs = {}
    if model_name.lower() == "cortex":
        model_kwargs.update(
            working_memory_fabric="qdt",
            qdt_hardware_profile="single_gpu_8_12gb",
            qdt_qspin_guarded_shadow=True,
            qdt_qspin_live_activation=False,
            qdt_qspin_live_kill_switch_enabled=True,
        )
        if cms_log_dir:
            model_kwargs["cms_log_dir"] = cms_log_dir
    model = get_model(model_name, VOCAB_SIZE, **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOK2IDX['<pad>'])
    results = {}
    for ep in range(n_epochs):
        tr_loss = train_epoch(model, train_loader, criterion, opt, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    results['loss'] = val_loss
    results['acc'] = val_acc
    if float(accept_min_acc) >= 0.0 and float(val_acc) < float(accept_min_acc):
        raise RuntimeError(
            f"Acceptance gate failed for {model_name}/{task_cls.__name__}: "
            f"acc={float(val_acc):.6f} < min={float(accept_min_acc):.6f}"
        )
    if float(accept_max_loss) >= 0.0 and float(val_loss) > float(accept_max_loss):
        raise RuntimeError(
            f"Acceptance gate failed for {model_name}/{task_cls.__name__}: "
            f"loss={float(val_loss):.6f} > max={float(accept_max_loss):.6f}"
        )
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
