"""
Plain-language summary
----------------------
What this file is for: Minimal training-loop sketch with CMS logging and consolidation EMA.
How it fits in the system: Learning aid / sketch more than production trainer.
Status: LOW-USE / sketch
Important notes for non-coders: Prefer copy_task_gpu_train.py for serious GPU runs.
"""

import torch
import torch.nn.functional as F

from mnemonic_cortex.cms_ops import ConsolidationEMAJob, cms_safety_clamp


def train_with_cms_cadence(
    model,
    train_loader,
    optimizer=None,
    num_epochs: int = 1,
    device: str = "cpu",
    consolidation_every: int = 3,
    consolidation_ema: float = 0.9,
):
    """
    Minimal training loop sketch with:
      - CMS logging cadence
      - optional periodic EMA consolidation
      - geometry safety clamp
    """
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Keep CMS geometry constraints sane if available.
            cortex = getattr(model, "cortex", None)
            if cortex is not None and getattr(cortex, "consolidated_lexicon", None) is not None:
                cms_safety_clamp(cortex.consolidated_lexicon)

        # Flush logs each epoch
        log_path = model.flush_cms_logger() if hasattr(model, "flush_cms_logger") else None
        print(f"epoch={epoch+1} wrote_logs={log_path}")

        # Optional periodic consolidation
        if (
            (epoch + 1) % consolidation_every == 0
            and log_path
            and hasattr(model, "cortex")
            and getattr(model.cortex, "consolidated_lexicon", None) is not None
        ):
            recs = torch.load(log_path, map_location="cpu")
            ConsolidationEMAJob(model.cortex.consolidated_lexicon, ema=consolidation_ema).step(recs)
            cms_safety_clamp(model.cortex.consolidated_lexicon)
