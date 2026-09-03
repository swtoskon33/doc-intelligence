"""Fine-tune LayoutLMv3 for token classification on FUNSD.

A real PyTorch training loop: AdamW with weight decay, per-epoch validation, best-model
checkpointing on validation F1, early stopping, and MLflow tracking of the config and
every metric. Seeded for reproducibility.

    python scripts/train_layoutlm.py
    LAYOUTLM_EPOCHS=1 python scripts/train_layoutlm.py     # quick run
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from doc_intelligence.layout.dataset import LABELS, load_funsd
from doc_intelligence.layout.training import (
    TrainConfig,
    build_dataset,
    compute_metrics,
    get_processor,
    load_model,
    pick_device,
    set_seed,
)
from doc_intelligence.tracking.mlflow_tracking import track_run

PATIENCE = 2  # epochs without validation improvement before stopping


def evaluate(model, loader, device):
    """Run the model over a loader and return metrics plus mean loss."""
    model.eval()
    all_logits, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            losses.append(float(out.loss))
            all_logits.append(out.logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    metrics = compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))
    metrics["loss"] = round(float(np.mean(losses)), 4)
    return metrics


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"device: {device} | model: {cfg.model_name}")

    splits = load_funsd(seed=cfg.seed)
    print("splits:", splits.sizes)

    processor = get_processor(cfg.model_name)
    train_ds = build_dataset(processor, splits.train, cfg.max_length)
    val_ds = build_dataset(processor, splits.validation, cfg.max_length)
    test_ds = build_dataset(processor, splits.test, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = load_model(cfg.model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, stale = -1.0, -1, 0
    history = []
    started = time.perf_counter()

    with track_run("doc-intelligence-layoutlmv3", f"funsd-{cfg.epochs}ep") as log:
        log(params={
            "model": cfg.model_name, "dataset": "nielsr/funsd-layoutlmv3",
            "train_size": splits.sizes["train"], "val_size": splits.sizes["validation"],
            "test_size": splits.sizes["test"], "epochs": cfg.epochs,
            "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay, "seed": cfg.seed,
            "max_length": cfg.max_length, "device": device, "num_labels": len(LABELS),
        })

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_losses = []
            for step, batch in enumerate(train_loader, start=1):
                inputs = {k: v.to(device) for k, v in batch.items()}
                out = model(**inputs)
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(float(out.loss))
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                          f"loss {np.mean(epoch_losses[-20:]):.4f}")

            train_loss = float(np.mean(epoch_losses))
            val = evaluate(model, val_loader, device)
            print(f"epoch {epoch}: train_loss {train_loss:.4f} | "
                  f"val_loss {val['loss']} | val_f1 {val['f1']}")
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **{
                k: v for k, v in val.items() if k != "report"}})
            log(metrics={
                "train_loss": train_loss, "val_loss": val["loss"],
                "val_precision": val["precision"], "val_recall": val["recall"],
                "val_f1": val["f1"],
            })

            if val["f1"] > best_f1:
                best_f1, best_epoch, stale = val["f1"], epoch, 0
                model.save_pretrained(out_dir)
                processor.save_pretrained(out_dir)
                print(f"  new best (f1 {best_f1}) -> {out_dir}")
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"  early stopping after epoch {epoch}")
                    break

        # final test evaluation with the best checkpoint
        model = load_model(str(out_dir)).to(device)
        test = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - started
        print(f"TEST: precision {test['precision']} recall {test['recall']} f1 {test['f1']}")
        log(metrics={
            "test_precision": test["precision"], "test_recall": test["recall"],
            "test_f1": test["f1"], "test_loss": test["loss"],
            "best_epoch": best_epoch, "training_seconds": round(elapsed, 1),
        })

    report = {
        "config": cfg.__dict__, "splits": splits.sizes, "history": history,
        "best_epoch": best_epoch, "test": {k: v for k, v in test.items() if k != "report"},
        "per_entity": test["report"], "training_seconds": round(elapsed, 1),
    }
    Path("docs/layoutlm_training.json").write_text(json.dumps(report, indent=2, default=float))
    print("wrote docs/layoutlm_training.json")


if __name__ == "__main__":
    main()
