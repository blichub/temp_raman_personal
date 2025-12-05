"""
Train the irregular-grid transformer classifier without resampling wavenumbers.

This script:
- Loads spectra directly from the SQLite DB via app.utils (no resampling).
- Sorts wavenumbers, normalizes intensities per sample, and keeps variable lengths.
- Uses collate_irregular for padding + masks.
- Saves the model and class mapping to app/model_cache.
"""

import json
import os
from pathlib import Path
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import app.utils as utils
from app.models.spectrum_model import IrregularSpectrumClassifier, collate_irregular
from app.models.irregular_config import MODEL_CFG

DB_PATH = "app/database/microplastics_reference.db"
MODEL_DIR = Path("app/model_cache")
MODEL_PATH = MODEL_DIR / "irregular_spectrum_classifier.pth"
CLASS_MAP_PATH = MODEL_DIR / "irregular_class_map.json"
CONFIG_PATH = MODEL_DIR / "irregular_config.json"


def load_dataset():
    sample_ids = utils.get_all_ids(DB_PATH)
    data = []

    for sid in sample_ids:
        intensities, wns, _ = utils.get_spectrum_data(sid, DB_PATH)
        if not intensities:
            continue

        pairs = sorted(zip(wns, intensities), key=lambda x: x[0])  # ensure order
        wns_sorted, ints_sorted = zip(*pairs)

        wns_t = torch.tensor(wns_sorted, dtype=torch.float32)
        ints_t = torch.tensor(ints_sorted, dtype=torch.float32)

        max_i = torch.max(ints_t).clamp(min=1e-6)
        ints_t = ints_t / max_i

        data.append({"wn": wns_t, "intensity": ints_t, "label": sid})

    return data, sample_ids


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducible split
    random.seed(42)
    torch.manual_seed(42)

    data, ids = load_dataset()
    if not data:
        raise RuntimeError("No spectra loaded from database; aborting training.")

    class_to_idx = {cid: i for i, cid in enumerate(ids)}
    for d in data:
        d["label"] = class_to_idx[d["label"]]

    use_cuda = torch.cuda.is_available() and os.environ.get("USE_CPU", "0") != "1"
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using CUDA device: {gpu_name}")
    else:
        print("CUDA not available or disabled (set USE_CPU=1 to force CPU). Using CPU.")

    # Train/validation split (90/10)
    val_size = max(1, int(0.1 * len(data)))
    train_size = len(data) - val_size
    train_ds, val_ds = random_split(
        data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_irregular(b, pad_value=0.0),
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_irregular(b, pad_value=0.0),
        pin_memory=use_cuda,
    )

    # Fast config (defined in one place for training + inference)
    model_cfg = dict(MODEL_CFG)

    num_classes = len(class_to_idx)
    model = IrregularSpectrumClassifier(
        num_classes=num_classes,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        nlayers=model_cfg["nlayers"],
        num_freqs=model_cfg["num_freqs"],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    epochs = 8
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):
            for x_wn, y_i, mask, labels in loader:
                x_wn = x_wn.to(device, non_blocking=use_cuda)
                y_i = y_i.to(device, non_blocking=use_cuda)
                mask = mask.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)
                logits = model(x_wn, y_i, key_padding_mask=mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total else 0.0
        model.train()
        return acc

    for epoch in range(epochs):
        running = 0.0
        total_steps = len(train_loader)
        print(f"Starting epoch {epoch + 1}/{epochs} ({total_steps} train batches)...")
        for step, (x_wn, y_i, mask, labels) in enumerate(train_loader, start=1):
            x_wn = x_wn.to(device, non_blocking=use_cuda)
            y_i = y_i.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(x_wn, y_i, key_padding_mask=mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.size(0)

            if step % max(1, total_steps // 5) == 0:
                print(f"  Batch {step}/{total_steps} - running_loss_avg: {running / (step * train_loader.batch_size):.4f}")

        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    CLASS_MAP_PATH.write_text(json.dumps(class_to_idx, indent=2))
    CONFIG_PATH.write_text(json.dumps(model_cfg, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class map to {CLASS_MAP_PATH}")
    print(f"Saved model config to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
