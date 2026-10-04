"""Model B1 — CNN/vision baseline trained from scratch (no pretraining).

Per literature review 2.1/2.2, a from-scratch CNN is expected to underperform a
pretrained/transfer-learned model on a dataset this size — that comparison is the
point of this baseline, not a bug if it loses to Model B2.

Trains from the cached uint8 pixel array (src/image_model/extract_pixel_cache.py)
rather than re-decoding JPEGs from disk every epoch — see REPRODUCIBILITY.md for why
(measured ~530ms/image disk I/O on this machine makes repeated per-epoch decoding
take hours; decoding once and training from an in-memory array is the fix).
Run: python -m src.image_model.baseline_cnn
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "pixel_cache_224.npz"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_b1_cnn_scratch.json"
EPOCHS = 3  # reduced from 8 — see REPRODUCIBILITY.md
BATCH_SIZE = 64  # larger batches reduce Python/loop overhead per epoch on CPU
LR = 5e-4  # lowered from 1e-3 — the 1e-3 run showed worsening val metrics epoch over
           # epoch (instability), consistent with too-aggressive updates for this
           # small architecture + strong class-imbalance weighting; see REPRODUCIBILITY.md
TRAIN_RESOLUTION = 96  # downsampled from the 224 used by B2's pretrained ResNet18.
# Rationale (measured, not guessed): at 224x224, two full attempts at this from-scratch
# CNN (one killed for memory, one killed for exceeding a 45-minute background runtime
# limit without finishing 3 epochs) showed conv-forward/backward cost dominating wall
# time on this machine's CPU (no GPU, 4 cores) -- consistent with the back-of-envelope
# cost of a 224x224 first conv layer on unaccelerated CPU kernels. B1 only needs to
# demonstrate the from-scratch-vs-pretrained comparison (Literature Review 2.1/2.2),
# not match B2's input resolution, so training at 96x96 (≈5.4x fewer pixels) is a
# scientifically reasonable scope reduction, documented here per development rule 13.

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CachedPixelDataset(Dataset):
    """Wraps an in-memory uint8 (N, H, W, 3) array (224x224, from extract_pixel_cache.py);
    downsamples to TRAIN_RESOLUTION and normalizes per-sample on the fly."""

    def __init__(self, pixels_uint8: np.ndarray, labels: list[int]):
        assert len(pixels_uint8) == len(labels)
        self.pixels = pixels_uint8
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.pixels[idx]).permute(2, 0, 1).float() / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(TRAIN_RESOLUTION, TRAIN_RESOLUTION), mode="bilinear", align_corners=False
        ).squeeze(0)
        return img, self.labels[idx]


class SmallCNN(nn.Module):
    """A few conv+pool blocks — intentionally small, trained from scratch."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 224->112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 112->56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 56->28
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    all_preds, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


def main() -> None:
    set_seed(42)
    torch.set_num_threads(os.cpu_count() or 4)  # default was half the cores on this 4-core machine
    device = torch.device("cpu")
    df = load_manifest()

    print(f"Loading pixel cache from {PIXEL_CACHE_PATH}...")
    cache = np.load(PIXEL_CACHE_PATH)
    cache_ids = list(cache["ids"])
    pixels = cache["pixels"]  # (N, 224, 224, 3) uint8
    id_to_row = {sid: i for i, sid in enumerate(cache_ids)}

    df["px_row"] = df["id"].map(id_to_row)
    n_before = len(df)
    df = df[df["px_row"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Note: {n_dropped} manifest samples missing from pixel cache — excluded.")

    def make_loader(split_name, shuffle):
        sub = df[df["split"] == split_name]
        rows = sub["px_row"].astype(int).values
        labels = sub["label"].map(LABEL_TO_INT).tolist()
        ds = CachedPixelDataset(pixels[rows], labels)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle), labels

    train_loader, labels_train = make_loader("train", shuffle=True)
    val_loader, _ = make_loader("val", shuffle=False)
    test_loader, _ = make_loader("test", shuffle=False)
    print(f"train={len(labels_train)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    class_counts = np.bincount(labels_train, minlength=3)
    class_weights = torch.tensor(len(labels_train) / (3 * class_counts), dtype=torch.float32)
    print(f"Class counts {class_counts}, loss weights {class_weights.tolist()}")

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        val_metrics = compute_metrics(val_labels, val_preds)
        print(
            f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_macro_f1={val_metrics['f1_macro']:.4f}  val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s. Best val macro F1: {best_val_f1:.4f}")

    model.load_state_dict(best_state)
    _, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    _, test_preds, test_labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)

    val_metrics = compute_metrics(val_labels, val_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    print_summary("Model B1 (CNN from scratch) — Validation", val_metrics)
    print_summary("Model B1 (CNN from scratch) — Test", test_metrics)

    save_results(
        {
            "model": "B1_cnn_scratch",
            "config": {
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "seed": 42,
                "dropped_samples": int(n_dropped), "train_wall_clock_seconds": elapsed,
            },
            "val": val_metrics,
            "test": test_metrics,
        },
        RESULTS_PATH,
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
