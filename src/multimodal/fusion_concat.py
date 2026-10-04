"""Model C — primary fusion: feature-level concatenation of frozen DistilBERT text
embeddings + frozen ResNet18 image embeddings, with an MLP classification head
trained jointly on the combined 1280-dim vector. See PROJECT_ARCHITECTURE.md.

Requires both embedding caches already built:
  python -m src.text_model.extract_distilbert_embeddings
  python -m src.image_model.extract_resnet_embeddings
Run: python -m src.multimodal.fusion_concat
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz"
IMAGE_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_c_concat_fusion.json"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3


class FusionMLP(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, num_classes: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, text_emb, image_emb):
        x = torch.cat([text_emb, image_emb], dim=1)
        return self.net(x)


def build_aligned_dataset(df, text_ids, text_emb, image_ids, image_emb):
    """Only samples present in BOTH embedding caches can be used for fusion —
    drop (and report) anything missing from either side."""
    text_id_to_row = {sid: i for i, sid in enumerate(text_ids)}
    image_id_to_row = {sid: i for i, sid in enumerate(image_ids)}
    df = df.copy()
    df["has_text"] = df["id"].isin(text_id_to_row)
    df["has_image"] = df["id"].isin(image_id_to_row)
    n_before = len(df)
    df = df[df["has_text"] & df["has_image"]].copy()
    n_dropped = n_before - len(df)
    df["text_row"] = df["id"].map(text_id_to_row)
    df["image_row"] = df["id"].map(image_id_to_row)
    return df, n_dropped


def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    for text_x, image_x, labels in loader:
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(text_x, image_x)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * text_x.size(0)
        all_preds.extend(logits.argmax(1).tolist())
        all_labels.extend(labels.tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


def train_fusion_model(df, text_emb_all, image_emb_all, verbose: bool = True):
    """Shared training routine (same seed, same best-val-checkpoint selection) used
    both by this script's main() and by generate_predictions.py — kept as one function
    so the two never silently diverge into reporting different numbers for 'Model C'."""
    set_seed(42)

    def make_loader(split_name, shuffle):
        sub = df[df["split"] == split_name]
        text_x = torch.tensor(text_emb_all[sub["text_row"].astype(int).values], dtype=torch.float32)
        image_x = torch.tensor(image_emb_all[sub["image_row"].astype(int).values], dtype=torch.float32)
        y = torch.tensor(sub["label"].map(LABEL_TO_INT).tolist(), dtype=torch.long)
        ds = TensorDataset(text_x, image_x, y)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle), y

    train_loader, y_train = make_loader("train", shuffle=True)
    val_loader, y_val = make_loader("val", shuffle=False)
    test_loader, y_test = make_loader("test", shuffle=False)

    class_counts = np.bincount(y_train.numpy(), minlength=3)
    class_weights = torch.tensor(len(y_train) / (3 * class_counts), dtype=torch.float32)
    if verbose:
        print(f"Train class counts {class_counts}, loss weights {class_weights.tolist()}")

    model = FusionMLP(text_dim=text_emb_all.shape[1], image_dim=image_emb_all.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1, best_state = -1.0, None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, train=False)
        val_metrics = compute_metrics(val_labels, val_preds)
        if verbose:
            print(
                f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                f"  val_macro_f1={val_metrics['f1_macro']:.4f}  val_acc={val_metrics['accuracy']:.4f}"
            )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - start
    if verbose:
        print(f"Training done in {elapsed:.1f}s. Best val macro F1: {best_val_f1:.4f}")

    model.load_state_dict(best_state)
    return model, val_loader, test_loader, elapsed, best_val_f1


def main() -> None:
    df = load_manifest()

    text_cache = np.load(TEXT_EMB_PATH)
    image_cache = np.load(IMAGE_EMB_PATH)
    df, n_dropped = build_aligned_dataset(
        df, text_cache["ids"], text_cache["embeddings"], image_cache["ids"], image_cache["embeddings"]
    )
    if n_dropped:
        print(f"Note: {n_dropped} samples dropped (missing from text and/or image embedding cache).")

    text_emb_all = text_cache["embeddings"]
    image_emb_all = image_cache["embeddings"]

    model, val_loader, test_loader, elapsed, best_val_f1 = train_fusion_model(df, text_emb_all, image_emb_all)
    optimizer = torch.optim.Adam(model.parameters())  # unused placeholder for run_epoch's signature below
    criterion = nn.CrossEntropyLoss()
    _, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, train=False)
    _, test_preds, test_labels = run_epoch(model, test_loader, optimizer, criterion, train=False)

    val_metrics = compute_metrics(val_labels, val_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    print_summary("Model C (Concat Fusion MLP) — Validation", val_metrics)
    print_summary("Model C (Concat Fusion MLP) — Test", test_metrics)

    save_results(
        {
            "model": "C_concat_fusion_mlp",
            "config": {
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "seed": 42,
                "text_dim": int(text_emb_all.shape[1]), "image_dim": int(image_emb_all.shape[1]),
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
