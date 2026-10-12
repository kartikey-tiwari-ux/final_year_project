"""
C       -- feature-concatenation fusion: frozen DistilBERT (768) + frozen
           ResNet18-on-SHARP-image (512) -> 2-hidden-layer MLP, class-weighted CE.
C-Blur  -- identical architecture/training, but the image embedding comes
           from the BLURRED image (text-illegible) instead.
Comparing C vs. C-Blur is the core novel experiment: if fusion gain survives
even when the image branch can no longer "read" the embedded meme text, the
gain is genuinely cross-modal; if it collapses to the text-only baseline, the
original fusion gain was largely driven by the image branch re-deriving the
same text signal from pixels.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, seed_everything
from metrics import compute_metrics, save_result

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


class FusionMLP(nn.Module):
    def __init__(self, in_dim, n_classes=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_split(image_prefix):
    text_tr = np.load(PROC / "distilbert_train.npy")
    text_va = np.load(PROC / "distilbert_validation.npy")
    text_te = np.load(PROC / "distilbert_test.npy")
    img_tr = np.load(PROC / f"{image_prefix}_train.npy")
    img_va = np.load(PROC / f"{image_prefix}_validation.npy")
    img_te = np.load(PROC / f"{image_prefix}_test.npy")

    Xtr = np.concatenate([text_tr, img_tr], axis=1).astype(np.float32)
    Xva = np.concatenate([text_va, img_va], axis=1).astype(np.float32)
    Xte = np.concatenate([text_te, img_te], axis=1).astype(np.float32)

    ytr = np.load(PROC / "labels_train.npy")
    yva = np.load(PROC / "labels_validation.npy")
    yte = np.load(PROC / "labels_test.npy")
    return Xtr, Xva, Xte, ytr, yva, yte


def run_fusion(image_prefix, name, epochs=40, batch_size=32, lr=1e-3, weight_decay=1e-4):
    Xtr, Xva, Xte, ytr, yva, yte = load_split(image_prefix)

    Xtr_t = torch.from_numpy(Xtr)
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)
    ytr_t = torch.from_numpy(ytr).long()
    yva_t = torch.from_numpy(yva).long()

    classes = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weights = torch.ones(3)
    for c, w in zip(classes, cw):
        class_weights[c] = w

    model = FusionMLP(in_dim=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = Xtr.shape[0]

    best_val_f1 = -1.0
    best_state = None
    history = []
    patience, bad_epochs = 8, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb, weight=class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t).argmax(1).numpy()
        val_metrics = compute_metrics(yva, val_pred)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_macro_f1": val_metrics["macro_f1"],
                         "val_accuracy": val_metrics["accuracy"]})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{name}] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xte_t).argmax(1).numpy()
    test_result = compute_metrics(yte, test_pred)
    test_result["val_macro_f1"] = best_val_f1
    test_result["training_history"] = history

    save_result(name, test_result, RESULTS)
    np.save(PROC / f"preds_{name}.npy", test_pred)
    return test_result


def main():
    seed_everything()
    run_fusion("resnet18_sharp", "C_concat_fusion_sharp")
    seed_everything()
    run_fusion("resnet18_blur", "Cblur_concat_fusion_blur")


if __name__ == "__main__":
    main()
