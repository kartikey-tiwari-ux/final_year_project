"""
B1      -- small 4-layer CNN trained from scratch on 96x96 sharp images
B2      -- frozen ResNet18 (sharp image) + class-weighted Logistic Regression
B2-Blur -- frozen ResNet18 (blurred image) + class-weighted Logistic Regression
           (the embedded-text-leakage ablation)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, seed_everything
from metrics import compute_metrics, save_result

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def run_logreg(feature_prefix, name):
    Xtr = np.load(PROC / f"{feature_prefix}_train.npy")
    Xva = np.load(PROC / f"{feature_prefix}_validation.npy")
    Xte = np.load(PROC / f"{feature_prefix}_test.npy")
    ytr = np.load(PROC / "labels_train.npy")
    yva = np.load(PROC / "labels_validation.npy")
    yte = np.load(PROC / "labels_test.npy")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)

    val_result = compute_metrics(yva, clf.predict(Xva))
    test_pred = clf.predict(Xte)
    test_result = compute_metrics(yte, test_pred)
    test_result["val_macro_f1"] = val_result["macro_f1"]

    save_result(name, test_result, RESULTS)
    np.save(PROC / f"preds_{name}.npy", test_pred)
    return test_result


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 96 -> 48
        x = self.pool(F.relu(self.conv2(x)))   # 48 -> 24
        x = self.pool(F.relu(self.conv3(x)))   # 24 -> 12
        x = self.pool(F.relu(self.conv4(x)))   # 12 -> 6
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def run_cnn_scratch(name="B1_cnn_scratch", epochs=6, batch_size=32, lr=1e-3):
    Xtr = torch.from_numpy(np.load(PROC / "cnn96_train.npy"))
    Xva = torch.from_numpy(np.load(PROC / "cnn96_validation.npy"))
    Xte = torch.from_numpy(np.load(PROC / "cnn96_test.npy"))
    ytr = torch.from_numpy(np.load(PROC / "labels_train.npy")).long()
    yva = torch.from_numpy(np.load(PROC / "labels_validation.npy")).long()
    yte_np = np.load(PROC / "labels_test.npy")
    yte = torch.from_numpy(yte_np).long()

    classes = np.unique(ytr.numpy())
    cw = compute_class_weight("balanced", classes=classes, y=ytr.numpy())
    class_weights = torch.ones(3)
    for c, w in zip(classes, cw):
        class_weights[c] = w

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb, weight=class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_pred = val_logits.argmax(1).numpy()
        val_metrics = compute_metrics(yva.numpy(), val_pred)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_macro_f1": val_metrics["macro_f1"],
                         "val_accuracy": val_metrics["accuracy"]})
        print(f"[B1 CNN] epoch {epoch+1}/{epochs} loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macroF1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xte).argmax(1).numpy()
    test_result = compute_metrics(yte_np, test_pred)
    test_result["val_macro_f1"] = best_val_f1
    test_result["training_history"] = history

    save_result(name, test_result, RESULTS)
    np.save(PROC / f"preds_{name}.npy", test_pred)
    return test_result


def main():
    seed_everything()
    run_cnn_scratch()
    run_logreg("resnet18_sharp", "B2_resnet18_sharp_logreg")
    run_logreg("resnet18_blur", "B2blur_resnet18_blur_logreg")


if __name__ == "__main__":
    main()
