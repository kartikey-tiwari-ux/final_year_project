"""Generates real result figures (PNG) into visualizations/, sourced directly from
results/*.json and the real dataset manifest -- no invented numbers.

Run: ./venv/Scripts/python.exe src/evaluation/generate_visualizations.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

LABEL_NAMES = ["positive", "neutral", "negative"]

# validated categorical palette (dataviz skill, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


# ---------------------------------------------------------------- figure 1 --
# Model comparison: accuracy vs macro-F1 across all 6 trained variants.
models = [
    ("A1 — TF-IDF\n+ LogReg", load("model_a1_tfidf_logreg")["test"]),
    ("A2 — Frozen\nDistilBERT", load("model_a2_distilbert_frozen")["test"]),
    ("B1 — CNN\nfrom scratch", load("model_b1_cnn_scratch")["test"]),
    ("B2 — Frozen\nResNet18", load("model_b2_resnet18_frozen")["test"]),
    ("C — Concat\nFusion", load("model_c_concat_fusion")["test"]),
    ("C — Late\nFusion (tuned)", load("model_c_late_fusion")["tuned_weighted_average"]["test"]),
]
names = [m[0] for m in models]
acc = [m[1]["accuracy"] * 100 for m in models]
f1m = [m[1]["f1_macro"] * 100 for m in models]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(names))
w = 0.36
b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color=BLUE, edgecolor=SURFACE, linewidth=2)
b2 = ax.bar(x + w / 2, f1m, w, label="Macro-F1", color=ORANGE, edgecolor=SURFACE, linewidth=2)
ax.set_ylabel("%")
ax.set_title("Model comparison — test set (450 samples)")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, max(acc) * 1.2)
ax.grid(axis="y", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
# Confusion matrices, small multiple, one per model (real counts from results/*.json).
cm_models = [
    ("A1 — TF-IDF + LogReg", load("model_a1_tfidf_logreg")["test"]["confusion_matrix"]),
    ("A2 — Frozen DistilBERT", load("model_a2_distilbert_frozen")["test"]["confusion_matrix"]),
    ("B1 — CNN from scratch", load("model_b1_cnn_scratch")["test"]["confusion_matrix"]),
    ("B2 — Frozen ResNet18", load("model_b2_resnet18_frozen")["test"]["confusion_matrix"]),
    ("C — Concat Fusion", load("model_c_concat_fusion")["test"]["confusion_matrix"]),
    ("C — Late Fusion (tuned)", load("model_c_late_fusion")["tuned_weighted_average"]["test"]["confusion_matrix"]),
]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (title, cm) in zip(axes.flat, cm_models):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_NAMES, fontsize=8, rotation=20)
    ax.set_yticklabels(LABEL_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8, color=INK_MUTED)
    ax.set_ylabel("True", fontsize=8, color=INK_MUTED)
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            color = SURFACE if cm[i, j] > thresh else INK
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9, color=color)
fig.suptitle("Confusion matrices — test set, all six model variants", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "confusion_matrices.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
# Training-set class distribution, computed from the real manifest.
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.data.mvsa_dataset import load_manifest  # noqa: E402

df = load_manifest()
train_counts = df[df["split"] == "train"]["label"].value_counts().reindex(LABEL_NAMES)

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(LABEL_NAMES, train_counts.values, color=[BLUE, ORANGE, AQUA], edgecolor=SURFACE, linewidth=2)
ax.set_ylabel("Training samples")
ax.set_ylim(0, train_counts.max() * 1.22)
ax.set_title(f"Training-set class distribution (n={int(train_counts.sum())})")
ax.grid(axis="y", linewidth=0.8)
ax.set_axisbelow(True)
for rect, val in zip(bars, train_counts.values):
    pct = val / train_counts.sum() * 100
    ax.annotate(f"{int(val)}\n({pct:.1f}%)", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(OUT / "class_distribution.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 4 --
# Modality conflict: accuracy on disagreement vs agreement subsets (MODALITY_CONFLICT_ANALYSIS.md).
# On the agreement subset text and image predictions are identical by construction,
# so their accuracy there is identical too (both = 74.8%).
groups = ["Disagree\n(240/450, 53.3%)", "Agree\n(210/450, 46.7%)"]
text_acc = [50.0, 74.8]
image_acc = [33.3, 74.8]
fusion_acc = [57.1, 72.9]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(groups))
w = 0.26
ax.bar(x - w, text_acc, w, label="Text only", color=BLUE, edgecolor=SURFACE, linewidth=2)
ax.bar(x, image_acc, w, label="Image only", color=ORANGE, edgecolor=SURFACE, linewidth=2)
ax.bar(x + w, fusion_acc, w, label="Fusion", color=AQUA, edgecolor=SURFACE, linewidth=2)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Modality-conflict analysis: accuracy by subset")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0, 90)
ax.grid(axis="y", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
for xi, vals in zip(x, zip(text_acc, image_acc, fusion_acc)):
    for off, v in zip((-w, 0, w), vals):
        ax.annotate(f"{v:.1f}", (xi + off, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(OUT / "modality_conflict.png", dpi=150)
plt.close(fig)

print("Wrote:", [p.name for p in OUT.glob("*.png")])
