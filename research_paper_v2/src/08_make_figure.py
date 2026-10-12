import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def main():
    b2 = load("B2_resnet18_sharp_logreg")
    b2blur = load("B2blur_resnet18_blur_logreg")
    c = load("C_concat_fusion_sharp")
    cblur = load("Cblur_concat_fusion_blur")
    a2 = load("A2_distilbert_logreg")

    groups = ["Text-only\n(A2)", "Image-only\nsharp (B2)", "Image-only\nblurred (B2-Blur)",
              "Fusion\nsharp (C)", "Fusion\nblurred (C-Blur)"]
    acc = [a2["accuracy"], b2["accuracy"], b2blur["accuracy"], c["accuracy"], cblur["accuracy"]]
    f1 = [a2["macro_f1"], b2["macro_f1"], b2blur["macro_f1"], c["macro_f1"], cblur["macro_f1"]]
    acc = [100 * v for v in acc]
    f1 = [100 * v for v in f1]

    x = range(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=200)
    bars1 = ax.bar([i - width / 2 for i in x], acc, width, label="Accuracy", color="#3B6EA5")
    bars2 = ax.bar([i + width / 2 for i in x], f1, width, label="Macro-F1", color="#D9822B")

    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Test-set score (%)", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=7.5)
    ax.set_ylim(0, max(acc + f1) * 1.18)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "sharp_vs_blur.png", bbox_inches="tight")
    print("Saved figures/sharp_vs_blur.png")


if __name__ == "__main__":
    main()
