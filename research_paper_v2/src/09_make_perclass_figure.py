import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"


def load(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def main():
    a2 = load("A2_distilbert_logreg")
    c = load("C_concat_fusion_sharp")
    cblur = load("Cblur_concat_fusion_blur")

    classes = ["negative", "neutral", "positive"]
    models = [("Text-only (A2)", a2, "#8C8C8C"),
              ("Fusion sharp (C)", c, "#3B6EA5"),
              ("Fusion blurred (C-Blur)", cblur, "#D9822B")]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    for i, (label, r, color) in enumerate(models):
        vals = [100 * r["per_class"][cl]["f1"] for cl in classes]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("Per-class F1 (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=8.5)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "perclass_f1.png", bbox_inches="tight")
    print("Saved figures/perclass_f1.png")


if __name__ == "__main__":
    main()
