"""
Compiles the final results table and runs:
  (1) Modality-conflict analysis (text A2 vs. image B2-sharp disagreement
      subset), mirroring the comparability analysis used in the reference
      project, to test whether the "fusion helps where modalities disagree"
      pattern generalizes from tweet-sentiment to meme-sentiment data.
  (2) The sharp-vs-blur comparison for B2 and C, the core novel ablation.
Writes results/SUMMARY.json and results/SUMMARY.md.
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, LABELS

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MODEL_FILES = {
    "A1_tfidf_logreg": "A1 -- TF-IDF + LogReg (text)",
    "A2_distilbert_logreg": "A2 -- Frozen DistilBERT + LogReg (text)",
    "B1_cnn_scratch": "B1 -- CNN from scratch (image, sharp)",
    "B2_resnet18_sharp_logreg": "B2 -- Frozen ResNet18 + LogReg (image, sharp)",
    "B2blur_resnet18_blur_logreg": "B2-Blur -- Frozen ResNet18 + LogReg (image, blurred)",
    "C_concat_fusion_sharp": "C -- Concat Fusion (text + sharp image)",
    "Cblur_concat_fusion_blur": "C-Blur -- Concat Fusion (text + blurred image)",
}


def load_result(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def modality_conflict_analysis():
    yte = np.load(PROC / "labels_test.npy")
    pred_text = np.load(PROC / "preds_A2_distilbert_logreg.npy")
    pred_img = np.load(PROC / "preds_B2_resnet18_sharp_logreg.npy")
    pred_fusion = np.load(PROC / "preds_C_concat_fusion_sharp.npy")

    disagree = pred_text != pred_img
    agree = ~disagree

    def acc(mask, pred):
        if mask.sum() == 0:
            return None
        return float((pred[mask] == yte[mask]).mean())

    out = {
        "n_test": int(len(yte)),
        "n_disagree": int(disagree.sum()),
        "pct_disagree": float(100.0 * disagree.sum() / len(yte)),
        "disagree_subset": {
            "text_only_acc": acc(disagree, pred_text),
            "image_only_acc": acc(disagree, pred_img),
            "fusion_acc": acc(disagree, pred_fusion),
        },
        "agree_subset": {
            "text_only_acc": acc(agree, pred_text),
            "image_only_acc": acc(agree, pred_img),
            "fusion_acc": acc(agree, pred_fusion),
        },
    }
    with open(RESULTS / "modality_conflict_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def blur_leakage_analysis():
    b2 = load_result("B2_resnet18_sharp_logreg")
    b2blur = load_result("B2blur_resnet18_blur_logreg")
    c = load_result("C_concat_fusion_sharp")
    cblur = load_result("Cblur_concat_fusion_blur")
    a2 = load_result("A2_distilbert_logreg")

    with open(PROC / "blur_validation_metric.json") as f:
        blur_metric = json.load(f)

    out = {
        "blur_validation_metric": blur_metric,
        "image_only_accuracy_drop_pp": round(
            100 * (b2["accuracy"] - b2blur["accuracy"]), 2),
        "image_only_macrof1_drop_pp": round(
            100 * (b2["macro_f1"] - b2blur["macro_f1"]), 2),
        "fusion_accuracy_drop_pp": round(
            100 * (c["accuracy"] - cblur["accuracy"]), 2),
        "fusion_macrof1_drop_pp": round(
            100 * (c["macro_f1"] - cblur["macro_f1"]), 2),
        "cblur_vs_text_only_accuracy_delta_pp": round(
            100 * (cblur["accuracy"] - a2["accuracy"]), 2),
        "cblur_vs_text_only_macrof1_delta_pp": round(
            100 * (cblur["macro_f1"] - a2["macro_f1"]), 2),
    }
    with open(RESULTS / "blur_leakage_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    summary_rows = []
    for key, label in MODEL_FILES.items():
        r = load_result(key)
        summary_rows.append({
            "model": label,
            "key": key,
            "accuracy": round(100 * r["accuracy"], 1),
            "macro_f1": round(100 * r["macro_f1"], 1),
            "weighted_f1": round(100 * r["weighted_f1"], 1),
            "per_class": r["per_class"],
        })

    conflict = modality_conflict_analysis()
    leakage = blur_leakage_analysis()

    summary = {
        "models": summary_rows,
        "modality_conflict_analysis": conflict,
        "blur_leakage_analysis": leakage,
    }
    with open(RESULTS / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["| Model | Acc. | Macro-F1 | Weighted-F1 |", "|---|---|---|---|"]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {row['accuracy']}% | {row['macro_f1']}% | {row['weighted_f1']}% |")
    with open(RESULTS / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nModality conflict analysis:", json.dumps(conflict, indent=2))
    print("\nBlur leakage analysis:", json.dumps(leakage, indent=2))


if __name__ == "__main__":
    main()
