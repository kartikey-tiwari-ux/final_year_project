import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from utils import LABELS


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    per_class = {
        LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
        for i in range(3)
    }
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def save_result(name, result, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] acc={result['accuracy']:.4f} macroF1={result['macro_f1']:.4f} "
          f"weightedF1={result['weighted_f1']:.4f}")
