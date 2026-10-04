"""Shared evaluation utilities — used identically by Models A, B, C so comparisons
are apples-to-apples (see PROJECT_REQUIREMENTS_ANALYSIS.md evaluation requirements)."""
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = ["positive", "neutral", "negative"]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_summary(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:     {metrics['f1_weighted']:.4f}")
    print(f"  Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:    {metrics['recall_macro']:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred, order={LABEL_NAMES}):")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
