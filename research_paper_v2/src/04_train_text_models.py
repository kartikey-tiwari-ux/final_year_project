"""
A1 -- TF-IDF + class-weighted Logistic Regression
A2 -- Frozen DistilBERT embeddings + class-weighted Logistic Regression
Evaluated on the held-out test split; predictions also saved for A2 (needed
by the modality-conflict analysis later).
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, seed_everything
from metrics import compute_metrics, save_result

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def run(feature_prefix, name):
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


def main():
    seed_everything()
    run("tfidf", "A1_tfidf_logreg")
    run("distilbert", "A2_distilbert_logreg")


if __name__ == "__main__":
    main()
