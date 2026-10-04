"""Model A2 — transformer text model: frozen DistilBERT embeddings + classifier head.

Requires embeddings already extracted: python -m src.text_model.extract_distilbert_embeddings
Run: python -m src.text_model.distilbert_classifier
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_a2_distilbert_frozen.json"


def main() -> None:
    set_seed(42)
    df = load_manifest()
    cache = np.load(EMB_PATH)
    emb_ids = list(cache["ids"])
    embeddings = cache["embeddings"]
    id_to_row = {sid: i for i, sid in enumerate(emb_ids)}

    df["emb_row"] = df["id"].map(id_to_row)
    assert df["emb_row"].notna().all(), "Some manifest ids missing from embedding cache"

    def split_arrays(split_name):
        sub = df[df["split"] == split_name]
        X = embeddings[sub["emb_row"].astype(int).values]
        y = sub["label"].map(LABEL_TO_INT).tolist()
        return X, y

    X_train, y_train = split_arrays("train")
    X_val, y_val = split_arrays("val")
    X_test, y_test = split_arrays("test")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=1.0)
    clf.fit(X_train, y_train)

    val_metrics = compute_metrics(y_val, clf.predict(X_val))
    test_metrics = compute_metrics(y_test, clf.predict(X_test))

    print_summary("Model A2 (Frozen DistilBERT + LogReg) — Validation", val_metrics)
    print_summary("Model A2 (Frozen DistilBERT + LogReg) — Test", test_metrics)

    save_results(
        {
            "model": "A2_distilbert_frozen_logreg",
            "config": {"embedding_dim": embeddings.shape[1], "class_weight": "balanced", "seed": 42},
            "val": val_metrics,
            "test": test_metrics,
        },
        RESULTS_PATH,
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
