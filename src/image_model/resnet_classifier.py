"""Model B2 — modern image model: frozen ResNet18 embeddings + classifier head.

Requires embeddings already extracted: python -m src.image_model.extract_resnet_embeddings
Run: python -m src.image_model.resnet_classifier
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_b2_resnet18_frozen.json"


def main() -> None:
    set_seed(42)
    df = load_manifest()
    cache = np.load(EMB_PATH)
    emb_ids = list(cache["ids"])
    embeddings = cache["embeddings"]
    id_to_row = {sid: i for i, sid in enumerate(emb_ids)}

    # Some images may have been dropped as missing/corrupt during extraction —
    # filter the manifest down to only samples that actually have an embedding,
    # logging how many were excluded (per TESTING.md / leakage-prevention discipline:
    # dropped samples must be visible, not silently vanish from the denominator).
    df["emb_row"] = df["id"].map(id_to_row)
    n_before = len(df)
    df = df[df["emb_row"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Note: {n_dropped} manifest samples had no image embedding (dropped at extraction) — excluded here.")

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

    print_summary("Model B2 (Frozen ResNet18 + LogReg) — Validation", val_metrics)
    print_summary("Model B2 (Frozen ResNet18 + LogReg) — Test", test_metrics)

    save_results(
        {
            "model": "B2_resnet18_frozen_logreg",
            "config": {
                "embedding_dim": embeddings.shape[1],
                "class_weight": "balanced",
                "seed": 42,
                "dropped_missing_images": n_dropped,
            },
            "val": val_metrics,
            "test": test_metrics,
        },
        RESULTS_PATH,
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
