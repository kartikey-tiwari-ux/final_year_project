"""Fusion ablation — late (decision-level) fusion: average the independently-trained
Model A2 (text) and Model B2 (image) predicted class probabilities, vs. validation-
tuned weighted average. Compared against Model C's feature-concatenation fusion to
answer: does joint training on combined features beat combining two independently
trained models' opinions? See PROJECT_ARCHITECTURE.md.

Refits A2/B2's LogisticRegression heads here (fast, deterministic given the seed) on
the same cached embeddings used by src/text_model/distilbert_classifier.py and
src/image_model/resnet_classifier.py, rather than depending on pickled models.
Run: python -m src.multimodal.fusion_late
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.multimodal.fusion_concat import build_aligned_dataset
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz"
IMAGE_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_c_late_fusion.json"


def main() -> None:
    set_seed(42)
    df = load_manifest()
    text_cache = np.load(TEXT_EMB_PATH)
    image_cache = np.load(IMAGE_EMB_PATH)
    df, n_dropped = build_aligned_dataset(
        df, text_cache["ids"], text_cache["embeddings"], image_cache["ids"], image_cache["embeddings"]
    )
    if n_dropped:
        print(f"Note: {n_dropped} samples dropped (missing from text and/or image embedding cache).")

    text_emb, image_emb = text_cache["embeddings"], image_cache["embeddings"]

    def arrays(split_name):
        sub = df[df["split"] == split_name]
        tx = text_emb[sub["text_row"].astype(int).values]
        im = image_emb[sub["image_row"].astype(int).values]
        y = sub["label"].map(LABEL_TO_INT).tolist()
        return tx, im, y

    tx_train, im_train, y_train = arrays("train")
    tx_val, im_val, y_val = arrays("val")
    tx_test, im_test, y_test = arrays("test")

    text_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(tx_train, y_train)
    image_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(im_train, y_train)

    def combined_probs(text_probs, image_probs, w):
        return w * text_probs + (1 - w) * image_probs

    val_text_probs = text_clf.predict_proba(tx_val)
    val_image_probs = image_clf.predict_proba(im_val)

    # Tune the text/image weight on validation only (never on test).
    best_w, best_f1 = 0.5, -1.0
    for w in np.arange(0.0, 1.01, 0.05):
        preds = combined_probs(val_text_probs, val_image_probs, w).argmax(1)
        f1 = compute_metrics(y_val, preds)["f1_macro"]
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    print(f"Best validation-tuned weight: w_text={best_w:.2f} (val macro F1={best_f1:.4f})")

    # Simple (unweighted) average, reported alongside the tuned version.
    simple_val_preds = combined_probs(val_text_probs, val_image_probs, 0.5).argmax(1)
    simple_val_metrics = compute_metrics(y_val, simple_val_preds)

    test_text_probs = text_clf.predict_proba(tx_test)
    test_image_probs = image_clf.predict_proba(im_test)
    simple_test_preds = combined_probs(test_text_probs, test_image_probs, 0.5).argmax(1)
    tuned_test_preds = combined_probs(test_text_probs, test_image_probs, best_w).argmax(1)

    simple_test_metrics = compute_metrics(y_test, simple_test_preds)
    tuned_test_metrics = compute_metrics(y_test, tuned_test_preds)
    tuned_val_preds = combined_probs(val_text_probs, val_image_probs, best_w).argmax(1)
    tuned_val_metrics = compute_metrics(y_val, tuned_val_preds)

    print_summary("Late Fusion (simple 0.5/0.5 average) — Validation", simple_val_metrics)
    print_summary("Late Fusion (simple 0.5/0.5 average) — Test", simple_test_metrics)
    print_summary(f"Late Fusion (tuned w_text={best_w:.2f}) — Validation", tuned_val_metrics)
    print_summary(f"Late Fusion (tuned w_text={best_w:.2f}) — Test", tuned_test_metrics)

    save_results(
        {
            "model": "C_late_fusion",
            "config": {"seed": 42, "dropped_samples": int(n_dropped), "tuned_weight_text": best_w},
            "simple_average": {"val": simple_val_metrics, "test": simple_test_metrics},
            "tuned_weighted_average": {"val": tuned_val_metrics, "test": tuned_test_metrics},
        },
        RESULTS_PATH,
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
