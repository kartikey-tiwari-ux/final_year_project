"""Generate and save per-sample test-set predictions from every trained model, so
error analysis and modality-conflict analysis (which need to compare individual
predictions, not just aggregate metrics) can be done without retraining.
Requires all embedding caches to already exist.
Run: python -m src.evaluation.generate_predictions
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_NAMES, LABEL_TO_INT, get_split, load_manifest
from src.multimodal.fusion_concat import build_aligned_dataset
from src.preprocessing.text_cleaning import clean_tweet_text
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "results" / "test_predictions.csv"


def main() -> None:
    set_seed(42)
    df = load_manifest()

    text_cache = np.load(PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz")
    image_cache = np.load(PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz")
    aligned_df, _ = build_aligned_dataset(
        df, text_cache["ids"], text_cache["embeddings"], image_cache["ids"], image_cache["embeddings"]
    )
    text_emb, image_emb = text_cache["embeddings"], image_cache["embeddings"]

    def arrays(split_name):
        sub = aligned_df[aligned_df["split"] == split_name]
        tx = text_emb[sub["text_row"].astype(int).values]
        im = image_emb[sub["image_row"].astype(int).values]
        y = sub["label"].map(LABEL_TO_INT).tolist()
        return sub, tx, im, y

    train_sub, tx_train, im_train, y_train = arrays("train")
    test_sub, tx_test, im_test, y_test = arrays("test")

    # --- A1: TF-IDF + LogReg ---
    train_df_raw, test_df_raw = get_split(df, "train"), get_split(df, "test")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train_df_raw["text"].apply(clean_tweet_text))
    X_test_tfidf = tfidf.transform(test_df_raw["text"].apply(clean_tweet_text))
    a1_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(
        X_train_tfidf, train_df_raw["label"].map(LABEL_TO_INT)
    )
    a1_preds_full = a1_clf.predict(X_test_tfidf)
    a1_pred_by_id = dict(zip(test_df_raw["id"], a1_preds_full))

    # --- A2: frozen DistilBERT + LogReg ---
    a2_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(tx_train, y_train)
    a2_preds = a2_clf.predict(tx_test)
    a2_probs = a2_clf.predict_proba(tx_test)

    # --- B2: frozen ResNet18 + LogReg ---
    b2_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(im_train, y_train)
    b2_preds = b2_clf.predict(im_test)
    b2_probs = b2_clf.predict_proba(im_test)

    # --- C: concat fusion — uses the EXACT SAME training routine (same seed, same
    # best-val-checkpoint selection) as src/multimodal/fusion_concat.py's own main(),
    # via the shared train_fusion_model() function, so this script can never silently
    # report a different "Model C" number than the one actually published in
    # results/model_c_concat_fusion.json. (An earlier version of this script
    # re-implemented training ad hoc without checkpoint selection and got a different,
    # worse number — 59.3% vs the real 64.4% — which was caught and fixed; see
    # REPRODUCIBILITY.md.) ---
    import torch

    from src.multimodal.fusion_concat import train_fusion_model

    model, _, _, _, _ = train_fusion_model(aligned_df, text_emb, image_emb, verbose=False)
    model.eval()
    with torch.no_grad():
        c_logits = model(torch.tensor(tx_test, dtype=torch.float32), torch.tensor(im_test, dtype=torch.float32))
    c_preds = c_logits.argmax(1).numpy()
    c_test_acc = (c_preds == np.array(y_test)).mean()
    print(f"Model C (concat fusion) test accuracy in this run: {c_test_acc:.4f} (should match results/model_c_concat_fusion.json)")

    out = test_sub[["id", "text", "label", "label_int"]].copy()
    out["pred_a1_tfidf"] = out["id"].map(a1_pred_by_id)
    out["pred_a2_distilbert"] = a2_preds
    out["pred_b2_resnet"] = b2_preds
    out["pred_c_concat_fusion"] = c_preds
    out["a2_confidence"] = a2_probs.max(axis=1)
    out["b2_confidence"] = b2_probs.max(axis=1)

    for col in ["pred_a1_tfidf", "pred_a2_distilbert", "pred_b2_resnet", "pred_c_concat_fusion"]:
        out[col] = out[col].astype(int)
        out[col.replace("pred_", "") + "_name"] = out[col].map(lambda i: LABEL_NAMES[int(i)])

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} test-set predictions across 4 models to {OUT_PATH}")


if __name__ == "__main__":
    main()
