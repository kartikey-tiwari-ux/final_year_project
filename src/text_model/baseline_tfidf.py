"""Model A1 — traditional NLP baseline: TF-IDF + Logistic Regression.

See PROJECT_ARCHITECTURE.md. Run: python -m src.text_model.baseline_tfidf
"""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_TO_INT, get_split, load_manifest
from src.evaluation.metrics import compute_metrics, print_summary, save_results
from src.preprocessing.text_cleaning import clean_tweet_text
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "results" / "model_a1_tfidf_logreg.json"


def main() -> None:
    set_seed(42)
    df = load_manifest()
    train_df, val_df, test_df = get_split(df, "train"), get_split(df, "val"), get_split(df, "test")

    X_train_text = train_df["text"].apply(clean_tweet_text).tolist()
    X_val_text = val_df["text"].apply(clean_tweet_text).tolist()
    X_test_text = test_df["text"].apply(clean_tweet_text).tolist()
    y_train = train_df["label"].map(LABEL_TO_INT).tolist()
    y_val = val_df["label"].map(LABEL_TO_INT).tolist()
    y_test = test_df["label"].map(LABEL_TO_INT).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, C=1.0
    )
    clf.fit(X_train, y_train)

    val_metrics = compute_metrics(y_val, clf.predict(X_val))
    test_metrics = compute_metrics(y_test, clf.predict(X_test))

    print_summary("Model A1 (TF-IDF + Logistic Regression) — Validation", val_metrics)
    print_summary("Model A1 (TF-IDF + Logistic Regression) — Test", test_metrics)

    save_results(
        {
            "model": "A1_tfidf_logreg",
            "config": {
                "max_features": 10000,
                "ngram_range": [1, 2],
                "class_weight": "balanced",
                "seed": 42,
            },
            "val": val_metrics,
            "test": test_metrics,
        },
        RESULTS_PATH,
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
