"""Demo application + research dashboard for the multimodal sentiment project.

Run: ./venv/Scripts/streamlit run app/streamlit_app.py

NON-CLINICAL DISCLAIMER (see ETHICS_AND_PRIVACY.md): this tool analyses
sentiment/emotion patterns in text and images for research and educational purposes.
It is NOT a diagnostic tool and cannot identify mental health conditions. It is not a
substitute for professional mental health assessment or care.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mvsa_dataset import LABEL_NAMES  # noqa: E402
from src.preprocessing.image_preprocessing import pretrained_transform  # noqa: E402
from src.preprocessing.text_cleaning import clean_tweet_text  # noqa: E402

st.set_page_config(page_title="Multimodal Sentiment Research Demo", layout="wide")

DISCLAIMER = (
    "**This tool analyses sentiment/emotion patterns in text and images for research "
    "and educational purposes. It is not a diagnostic tool and cannot identify mental "
    "health conditions. It is not a substitute for professional mental health "
    "assessment or care.**"
)


@st.cache_resource
def load_text_model():
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_image_model():
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


@st.cache_resource
def load_classifiers():
    """Refit the lightweight LogReg heads from the cached embeddings (fast, avoids
    needing a separate model-serialization step for this demo)."""
    from sklearn.linear_model import LogisticRegression

    from src.data.mvsa_dataset import LABEL_TO_INT, load_manifest
    from src.multimodal.fusion_concat import build_aligned_dataset, train_fusion_model

    df = load_manifest()
    text_cache = np.load(PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz")
    image_cache = np.load(PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz")
    aligned_df, _ = build_aligned_dataset(
        df, text_cache["ids"], text_cache["embeddings"], image_cache["ids"], image_cache["embeddings"]
    )
    train_sub = aligned_df[aligned_df["split"] == "train"]
    tx_train = text_cache["embeddings"][train_sub["text_row"].astype(int).values]
    im_train = image_cache["embeddings"][train_sub["image_row"].astype(int).values]
    y_train = train_sub["label"].map(LABEL_TO_INT).tolist()

    text_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(tx_train, y_train)
    image_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(im_train, y_train)
    fusion_model, _, _, _, _ = train_fusion_model(
        aligned_df, text_cache["embeddings"], image_cache["embeddings"], verbose=False
    )
    return text_clf, image_clf, fusion_model


def embed_text(text: str, tokenizer, model) -> np.ndarray:
    cleaned = clean_tweet_text(text)
    enc = tokenizer(cleaned, padding=True, truncation=True, max_length=96, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled.numpy()


def embed_image(image: Image.Image, model) -> np.ndarray:
    tensor = pretrained_transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    return out.numpy()


def predict_page():
    st.title("Multimodal Sentiment Analysis — Demo")
    st.warning(DISCLAIMER)
    st.caption(
        "Labels are the dataset's own sentiment categories (positive/neutral/negative), "
        "used as a research proxy for mental-health-related sentiment patterns — see "
        "PROJECT_REQUIREMENTS_ANALYSIS.md and ETHICS_AND_PRIVACY.md."
    )

    text_input = st.text_area("Post text", placeholder="Type or paste a social-media-style post...")
    image_file = st.file_uploader("Post image (optional)", type=["jpg", "jpeg", "png"])

    if st.button("Analyze", type="primary"):
        if not text_input.strip() and image_file is None:
            st.error("Provide text and/or an image.")
            return

        tokenizer, text_model = load_text_model()
        resnet_model = load_image_model()
        text_clf, image_clf, fusion_model = load_classifiers()

        cols = st.columns(3)
        text_probs = image_probs = None
        text_emb = image_emb = None

        if text_input.strip():
            text_emb = embed_text(text_input, tokenizer, text_model)
            text_probs = text_clf.predict_proba(text_emb)[0]
            with cols[0]:
                st.subheader("Text-only prediction")
                pred_idx = int(np.argmax(text_probs))
                st.metric("Predicted sentiment", LABEL_NAMES[pred_idx], f"{text_probs[pred_idx]:.1%} confidence")
                st.bar_chart(pd.Series(text_probs, index=LABEL_NAMES))

        if image_file is not None:
            image = Image.open(image_file).convert("RGB")
            with cols[1]:
                st.image(image, caption="Uploaded image", width=200)
            image_emb = embed_image(image, resnet_model)
            image_probs = image_clf.predict_proba(image_emb)[0]
            with cols[1]:
                st.subheader("Image-only prediction")
                pred_idx = int(np.argmax(image_probs))
                st.metric("Predicted sentiment", LABEL_NAMES[pred_idx], f"{image_probs[pred_idx]:.1%} confidence")
                st.bar_chart(pd.Series(image_probs, index=LABEL_NAMES))

        if text_emb is not None and image_emb is not None:
            with torch.no_grad():
                fusion_logits = fusion_model(
                    torch.tensor(text_emb, dtype=torch.float32), torch.tensor(image_emb, dtype=torch.float32)
                )
                fusion_probs = torch.softmax(fusion_logits, dim=1).numpy()[0]
            with cols[2]:
                st.subheader("Multimodal (fusion) prediction")
                pred_idx = int(np.argmax(fusion_probs))
                st.metric("Predicted sentiment", LABEL_NAMES[pred_idx], f"{fusion_probs[pred_idx]:.1%} confidence")
                st.bar_chart(pd.Series(fusion_probs, index=LABEL_NAMES))

            st.divider()
            st.subheader("Modality comparison")
            comparison = pd.DataFrame(
                {"positive": [], "neutral": [], "negative": []}
            )
            comparison.loc["Text only"] = text_probs
            comparison.loc["Image only"] = image_probs
            comparison.loc["Fusion"] = fusion_probs
            st.dataframe(comparison.style.format("{:.1%}"))
            if np.argmax(text_probs) != np.argmax(image_probs):
                st.info(
                    "Text and image models disagree on this post — see "
                    "MODALITY_CONFLICT_ANALYSIS.md for how often this happens on the "
                    "test set (53.3%) and how fusion performs specifically in these cases."
                )
        elif text_emb is not None:
            st.info("Upload an image too to see the multimodal fusion prediction and comparison.")


def dashboard_page():
    st.title("Research Dashboard")
    st.warning(DISCLAIMER)

    results_dir = PROJECT_ROOT / "results"
    model_files = {
        "A1 — TF-IDF + LogReg": "model_a1_tfidf_logreg.json",
        "A2 — Frozen DistilBERT + LogReg": "model_a2_distilbert_frozen.json",
        "B1 — CNN from scratch": "model_b1_cnn_scratch.json",
        "B2 — Frozen ResNet18 + LogReg": "model_b2_resnet18_frozen.json",
        "C — Concat Fusion": "model_c_concat_fusion.json",
    }

    rows = []
    for name, fname in model_files.items():
        fpath = results_dir / fname
        if not fpath.exists():
            continue
        d = json.loads(fpath.read_text())
        test = d["test"]
        rows.append(
            {
                "Model": name,
                "Accuracy": test["accuracy"],
                "Macro F1": test["f1_macro"],
                "Weighted F1": test["f1_weighted"],
                "Macro Precision": test["precision_macro"],
                "Macro Recall": test["recall_macro"],
            }
        )

    if not rows:
        st.info("No results found yet. Run the training scripts under src/ first.")
        return

    df = pd.DataFrame(rows).set_index("Model")
    st.subheader("Model comparison (test set, 450 samples)")
    st.dataframe(df.style.format("{:.1%}"))
    st.bar_chart(df[["Accuracy", "Macro F1"]])

    st.subheader("Dataset class distribution")
    st.caption("Training set (3,611 samples) — see DATASET_SELECTION.md")
    class_dist = pd.Series({"positive": 2147, "neutral": 376, "negative": 1088})
    st.bar_chart(class_dist)

    st.subheader("Confusion matrices")
    selected = st.selectbox("Model", list(model_files.keys()))
    fpath = results_dir / model_files[selected]
    if fpath.exists():
        d = json.loads(fpath.read_text())
        cm = pd.DataFrame(d["test"]["confusion_matrix"], index=LABEL_NAMES, columns=LABEL_NAMES)
        st.dataframe(cm.style.background_gradient(cmap="Blues"))

    st.caption(
        "Full analysis: see MODALITY_CONFLICT_ANALYSIS.md and ERROR_ANALYSIS.md in the project root."
    )


def main():
    page = st.sidebar.radio("Page", ["Predict", "Research Dashboard"])
    if page == "Predict":
        predict_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
