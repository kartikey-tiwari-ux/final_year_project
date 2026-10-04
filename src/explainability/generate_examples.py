"""Generate real explainability examples (text saliency + image Grad-CAM) on a
handful of actual test-set samples, saved for the report/paper. Not a rigorously
validated attribution method — exploratory, per RESEARCH_GAP.md point 4.
Run: python -m src.explainability.generate_examples
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from src.data.mvsa_dataset import LABEL_NAMES, LABEL_TO_INT, load_manifest
from src.explainability.image_gradcam import GradCAM
from src.explainability.text_saliency import TextSaliencyExplainer
from src.multimodal.fusion_concat import build_aligned_dataset
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "explainability_examples"


def main() -> None:
    set_seed(42)
    df = load_manifest()
    text_cache = np.load(PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz")
    image_cache = np.load(PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz")
    aligned_df, _ = build_aligned_dataset(
        df, text_cache["ids"], text_cache["embeddings"], image_cache["ids"], image_cache["embeddings"]
    )
    train_sub = aligned_df[aligned_df["split"] == "train"]
    test_sub = aligned_df[aligned_df["split"] == "test"].reset_index(drop=True)

    tx_train = text_cache["embeddings"][train_sub["text_row"].astype(int).values]
    im_train = image_cache["embeddings"][train_sub["image_row"].astype(int).values]
    y_train = train_sub["label"].map(LABEL_TO_INT).tolist()

    text_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(tx_train, y_train)
    image_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(im_train, y_train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 3 real test examples to explain.
    sample_rows = test_sub.sample(n=3, random_state=42)

    text_explainer = None
    examples_summary = []

    from torchvision.models import ResNet18_Weights, resnet18

    from src.preprocessing.image_preprocessing import pretrained_transform

    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.eval()
    target_layer = resnet.layer4[-1].conv2  # last conv block before global pooling

    for i, row in sample_rows.iterrows():
        sample_id = row["id"]
        text = row["text"]
        true_label = row["label"]

        # --- Text saliency ---
        text_pred_idx = int(text_clf.predict([text_cache["embeddings"][row["text_row"]]])[0])
        weight_vec = torch.tensor(text_clf.coef_[text_pred_idx], dtype=torch.float32)
        bias = torch.tensor(text_clf.intercept_[text_pred_idx], dtype=torch.float32)

        if text_explainer is None:
            text_explainer = TextSaliencyExplainer(weight_vec, bias)
        else:
            text_explainer.classifier_weights = weight_vec
            text_explainer.classifier_bias = bias

        tokens, saliency = text_explainer.explain(text)
        top_tokens = sorted(zip(tokens, saliency), key=lambda x: -x[1])[:5]

        # --- Image Grad-CAM ---
        image_pred_idx = int(image_clf.predict([image_cache["embeddings"][row["image_row"]]])[0])
        image_weight_vec = torch.tensor(image_clf.coef_[image_pred_idx], dtype=torch.float32)

        img = Image.open(row["image_path"]).convert("RGB")
        img_tensor = pretrained_transform(img).unsqueeze(0)

        # Build a backbone that exposes conv features (everything before avgpool/fc).
        backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        cam_tool = GradCAM(backbone, target_layer)
        cam = cam_tool.generate(img_tensor, image_weight_vec)

        example = {
            "id": str(sample_id),
            "true_label": true_label,
            "text": text,
            "text_prediction": LABEL_NAMES[text_pred_idx],
            "top_salient_tokens": [(t, round(float(s), 3)) for t, s in top_tokens],
            "image_prediction": LABEL_NAMES[image_pred_idx],
            "gradcam_heatmap_stats": {
                "max": float(cam.max()),
                "mean": float(cam.mean()),
                "shape": list(cam.shape),
            },
        }
        examples_summary.append(example)
        np.save(OUT_DIR / f"gradcam_{sample_id}.npy", cam)
        print(f"id={sample_id} true={true_label} text_pred={LABEL_NAMES[text_pred_idx]} "
              f"image_pred={LABEL_NAMES[image_pred_idx]}")
        print(f"  top tokens: {example['top_salient_tokens']}")

    with open(OUT_DIR / "examples_summary.json", "w", encoding="utf-8") as f:
        json.dump(examples_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(examples_summary)} explainability examples to {OUT_DIR}")


if __name__ == "__main__":
    main()
