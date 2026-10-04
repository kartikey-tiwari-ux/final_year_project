"""Build the MVSA-Single manifest: one row per (id, text, image_path, label, split).

Data provenance (see DATASET_SELECTION.md / REPRODUCIBILITY.md for the full story):
- Images + raw text: Niu, Zhu, Pang & El Saddik (MMM 2016), 4,869 pairs, obtained via
  the community HuggingFace mirror `xwycyj/MVSA-Single` after the official OneDrive
  link (used by most citing papers) returned 404 at the time this project was built.
- Labels + the standard cleaned train/dev/test split (4,511 pairs; 358 pairs with
  low annotator agreement excluded, per the field's standard MVSA-Single cleaning
  convention): Li, Xu, Zhu & Zhao, "CLMLF: A Contrastive Learning and Multi-Layer
  Fusion Method for Multimodal Sentiment Detection," Findings of ACL: NAACL 2022,
  https://github.com/Link-Li/CLMLF (fold 1 of their published 10-fold split).
- Label mapping (0=positive, 1=neutral, 2=negative) verified empirically in this
  project by inspecting sample texts per label (not assumed from memory) — see
  REPRODUCIBILITY.md for the spot-check.

This script does not invent labels or re-split data; it merges two independently
published, citable sources and verifies 1:1 id coverage before writing the manifest.
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "mvsa_single_labels"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "mvsa_single" / "extracted" / "data"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "mvsa_single_manifest.csv"

LABEL_NAMES = {0: "positive", 1: "neutral", 2: "negative"}
SPLIT_FILES = {"train": "train", "dev": "val", "test": "test"}


def main() -> None:
    rows = []
    for json_split, split_name in SPLIT_FILES.items():
        items = json.loads((LABELS_DIR / f"{json_split}.json").read_text(encoding="utf-8"))
        for item in items:
            sample_id = item["id"]
            img_path = IMAGES_DIR / f"{sample_id}.jpg"
            txt_path = IMAGES_DIR / f"{sample_id}.txt"
            if not img_path.exists() or not txt_path.exists():
                raise FileNotFoundError(f"Missing file(s) for id {sample_id}")

            raw_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            label_int = item["emotion_label"]
            rows.append(
                {
                    "id": sample_id,
                    "text": raw_text,
                    "image_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "label": LABEL_NAMES[label_int],
                    "label_int": label_int,
                    "split": split_name,
                }
            )

    df = pd.DataFrame(rows)
    assert df["id"].is_unique, "Duplicate ids found across splits — leakage risk"
    assert set(df["split"].unique()) == {"train", "val", "test"}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df["split"].value_counts())
    print(df.groupby("split")["label"].value_counts())


if __name__ == "__main__":
    main()
