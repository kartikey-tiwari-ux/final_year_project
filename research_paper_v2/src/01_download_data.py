"""
Downloads the MMSoc_Memotion dataset (HuggingFace mirror of the SemEval-2020
Task 8 Memotion dataset: real internet memes, human-annotated for sentiment,
humor, sarcasm, offensiveness, and motivational content).

Subsamples (stratified by sentiment) to a resource-constrained scale
comparable to the reference project (CPU-only, 8GB RAM), and writes:
  - data/raw/{train,val,test}.parquet  (text + label columns, no images)
  - data/raw/images/{split}/{idx}.jpg  (decoded images)
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_DIR = RAW / "images"
RAW.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

# Target subsample sizes per split (stratified by 3-class collapsed sentiment).
# Kept modest and documented, mirroring the reference project's own
# resource-constrained scale (4,511 total samples).
TARGET = {"train": 2800, "validation": 350, "test": 350}

SENTIMENT_MAP = {
    "very_positive": "positive",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "very_negative": "negative",
}


def stratified_subsample(ds, n_target, label_col="sentiment_3class"):
    by_label = {}
    for i, lab in enumerate(ds[label_col]):
        by_label.setdefault(lab, []).append(i)
    labels = sorted(by_label.keys())
    n_per = {lab: len(idxs) for lab, idxs in by_label.items()}
    total = sum(n_per.values())
    chosen = []
    for lab in labels:
        frac = n_per[lab] / total
        k = max(1, round(n_target * frac))
        idxs = by_label[lab][:]
        random.Random(SEED).shuffle(idxs)
        chosen.extend(idxs[:k])
    random.Random(SEED).shuffle(chosen)
    return chosen[:n_target]


def process_split(ds, split_name, hf_split_name):
    ds = ds.map(lambda ex: {"sentiment_3class": SENTIMENT_MAP[ex["sentiment"]]})
    idxs = stratified_subsample(ds, TARGET[split_name])
    sub = ds.select(idxs)

    out_img_dir = IMG_DIR / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for new_idx, ex in enumerate(sub):
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_path = out_img_dir / f"{new_idx}.jpg"
        img.save(img_path, "JPEG", quality=90)
        rows.append({
            "idx": new_idx,
            "text_ocr": ex.get("text_ocr", "") or "",
            "text_corrected": ex.get("text_corrected", "") or "",
            "humor": ex.get("humor", ""),
            "sarcasm": ex.get("sarcasm", ""),
            "offensive": ex.get("offensive", ""),
            "motivational": ex.get("motivational", ""),
            "sentiment_5class": ex.get("sentiment", ""),
            "sentiment_3class": ex["sentiment_3class"],
            "image_path": str(img_path.relative_to(ROOT)),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(RAW / f"{split_name}.parquet", index=False)
    print(f"{split_name}: {len(df)} rows written; class dist:",
          df["sentiment_3class"].value_counts().to_dict())
    return df


def main():
    print("Loading Ahren09/MMSoc_Memotion from the Hugging Face Hub...")
    ds_dict = load_dataset("Ahren09/MMSoc_Memotion")
    print({k: len(v) for k, v in ds_dict.items()})

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    available = set(ds_dict.keys())
    print("available splits:", available)

    summary = {}
    for split_name, hf_name in split_map.items():
        if hf_name not in ds_dict:
            # fall back: some mirrors only expose 'train'; handle below.
            continue
        df = process_split(ds_dict[hf_name], split_name, hf_name)
        summary[split_name] = len(df)

    with open(RAW / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done.", summary)


if __name__ == "__main__":
    main()
