"""Duplicate-hash leakage audit: confirm no exact-duplicate text or image appears in
more than one split. The CLMLF split is id-unique (checked in build_mvsa_manifest.py),
but a near-duplicate repost (same image/text re-shared under a different id) could
still leak signal across splits — this check looks for that directly.
Run: python -m src.data.check_leakage
"""
import hashlib
from collections import defaultdict
from pathlib import Path

from src.data.mvsa_dataset import load_manifest


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> None:
    df = load_manifest()

    # --- Text duplicates ---
    text_to_splits = defaultdict(set)
    for _, row in df.iterrows():
        text_to_splits[row["text"].strip()].add(row["split"])
    cross_split_text = {t: s for t, s in text_to_splits.items() if len(s) > 1}

    # --- Image duplicates (exact byte-for-byte) ---
    image_hash_to_splits = defaultdict(set)
    image_hash_to_ids = defaultdict(list)
    for _, row in df.iterrows():
        h = file_md5(row["image_path"])
        image_hash_to_splits[h].add(row["split"])
        image_hash_to_ids[h].append(row["id"])
    cross_split_images = {h: s for h, s in image_hash_to_splits.items() if len(s) > 1}

    print(f"Total samples: {len(df)}")
    print(f"Unique texts: {len(text_to_splits)} (exact duplicates exist: {len(df) - len(text_to_splits)})")
    print(f"Texts appearing in >1 split: {len(cross_split_text)}")
    print(f"Unique images (by content hash): {len(image_hash_to_splits)} "
          f"(exact duplicates exist: {len(df) - len(image_hash_to_splits)})")
    print(f"Images appearing in >1 split: {len(cross_split_images)}")

    if cross_split_images:
        print("\nSample cross-split duplicate image ids (first 5 groups):")
        shown = 0
        for h, splits in cross_split_images.items():
            if shown >= 5:
                break
            print(f"  hash={h[:10]}... splits={splits} ids={image_hash_to_ids[h]}")
            shown += 1

    report_path = Path(__file__).resolve().parents[2] / "results" / "leakage_audit.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Unique texts: {len(text_to_splits)}\n")
        f.write(f"Texts appearing in >1 split: {len(cross_split_text)}\n")
        f.write(f"Unique images (by content hash): {len(image_hash_to_splits)}\n")
        f.write(f"Images appearing in >1 split: {len(cross_split_images)}\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
