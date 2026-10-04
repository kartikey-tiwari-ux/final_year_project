"""Cache resized raw pixels (uint8, pre-normalization) for every image, once.

Model B1 (CNN trained from scratch) needs raw pixels, not frozen embeddings — but
re-decoding ~4,500 JPEGs from disk every training epoch is the same I/O bottleneck
measured in REPRODUCIBILITY.md (~530ms/image on this machine), which would make even
a small number of epochs take hours. Decoding once and caching as uint8 (not float32 —
4x smaller, ~679MB total vs ~2.7GB) avoids repeating that cost every epoch while
staying within this machine's 8GB RAM. Run: python -m src.image_model.extract_pixel_cache
"""
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.mvsa_dataset import load_manifest
from src.preprocessing.image_preprocessing import load_image_safely

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "pixel_cache_224.npz"
SIZE = 224


def main() -> None:
    df = load_manifest()
    valid_ids, pixel_arrays, dropped = [], [], []

    print(f"Decoding+resizing {len(df)} images to {SIZE}x{SIZE} uint8 (one-time cost)...")
    start = time.time()
    for _, row in df.iterrows():
        img = load_image_safely(Path(row["image_path"]))
        if img is None:
            dropped.append(row["id"])
            continue
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        pixel_arrays.append(arr)
        valid_ids.append(row["id"])
    elapsed = time.time() - start
    print(f"Loaded {len(valid_ids)} images, dropped {len(dropped)}. Took {elapsed:.1f}s.")

    pixels = np.stack(pixel_arrays, axis=0)  # (N, H, W, 3) uint8
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, ids=np.array(valid_ids), pixels=pixels)
    print(f"Saved {pixels.shape} ({pixels.nbytes / 1e6:.1f} MB uncompressed) to {OUT_PATH}")


if __name__ == "__main__":
    main()
