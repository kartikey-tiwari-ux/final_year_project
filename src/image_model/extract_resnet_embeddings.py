"""Extract frozen ResNet18 (ImageNet-pretrained) penultimate-layer embeddings for
every sample and cache to disk. Same rationale as the DistilBERT embedding cache —
see PROJECT_ARCHITECTURE.md. Missing/corrupt images are dropped and logged, not
silently skipped. Run: python -m src.image_model.extract_resnet_embeddings
"""
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from src.data.mvsa_dataset import load_manifest
from src.preprocessing.image_preprocessing import load_image_safely, pretrained_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "resnet18_embeddings.npz"
BATCH_SIZE = 32


def build_feature_extractor():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # drop the 1000-way classifier head; keep 512-dim pooled features
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    df = load_manifest()
    model = build_feature_extractor()
    device = torch.device("cpu")
    model.to(device)

    # Stream in batches rather than materializing all ~4.5k transformed tensors at
    # once (224x224x3 float32 each -> ~2.7GB if held in memory simultaneously).
    valid_ids = []
    dropped = []
    embeddings = []
    batch_tensors, batch_ids = [], []

    def flush_batch():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        out = model(batch)
        embeddings.append(out.cpu().numpy())
        valid_ids.extend(batch_ids)
        batch_tensors.clear()
        batch_ids.clear()

    print(f"Extracting ResNet18 embeddings for up to {len(df)} images (batch_size={BATCH_SIZE})...")
    start = time.time()
    for _, row in df.iterrows():
        img = load_image_safely(Path(row["image_path"]))
        if img is None:
            dropped.append(row["id"])
            continue
        batch_tensors.append(pretrained_transform(img))
        batch_ids.append(row["id"])
        if len(batch_tensors) == BATCH_SIZE:
            flush_batch()
    flush_batch()

    embeddings = np.concatenate(embeddings, axis=0)
    elapsed = time.time() - start
    print(f"Loaded {len(valid_ids)} images, dropped {len(dropped)} missing/corrupt: {dropped[:10]}")
    print(f"Done in {elapsed:.1f}s ({elapsed / max(len(valid_ids),1) * 1000:.1f} ms/image)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, ids=np.array(valid_ids), embeddings=embeddings)
    print(f"Saved {embeddings.shape} to {OUT_PATH}")
    if dropped:
        dropped_path = PROJECT_ROOT / "results" / "dropped_images.json"
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        json.dump({"dropped_ids": [str(d) for d in dropped], "count": len(dropped)}, open(dropped_path, "w"))
        print(f"Logged {len(dropped)} dropped ids to {dropped_path}")


if __name__ == "__main__":
    main()
