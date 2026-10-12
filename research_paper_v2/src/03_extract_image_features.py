"""
Image feature extraction for the three splits, in two conditions:
  - "sharp": the original meme image, resized/normalized
  - "blur":  the same image with a strong Gaussian blur applied BEFORE resize,
             destroying fine edges (incl. embedded meme-text strokes) while
             preserving coarse colour/composition -- the key ablation for the
             "embedded-text leakage" research question.

For each condition, extracts frozen ImageNet-pretrained ResNet18
penultimate-layer embeddings (512-dim) -> for B2 / B2-Blur and fusion C / C-Blur.

Also saves 96x96 tensors (sharp only) for the from-scratch CNN (B1), and
computes a Laplacian-variance "focus measure" on a sample of sharp vs blurred
images as an objective, OCR-free check that the blur meaningfully destroys
fine (text-scale) detail.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import laplace
from torchvision import models, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, seed_everything

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
SPLITS = ["train", "validation", "test"]

BLUR_RADIUS = 14  # Gaussian blur radius (px) applied at native resolution before resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resnet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

cnn_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])


def load_split_df(split):
    return pd.read_parquet(RAW / f"{split}.parquet")


def blurred(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))


def laplacian_variance(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float64)
    lap = laplace(g)
    return float(lap.var())


def build_resnet():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


@torch.no_grad()
def resnet_embed_split(df, split, resnet, condition, batch_size=32):
    paths = [ROOT / p for p in df["image_path"].tolist()]
    embs = []
    batch_imgs = []

    def flush():
        if not batch_imgs:
            return np.zeros((0, 512), dtype=np.float32)
        x = torch.stack(batch_imgs, dim=0)
        out = resnet(x)
        return out.numpy().astype(np.float32)

    for p in paths:
        img = Image.open(p).convert("RGB")
        if condition == "blur":
            img = blurred(img)
        batch_imgs.append(resnet_tf(img))
        if len(batch_imgs) == batch_size:
            embs.append(flush())
            batch_imgs = []
    if batch_imgs:
        embs.append(flush())

    return np.concatenate(embs, axis=0)


def build_cnn_tensors(df, split):
    paths = [ROOT / p for p in df["image_path"].tolist()]
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        arrs.append(cnn_tf(img).numpy())
    arr = np.stack(arrs, axis=0).astype(np.float32)
    np.save(PROC / f"cnn96_{split}.npy", arr)
    print(f"{split}: cnn96 tensor {arr.shape}")


def blur_validation_metric(dfs, n_sample=150, seed=42):
    rng = np.random.RandomState(seed)
    df = dfs["test"]
    idxs = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    sharp_vars, blur_vars = [], []
    for i in idxs:
        p = ROOT / df.iloc[i]["image_path"]
        img = Image.open(p).convert("RGB")
        sharp_vars.append(laplacian_variance(img))
        blur_vars.append(laplacian_variance(blurred(img)))
    sharp_vars = np.array(sharp_vars)
    blur_vars = np.array(blur_vars)
    result = {
        "n_sample": int(len(idxs)),
        "blur_radius_px": BLUR_RADIUS,
        "mean_laplacian_variance_sharp": float(sharp_vars.mean()),
        "mean_laplacian_variance_blurred": float(blur_vars.mean()),
        "pct_reduction": float(100.0 * (1 - blur_vars.mean() / sharp_vars.mean())),
    }
    with open(PROC / "blur_validation_metric.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Blur validation metric:", result)


def main():
    seed_everything()
    dfs = {s: load_split_df(s) for s in SPLITS}

    print("Building ResNet18 (ImageNet-pretrained, frozen)...")
    resnet = build_resnet()

    for condition in ["sharp", "blur"]:
        for s in SPLITS:
            t0 = time.time()
            emb = resnet_embed_split(dfs[s], s, resnet, condition)
            dt = time.time() - t0
            np.save(PROC / f"resnet18_{condition}_{s}.npy", emb)
            print(f"[{condition}] {s}: {emb.shape} in {dt:.1f}s "
                  f"({1000*dt/len(emb):.1f} ms/sample)")

    for s in SPLITS:
        build_cnn_tensors(dfs[s], s)

    blur_validation_metric(dfs)
    print("Image feature extraction complete.")


if __name__ == "__main__":
    main()
