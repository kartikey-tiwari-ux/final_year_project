"""Extract frozen DistilBERT [CLS] embeddings for every sample and cache to disk.

Rationale (CPU-only environment — see PROJECT_ARCHITECTURE.md / REPRODUCIBILITY.md):
running the transformer forward pass once per sample and caching the result is far
faster than re-running it every training step, and is a standard "linear probe"
transfer-learning setup. Run: python -m src.text_model.extract_distilbert_embeddings
"""
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data.mvsa_dataset import load_manifest
from src.preprocessing.text_cleaning import clean_tweet_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "distilbert_embeddings.npz"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
MAX_LENGTH = 96  # tweets are short; covers the vast majority without truncation


@torch.no_grad()
def extract(texts: list[str], tokenizer, model, device) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        # Mean-pool over tokens (masking padding) — more stable than raw [CLS] for
        # DistilBERT, which has no next-sentence-prediction pretraining to anchor CLS.
        last_hidden = out.last_hidden_state  # (batch, seq, hidden)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    device = torch.device("cpu")
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    df = load_manifest()
    texts = df["text"].apply(clean_tweet_text).tolist()

    print(f"Extracting embeddings for {len(texts)} samples (batch_size={BATCH_SIZE})...")
    start = time.time()
    embeddings = extract(texts, tokenizer, model, device)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / len(texts) * 1000:.1f} ms/sample)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        ids=df["id"].values,
        embeddings=embeddings,
    )
    print(f"Saved {embeddings.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
