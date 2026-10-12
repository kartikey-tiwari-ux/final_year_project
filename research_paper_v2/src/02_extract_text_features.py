"""
Text feature extraction for the three splits:
  - TF-IDF (1-2 grams, max 10k features), fit on train only  -> for A1
  - Frozen DistilBERT (distilbert-base-uncased) mean-pooled embeddings -> for A2 and fusion
Saves .npy / .joblib artifacts under data/processed/.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, LABEL2ID, seed_everything

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]


def load_splits():
    return {s: pd.read_parquet(RAW / f"{s}.parquet") for s in SPLITS}


def build_tfidf(dfs):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    vec.fit(dfs["train"]["text_corrected"].fillna(""))
    for s in SPLITS:
        X = vec.transform(dfs[s]["text_corrected"].fillna(""))
        np.save(PROC / f"tfidf_{s}.npy", X.toarray().astype(np.float32))
    joblib.dump(vec, PROC / "tfidf_vectorizer.joblib")
    print("TF-IDF done. Vocab size:", len(vec.vocabulary_))


@torch.no_grad()
def distilbert_embed(texts, tokenizer, model, batch_size=16):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt")
        out = model(**enc)
        hidden = out.last_hidden_state  # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(pooled.numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def build_distilbert(dfs):
    print("Loading distilbert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    for s in SPLITS:
        texts = dfs[s]["text_corrected"].fillna("").tolist()
        t0 = time.time()
        emb = distilbert_embed(texts, tokenizer, model)
        dt = time.time() - t0
        np.save(PROC / f"distilbert_{s}.npy", emb)
        print(f"{s}: {emb.shape} in {dt:.1f}s ({1000*dt/len(texts):.1f} ms/sample)")


def save_labels(dfs):
    for s in SPLITS:
        y = dfs[s]["sentiment_3class"].map(LABEL2ID).to_numpy()
        np.save(PROC / f"labels_{s}.npy", y)


def main():
    seed_everything()
    dfs = load_splits()
    save_labels(dfs)
    build_tfidf(dfs)
    build_distilbert(dfs)
    print("Text feature extraction complete.")


if __name__ == "__main__":
    main()
