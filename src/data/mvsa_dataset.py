"""Load the MVSA-Single manifest built by build_mvsa_manifest.py."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "mvsa_single_manifest.csv"

LABEL_NAMES = ["positive", "neutral", "negative"]
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_NAMES)}


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run: python -m src.data.build_mvsa_manifest"
        )
    df = pd.read_csv(MANIFEST_PATH)
    df["image_path"] = df["image_path"].apply(lambda p: str(PROJECT_ROOT / p))
    return df


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    assert split in {"train", "val", "test"}
    return df[df["split"] == split].reset_index(drop=True)
