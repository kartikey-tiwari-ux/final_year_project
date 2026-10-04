"""Download the Memotion Dataset 7k (fallback dataset — see DATASET_SELECTION.md).

Sharma et al., "SemEval-2020 Task 8: Memotion Analysis - the Visuo-Lingual Metaphor!"
~6,992 memes with OCR'd caption text. Reported CC BY 4.0 (re-verify on the live Kaggle
page). Only the overall-sentiment label (Task A) is used by this project's main
comparison, per the scoping note in DATASET_SELECTION.md.

Requires a Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token,
which downloads kaggle.json. Place it at %USERPROFILE%\\.kaggle\\kaggle.json, or set
KAGGLE_USERNAME / KAGGLE_KEY in .env (see .env.example).
"""
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "memotion_7k"
KAGGLE_DATASET = "williamscott701/memotion-dataset-7k"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(RAW_DIR),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(
            "\nFAILED. Most likely cause: missing/invalid Kaggle API credentials.\n"
            "Set them up at https://www.kaggle.com/settings -> API -> Create New Token,\n"
            "save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json, then retry."
        )
        return 1

    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        print("Download reported success but no .zip file found — check output above.")
        return 1

    print(f"Extracting {zips[0].name}...")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(RAW_DIR)
    print(f"Done. Extracted to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
