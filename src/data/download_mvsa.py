"""Download the MVSA-Single dataset (primary dataset — see DATASET_SELECTION.md).

MVSA-Single: Niu, Zhu, Pang & El Saddik, "Sentiment Analysis on Multi-View Social
Data," MMM 2016. ~4,869-5,129 Twitter text-image pairs, 3-class sentiment label
(positive/neutral/negative), one human annotator per pair.

No formal open license is published by the maintainers; usage here is strictly for
non-commercial academic coursework, with citation (see ETHICS_AND_PRIVACY.md,
DATASET_SELECTION.md). This script does not redistribute the data — it only fetches
it into the local, gitignored data/raw/ directory for this project's own use.

If the OneDrive link below is unreachable (link rot is a known risk for informally
distributed academic datasets), fall back to download_memotion.py instead and update
REPRODUCIBILITY.md noting which source actually worked.
"""
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "mvsa_single"

# Official MVSA-Single OneDrive share link (see DATASET_SELECTION.md for provenance).
# OneDrive personal share links can be coerced into a direct-download stream by
# appending "&download=1" to the share URL.
ONEDRIVE_SHARE_URL = (
    "https://portland-my.sharepoint.com/:u:/g/personal/"
    "shiaizhu2-c_my_cityu_edu_hk/Ebcsf1kUpL9Do_u4UfNh7CgBC19i6ldyYbDZwr6lVbkGQQ"
)


def _direct_download_url(share_url: str) -> str:
    sep = "&" if "?" in share_url else "?"
    return f"{share_url}{sep}download=1"


def download(dest_zip: Path) -> bool:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = _direct_download_url(ONEDRIVE_SHARE_URL)
    print(f"Attempting MVSA-Single download from: {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "text/html" in content_type:
                print(
                    "FAILED: server returned an HTML page, not a file "
                    "(the share link likely needs interactive sign-in or has expired)."
                )
                return False
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(dest_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
            print(f"Downloaded {written / 1e6:.1f} MB (reported total: {total / 1e6:.1f} MB)")
            return True
    except requests.RequestException as e:
        print(f"FAILED: {e}")
        return False


def main() -> int:
    dest_zip = RAW_DIR / "mvsa_single.zip"
    if not download(dest_zip):
        print(
            "\nMVSA-Single automatic download failed. Next steps:\n"
            "  1. Try the link manually in a browser: "
            f"{ONEDRIVE_SHARE_URL}\n"
            "  2. If it's truly unreachable, fall back to:"
            " python -m src.data.download_memotion\n"
            "     and update REPRODUCIBILITY.md noting the fallback was used."
        )
        return 1

    print("Extracting...")
    with zipfile.ZipFile(dest_zip) as zf:
        zf.extractall(RAW_DIR)
    print(f"Done. Extracted to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
