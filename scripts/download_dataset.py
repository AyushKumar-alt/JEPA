"""
Dataset Download and Integrity Verification Utility for BEV Decision Mix.
Downloads the pinned v1.0-150k revision from Hugging Face if files are missing.
"""

from pathlib import Path
import hashlib
import urllib.request
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bev-decision"

DATASET_FILES = {
    "train.parquet": {
        "url": "https://huggingface.co/datasets/avbiswas/bev-decision/resolve/main/data/train.parquet",
        "sha256": "1b163182db155ba986c8ab8d3d582ac09ec1b1dcbc13c65d048e61379be9fe5b",
        "expected_rows": 125614,
    },
    "test.parquet": {
        "url": "https://huggingface.co/datasets/avbiswas/bev-decision/resolve/main/data/test.parquet",
        "sha256": "eba9f9df6ad54ccdc3aa6516ab97e4f9dea4030422e0dc90510b85e462206a61",
        "expected_rows": 24386,
    },
}


def verify_file(filepath: Path, expected_sha256: str, expected_rows: int) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash != expected_sha256:
        print(f"Checksum mismatch for {filepath.name}: {actual_hash} != {expected_sha256}")
        return False
    table = pq.read_table(str(filepath))
    if table.num_rows != expected_rows:
        print(f"Row count mismatch for {filepath.name}: {table.num_rows} != {expected_rows}")
        return False
    return True


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, meta in DATASET_FILES.items():
        dest = DATA_DIR / filename
        if verify_file(dest, meta["sha256"], meta["expected_rows"]):
            print(f"[OK] {filename} is present, bitwise verified, and contains {meta['expected_rows']:,} rows.")
            continue

        print(f"Downloading {filename} from {meta['url']}...")
        urllib.request.urlretrieve(meta["url"], str(dest))
        if verify_file(dest, meta["sha256"], meta["expected_rows"]):
            print(f"[SUCCESS] Downloaded and verified {filename}.")
        else:
            raise RuntimeError(f"Failed to verify integrity for {filename} after download!")


if __name__ == "__main__":
    ensure_dataset()
