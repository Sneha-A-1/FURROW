"""Download the CAF data set and verify its SHA-256 checksum (no processing).

Usage: python scripts/download_data.py [--raw data/raw]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.ingest import download  # noqa: E402
from furrow.data.ingest import CAF_SHA256, sha256  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    a = ap.parse_args()
    rda = download(Path(a.raw))
    h = sha256(rda)
    print(json.dumps({"file": str(rda), "sha256": h, "expected": CAF_SHA256, "ok": h == CAF_SHA256}, indent=2))
    sys.exit(0 if h == CAF_SHA256 else 1)
