"""Download (if needed), verify and prepare the CAF data set.

Usage: python scripts/prepare_data.py [--raw data/raw] [--out data/processed]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.ingest import download  # noqa: E402
from furrow.data.pipeline import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    rda = download(Path(a.raw))
    stats = prepare(rda, Path(a.out))
    print(json.dumps(stats, indent=2))
