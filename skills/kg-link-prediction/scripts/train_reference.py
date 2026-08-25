"""Stable entrypoint for the kg-link-prediction skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.kg_link_prediction.train import main

if __name__ == "__main__":
    raise SystemExit(main())
