"""Stable entrypoint for the temporal-models skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.temporal_models.train import main

if __name__ == "__main__":
    raise SystemExit(main())
