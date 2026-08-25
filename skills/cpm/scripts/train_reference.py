"""Stable entrypoint for the cpm skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.cpm.train import main

if __name__ == "__main__":
    raise SystemExit(main())
