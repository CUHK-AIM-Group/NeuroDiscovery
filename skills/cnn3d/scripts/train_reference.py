"""Stable entrypoint for the cnn3d skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.cnn3d.train import main

if __name__ == "__main__":
    raise SystemExit(main())
