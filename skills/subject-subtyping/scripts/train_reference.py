"""Stable entrypoint for the subject-subtyping skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.subtyping.train import main

if __name__ == "__main__":
    raise SystemExit(main())
