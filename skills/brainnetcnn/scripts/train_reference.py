"""Canonical BrainNetCNN skill entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.brainnetcnn.scripts.train import main


if __name__ == "__main__":
    main()
