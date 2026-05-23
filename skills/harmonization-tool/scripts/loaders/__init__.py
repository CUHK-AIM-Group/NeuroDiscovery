"""Real-data loaders for harmonization pilots."""
from .adhd200_real import load_adhd200_connectomes
from .abide_real import load_abide_connectomes

__all__ = ["load_adhd200_connectomes", "load_abide_connectomes"]
