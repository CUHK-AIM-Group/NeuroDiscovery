"""Dependency-free survival metrics."""

from __future__ import annotations

import numpy as np


def concordance_index(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    """Harrell's C-index where larger score means greater event risk."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    comparable = 0
    concordant = 0.0
    for i in range(len(duration)):
        if not event[i]:
            continue
        later = np.flatnonzero(duration > duration[i])
        comparable += len(later)
        concordant += np.sum(risk[i] > risk[later])
        concordant += 0.5 * np.sum(risk[i] == risk[later])
    return float(concordant / comparable) if comparable else 0.5
