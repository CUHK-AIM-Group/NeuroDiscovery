"""Abstract Scorer interface so we can swap ComplEx → ULTRA / Gamma later
without touching CLI or plausibility code."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Scorer(ABC):
    """A KG link-prediction scorer."""

    @abstractmethod
    def score_triple(self, source_id: str, relation_type: str, target_id: str) -> float:
        """Return P((s, p, o) is true), in [0, 1]."""

    def score_batch(
        self, triples: list[tuple[str, str, str]]
    ) -> list[float]:
        """Default: per-triple loop. Subclasses can override for speed."""
        return [self.score_triple(s, p, o) for s, p, o in triples]

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of this scorer / checkpoint, written into hyp.metadata."""
