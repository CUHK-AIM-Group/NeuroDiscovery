"""Phase 4: hypothesis expansion.

Literature novelty check + evolutionary mutation to expand the hypothesis space.

Usage:
    python -m neurooracle.phase4 evolve --input neurooracle/data/hypotheses.json
    python -m neurooracle.phase4 novelty --input neurooracle/data/hypotheses.json
"""

from .src.novelty_checker import NoveltyChecker
from .src.evolution_engine import EvolutionEngine

if __name__ == "__main__":
    from .src.hypothesis_cli import main
    main()
