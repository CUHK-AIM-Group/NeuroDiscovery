"""Phase 3: hypothesis engine.

Batch-generate, score, and rank testable hypotheses. Includes a Critic Agent
iterative review.

Usage:
    python -m neurooracle.phase3 batch --output neurooracle/data/hypotheses.json
    python -m neurooracle.phase3 rank --input neurooracle/data/hypotheses.json --top 20
    python -m neurooracle.phase3 imaging-batch --dataset UKB
    python -m neurooracle.phase3 paths "hippocampus" "Alzheimer Disease"
    python -m neurooracle.phase3 critic --input neurooracle/data/hypotheses.json --top 20
"""

from .src.hypothesis_engine import HypothesisEngine, Hypothesis
from .src.critic_agent import CriticAgent

if __name__ == "__main__":
    from .src.hypothesis_cli import main
    main()
