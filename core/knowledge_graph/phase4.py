"""Phase 4: 假设扩展

文献新颖性检查 + 进化变异扩展假设空间。

Usage:
    python -m core.knowledge_graph.phase4 evolve --input core/knowledge_graph/data/hypotheses.json
    python -m core.knowledge_graph.phase4 novelty --input core/knowledge_graph/data/hypotheses.json
"""

from .src.novelty_checker import NoveltyChecker
from .src.evolution_engine import EvolutionEngine

if __name__ == "__main__":
    from .src.hypothesis_cli import main
    main()
