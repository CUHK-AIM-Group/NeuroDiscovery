"""Phase 3: 假设引擎

批量生成、评分、排序可检验假设。包含 Critic Agent 迭代审查。

Usage:
    python -m core.knowledge_graph.phase3 batch --output core/knowledge_graph/data/hypotheses.json
    python -m core.knowledge_graph.phase3 rank --input core/knowledge_graph/data/hypotheses.json --top 20
    python -m core.knowledge_graph.phase3 imaging-batch --dataset UKB
    python -m core.knowledge_graph.phase3 paths "hippocampus" "Alzheimer Disease"
    python -m core.knowledge_graph.phase3 critic --input core/knowledge_graph/data/hypotheses.json --top 20
"""

from .src.hypothesis_engine import HypothesisEngine, Hypothesis
from .src.critic_agent import CriticAgent

if __name__ == "__main__":
    from .src.hypothesis_cli import main
    main()
