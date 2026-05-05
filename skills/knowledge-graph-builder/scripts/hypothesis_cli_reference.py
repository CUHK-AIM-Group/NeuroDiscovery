"""Hypothesis Engine CLI Reference - for agent use.

The executable CLI lives in core/knowledge_graph/hypothesis_cli.py.
This file documents the usage patterns for agent reference.

Usage (run from project root):

    # Batch generate hypotheses across the entire graph
    python -m core.knowledge_graph.hypothesis_cli batch --output data/hypotheses.json

    # Load and re-rank saved hypotheses
    python -m core.knowledge_graph.hypothesis_cli rank --input data/hypotheses.json --top 20

    # Interactive queries
    python -m core.knowledge_graph.hypothesis_cli paths "hippocampus" "Alzheimer Disease"
    python -m core.knowledge_graph.hypothesis_cli bridge "hippocampus" --target-domain disease
    python -m core.knowledge_graph.hypothesis_cli contradictions --domain disease
    python -m core.knowledge_graph.hypothesis_cli gaps --domain-a neuroanatomy --domain-b disease
    python -m core.knowledge_graph.hypothesis_cli explore "hippocampus"
    python -m core.knowledge_graph.hypothesis_cli stats

Programmatic usage:
    from core.knowledge_graph import load_graph, HypothesisEngine

    kg = load_graph()
    engine = HypothesisEngine(kg)

    # batch generate
    hypotheses = engine.batch_generate()
    engine.save_hypotheses(hypotheses, "data/hypotheses.json")

    # load and rank
    hypotheses = engine.load_hypotheses("data/hypotheses.json")
    ranked = engine.rank_hypotheses(hypotheses, top_n=50)

    # each hypothesis has 4 scores:
    #   confidence_score - evidence quality
    #   novelty_score - how unexpected
    #   evidence_score - statistical strength
    #   testability_score - can NeuroClaw execute this?
    #   composite_score - combined ranking
"""
