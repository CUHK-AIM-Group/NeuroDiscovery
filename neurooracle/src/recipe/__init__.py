"""Input Recipe: LLM brainstorm of computable quantities from a data inventory.

Single-shot, context-free: the LLM is told only what raw data is available
(modalities, representative gene/biomarker/score categories) and asked to
propose N computable quantities (e.g. "hippocampal volume from T1",
"frontal - parietal cortical thickness difference"). It is NOT told what
downstream task these will feed.

Recipes are persisted to JSON; they may later be promoted into the KG as
domain="recipe" nodes (two-step flow). Recipes do not participate in KGE
training.
"""

from .generator import (
    Recipe,
    build_inventory,
    brainstorm_recipes,
    brainstorm_recipes_batched,
    link_to_concepts,
    tag_atoms,
)

__all__ = ["Recipe", "build_inventory", "brainstorm_recipes",
           "brainstorm_recipes_batched", "link_to_concepts", "tag_atoms"]
