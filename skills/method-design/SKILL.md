---
name: method-design
description: "Use this skill whenever the user wants to formalize a network architecture and derive theoretical components from a research idea. Triggers include: 'method design', 'design method', 'network architecture', 'formula derivation', 'method-design', 'theoretical framework', 'derive equations', 'compare alternatives', 'architecture tournament', 'rank designs', 'design space exploration', or any request to transform IDEA.md into a detailed METHOD.md. This skill is the **mandatory interface-layer method formalizer** in NeuroClaw: it reads IDEA.md, optionally drafts N alternative architectures and ranks them via a Bradley-Terry pairwise tournament (Tournament Mode), designs concrete network structures (layers, modules, connections), performs mathematical derivations (equations, loss functions, proofs), and always outputs a structured METHOD.md."
license: MIT License (NeuroClaw custom skill – freely modifiable within the project)
layer: interface
skill_type: workflow
dependencies:
  - academic-research-hub
---
# Method Design

## Overview
This skill implements the **Framework Formalization & Theoretical Derivation** process for the NeuroClaw method-design phase.

It acts as the Method Architect within the multi-agent framework:
- Reads the latest **IDEA.md** from the workspace.
- Designs the specific neural network structure (e.g., CNN backbone, attention modules, MRI-specific layers) tailored to the neuroscience task.
- Derives all necessary formulas (loss functions, gradients, convergence proofs, etc.) using symbolic or step-by-step reasoning.
- Produces a clean, publication-ready **METHOD.md** with sections: Architecture Diagram (text description), Detailed Layers, Mathematical Formulation, Implementation Notes, and Pseudocode.

If any part is ambiguous, it asks the user for clarification before proceeding.  
**Research use only** — the output is a mathematically rigorous METHOD.md ready for experiment-controller and paper-writing.

## Quick Reference (Method Flow)

| Step | Description                          | Output File            |
|------|--------------------------------------|------------------------|
| 1. Read & Parse | Load and analyze IDEA.md            | 01_idea_summary.md    |
| 2a. Draft Alternatives (Tournament Mode) | Sketch N candidate architectures | 02a_candidates.json |
| 2b. Lit Probe (Tournament Mode) | Per-candidate literature check via academic-research-hub | 02b_lit_check.md |
| 2c. Tournament (Tournament Mode) | Pairwise LLM-judge + Bradley-Terry ranking | 02c_ranking.csv |
| 2. Architecture Design | Define network layers & modules (top-1 from tournament if used) | 02_architecture.md |
| 3. Formula Derivation | Derive equations & proofs           | 03_formulas.md        |
| 4. Pseudocode & Notes | Generate implementation details     | 04_pseudocode.md      |
| 5. Finalize | Compile and polish                  | METHOD.md             |

## Installation
```bash
# Place files in: skills/method-design/
```

## Tournament Mode (Optional Step 2 Expansion)

When the user asks to "compare alternatives", "rank designs", "explore the design space",
or provides multiple architecture sketches, run a Bradley-Terry pairwise tournament
instead of single-shot architecture selection.

### Inputs
- N >= 2 architecture candidates, each with `name` and `rationale` fields
- Saved as `02a_candidates.json` (JSON array)

### Mechanism
1. Generate all C(N,2) unordered pairs (capped at 300 for cost)
2. For each pair, prompt an LLM judge with the criteria in
   `scripts/tournament.py::ARCHITECTURE_JUDGE_SYSTEM_PROMPT`
   (soundness > novelty > tractability > NeuroClaw compatibility > falsifiability)
3. Aggregate verdicts into a Bradley-Terry strength score via
   `choix.ilsr_pairwise` (alpha=0.1)
4. Pick top-1 for downstream Step 3 (formula derivation)

### Invocation
```bash
python skills/method-design/scripts/tournament.py \
    --candidates 02a_candidates.json \
    --output 02c_ranking.csv \
    --backend openai --model gpt-4o-mini
```

### When to skip Tournament Mode
- User has already named a single specific architecture
- N < 2 candidates available
- User explicitly asks for single-shot design

### Cost notes
- Each pair costs ~1 LLM call; total cost = min(C(N,2), 300) calls
- For N=5: 10 calls (~$0.01 with gpt-4o-mini)
- For N=10: 45 calls
- Concurrency capped at 16 by default

### Provenance
Pairwise tournament approach adapted from FutureHouse/Robin
(arXiv:2505.13400, Nature 2026), MIT License. NeuroClaw adaptation:
LLM-backend-agnostic via injectable `llm_call`, neuroimaging-specific
judge criteria, no Edison platform dependency.

## Important Notes & Limitations
- Always starts from the latest IDEA.md; stops and prompts if missing.
- Every step saved as numbered .md files for transparency and resumption.
- Uses symbolic derivation (no hallucinated math); can call claw-shell or code tools internally for verification.
- Final output always saved as `METHOD.md` in workspace root.
- Diagrams are described in text (PlantUML/Mermaid ready); no image generation here.

## When to Call This Skill
- Immediately after research-idea skill completes IDEA.md
- When the user wants to specify network details or mathematical foundation
- Before experiment-controller or paper-writing

## Complementary / Related Skills
- `research-idea` → provides IDEA.md (input)
- `experiment-controller` → consumes METHOD.md
- `paper-writing` → consumes METHOD.md

## Reference
NeuroClaw architecture (section 1.5 method-design skill).  
Flow: IDEA.md parsing → architecture design → equation derivation → pseudocode → METHOD.md  

Created At: 2026-03-24 00:00 HKT
Last Updated At: 2026-05-27 (Tournament Mode added)  
Author: chengwang96