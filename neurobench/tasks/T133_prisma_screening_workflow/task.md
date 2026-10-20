# T133_prisma_screening_workflow: PRISMA-Style Systematic Screening
## Task Description

Execute a PRISMA-style screening workflow for a given review query: retrieve
candidate papers, apply staged inclusion/exclusion criteria, and report the
funnel counts required for a PRISMA flow diagram.

## Input Requirement

Required input(s):

- Review query string and inclusion/exclusion criteria file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Stage counts must be reported: identified -> deduplicated -> title/abstract
  screened -> full-text assessed -> included.
- Every exclusion at full-text stage needs a one-line reason.
- LLM-assisted screening is allowed but criteria must be applied verbatim
  from the criteria file.
- Save all generated artifacts to:
  - benchmark_results/T133_prisma_screening_workflow/

## Expected Output

Expected output artifact(s):

- `screened_papers.csv` (one row per paper: stage reached, decision, reason)
- `prisma_flow.json` (counts per stage, ready to plot)
- `included_summary.md` (table of included papers with key fields)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
