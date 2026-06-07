# Phase 2 Rerun Plan

This note preserves the current Phase-2 execution plan across long Codex sessions.

## Model Cascade

Production claim extraction should use this adaptive cascade:

1. `claude-sonnet-4-6`
2. `claude-opus-4-6`
3. `claude-opus-4-7`
4. `deepseek-v3.2`

Do not include the following until their proxy channels are retested:

- `qwen3.6-plus`
- `gemini-3.1-pro-preview`
- `gemini-3-flash-preview`
- `gemini-3.5-flash`
- `gpt-5.2`
- `minimax-m2.7`
- `grok-4-20-non-reasoning`

`gpt-5.4` and `gpt-5.5` can produce reasonable claims, but current runs show
timeouts and sparse/empty extraction, so they are not production defaults.

## Execution Order

1. Run a 50-paper cached smoke test using the production cascade.
   - Check extraction errors, JSON parse errors, zero-claim rate, method/procedure
     leakage, vague endpoint drops, and background claim handling.

2. Rerun the existing base abstract cache in `neurooracle/data/full_snapshot_v2/`.
   - Reuse cached PubMed abstracts.
   - Re-extract claims because the atom-aware prompt and endpoint controls changed.

3. Process the base cache in batches.
   - Start with batches of 1000-2000 papers.
   - Start with `--max-workers 2` or `--max-workers 3`; increase only if stable.
   - Review `claims_added`, `claims_skipped_noise`, `claims_skipped_background`,
     `entities_dropped`, `predicates_refined`, and extraction errors after each batch.

4. After the base KG is rebuilt, fetch additional PubMed abstracts.
   - Keep the broad public KG objective.
   - Use purpose-aware queries around genes, imaging markers, cognitive/clinical
     outcomes, PRS, PET/MRI/DTI/fMRI, and disease-relevant bridges.

5. Add case-study enrichment after the broad graph is stable.
   - Case-study logic should target `GENE_TARGET -> IMAGING_MARKER -> OUTCOME`.
   - Disease is implicit in the outcome/query context.

6. For each production batch, produce a compact QA sample.
   - Include genetic risk / PRS claims, imaging marker claims, cognitive/clinical
     phenotype claims, dropped entity examples, and method/procedure leakage checks.
