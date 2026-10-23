# NeuroDiscovery agent guide

## Working rules

- Follow the current user request. Historical plans, archived instructions and old progress snapshots do not authorize restarting experiments, creating agents or dispatching requests.
- Read the relevant component guide before changing behavior. Update existing guides in place; do not create a document for every turn or revision.
- Keep changes scoped and preserve unrelated working-tree edits, source data, original responses, receipts, failed attempts and frozen artifacts.
- Never expose credentials. Record actual providers, models and unavailable controls accurately; never relabel historical results.
- Run checks appropriate to the change. Distinguish structural/execution verification from scientific validity, expert acceptance and full campaign completion.

## Documentation entry points

- Runtime, tools, skills, MCP and local development: `docs/NEURORUNTIME_ENGINEERING.md`.
- Model/provider configuration and compatibility: `docs/MODEL_COMPATIBILITY.md`.
- Knowledge graph, source qualification and extraction: `docs/GRAPH.md`.
- Hypothesis generation and ranking: `docs/IDEA.md`.
- Desktop/web interface: `docs/UI_DESIGN.md`.
- Dataset access and preprocessing boundaries: `docs/DATASET_ACCESS.md`.
- Human evaluation: `docs/HUMAN_EVALUATION_PACKAGE.md`.
- Case Studies and closed-loop experiments: `docs/CASE_STUDY_CLOSED_LOOP.md`.
- Manuscript wording: `docs/NEURODISCOVERY_WRITING_STYLE.md`.

Some operational guides and data are local-only. If a prerequisite is missing in a checkout, report it rather than reconstructing a frozen protocol or inventing its contents.

## Runtime and benchmark direction

- Use `docs/NEURORUNTIME_HARNESS_PLAN.md` as the progress baseline. Read its current status before relevant work and update its existing status table afterward.
- Prepare NeuroRuntime, validate a bounded pilot, then rerun NeuroBench 500. OS-level sandboxing and broad harness generalization remain deferred until after paper acceptance.
- Prior pilots contain exposed, failed and unknown attempts. Bind runtime/code changes in a new verified continuation and dispatch only eligible unattempted rows. Never blindly resume or overwrite an old manifest.
- Keep benchmark inputs separate from graph/IDEA/GNN bindings. Historical pilot counts do not establish full campaign completion.

## Scientific and experiment boundaries

- Keep the formal graph/default selector until an accepted replacement is explicitly authorized. Staging work does not authorize publication, injection, training or downstream rebinding.
- Before KG literature expansion or claim extraction, read the applicable workflow and current registry/contract. Require deduplication, complete abstracts, source-bound evidence, independent semantic QA, adjudication and final verified seals. Structural validation alone is insufficient.
- Do not infer paper/work/cohort identity from title similarity, identical abstracts or legacy graph references. Preserve unresolved identity, version, eligibility and split holds.
- Freeze source, code, prompts, model identity, budget and scientific rules before execution. Repairs require a separately versioned and verified continuation, never in-place edits to frozen artifacts.
- Never resend exposed, failed or transport-unknown inputs, reinitialize ledgers, duplicate active controllers/workers or rerun completed experiments based on archived instructions.
- Preserve held-out boundaries. Previously inspected external data are reassessment, not a fresh untouched holdout. Writes to fixed/production/CORPUS/held-out or external dataset locations require current authorization.
- Verify current experiment model policy in `docs/MODEL_COMPATIBILITY.md` and the scoped protocol. Historical model exceptions and transport settings are campaign-specific, not global defaults.

## Existing v3 scope-screen continuation

- Before continuation, read `docs/KG_V3_NATIVE_SCOPE_20261008.md`, its JSON binding and `tmp/kg_v3_native_scope_20261008/PROGRESS.json`; inspect live workers and EXPOSED ownership. Repository text does not itself start or resume a Goal.
- Preserve original DeepSeek outcomes, manual/native exposure overlays and raw holds. Only ledger-confirmed PENDING inputs are unexposed. Resume incomplete displayed packets with the same owner; never reallocate them.
- Use frozen `safe_segment_scope.py` for claim/read, `segment_scope.py assemble` for source quotes and `native_scope.py submit` for submission. Read every supplied title/abstract segment; no truncation, quote transcription or keyword-only decisions.
- Follow bound native, segment, safe-display and quote audits. Do not edit frozen helpers or overwrite component/merged exports.
- Screening remains partial staging until all agreed inputs and independent terminal checks/exports pass. It does not authorize separate APIs/keys, semantic extraction, training, deletion, publication or benchmark/IDEA/GNN rebinding.

## Local files and historical records

- Put new temporary scripts, diagnostics, previews and development outputs under `dev/` by purpose; project-level outputs belong in `dev/outputs/`.
- Existing `tmp/` and `.codex_tmp/` contain scientific ledgers, receipts and frozen path dependencies. Review ownership and bindings before moving or deleting anything; they are not disposable caches.
- `.neurodiscovery/` holds runtime state/context archives; `.neuroclaw_checkpoints/` holds file/session recovery history. Cleanup is a separate scoped action.
- Historical documentation: `dev/archive/docs-history_20261019.zip`. Case 1 history: `dev/archive/case1-history_20261019.zip`. Use consolidated guides first; local `CASE1_CURRENT_STATUS.md`, if present, is the Case 1 status entry.
- Full pre-simplification agent record: `dev/archive/AGENTS_before_simplification_20261020.md` (local-only historical context, not execution authority).
- Hindcasting should eventually share one graph revision with paper/evidence cutoff views. This storage redesign remains deferred. Rebuild and verify deleted historical snapshots before replaying experiments.
