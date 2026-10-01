# Idea AutoResearch failure diagnosis

## Evidence inspected

Read-only local workbench request `acd5af04-4466-41a9-ae67-1c69e39d70ca`,
2026-09-22 18:48:41–18:57:20 UTC, matches the screenshot's cerebellum/ADHD
question. It contains `autoresearch: null`, no AutoResearch evidence IDs, 16 tool
invocations and nine recorded model calls. It followed the ordinary eight-round
tool loop plus final-response path, not the AutoResearch lifecycle. The saved
record does not preserve the incoming selected-mode payload, so it cannot prove
whether the user selection was lost before dispatch or the old runtime ignored it.
No new inference was dispatched to reproduce the research.

The tools repeatedly invoked bare `python` without a usable PATH entry, then
searched for a virtual environment. A recursive `findstr /s` across Markdown/JSON
timed out at 180 seconds. Piping results through exclusions for node_modules and
.venv filters output **after scanning**, not before traversal. There was no actual
bounded concept/evidence API query. The assistant appropriately declined to
invent hypotheses or PMID support; the final prose was not successful research.

The running development backend was PID 36492, port 4170, launched at 18:47:55 UTC
with the bundled interpreter. Its `/api/harness` response has no current runtime
contract, confirming newer on-disk changes are not active until restart. Later
logs show the concept graph loaded, while `/api/kg/shared-claims` returned 503.
Graph presence and reviewed-evidence service availability are separate conditions;
downloading the same graph is not evidence that the reviewed service is repaired.

## Corrections

- AutoResearch mode is saved per conversation and restored when switching or
  refreshing. New chats still start Off. Each request binds to its conversation's
  mode rather than a stale global menu value. Queued/resumed requests retain scope.
- Before AutoResearch dispatch, the renderer checks a backend runtime contract.
  Old backends receive an explicit restart message without posting research. The
  backend acknowledges scope and returns it with results; mismatches request Stop.
- Shell children prepend the actual runtime interpreter directory to PATH and
  expose `NEURODISCOVERY_PYTHON`. Desktop prompts name the actual executable.
- Idea HTTP requests receive a bounded local KG retrieval route and warnings
  against broad recursive corpus/environment scans. This does not fabricate graph
  evidence, silently relax novelty gates, or authorize a graph rebuild/download.
- The graph actions menu exposes Hugging Face download in both concept and claim
  views, even when update checks fail. Download still requires confirmation and
  uses the existing server route/repository. Idle download status is not labeled
  successful. No actual graph file was downloaded or replaced during this fix.

Restart the development client/backend, select Idea in the intended conversation,
and explicitly submit the research again. The old run is not retroactively a
resumable AutoResearch checkpoint. Research success still depends on real usable
retrieval/evidence and acceptance, not merely removing the ordinary round cap.
