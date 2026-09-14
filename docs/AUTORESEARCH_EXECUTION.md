# AutoResearch execution contract

Enabling a scope in the client and sending a task selects persistent execution,
not a plan-and-confirm workflow. Data, model, idea and end-to-end boundaries are
unchanged. `/help` remains read-only guidance and never launches a model or job.

## Continue through delivery

- Inspect supplied inputs and relevant existing artifacts first. Optional settings
  are not a mandatory questionnaire: use and disclose reasonable defaults.
- Continue normal in-scope steps, diagnose recoverable errors, and try safe repairs
  without asking whether to continue. Duration alone is not a confirmation gate.
- A text-only model reply is progress, not completion. The runtime requests the
  next action in the same turn, retaining tool context and token accounting.
- AutoResearch has no default total iteration cutoff. Ordinary chat retains its
  existing eight-round default. An explicitly set positive
  `NEUROCLAW_MAX_TOOL_ITERATIONS` caps AutoResearch iterations as written, without
  the ordinary-chat 20-round clamp. An invalid explicit value blocks startup
  instead of silently enabling unlimited execution. Provider retries and
  per-command timeouts remain bounded; existing user/scientific budgets bind.
- The runtime never automatically increases budgets, changes providers, replays
  a whole failed run, or restarts completed/frozen experiments.

## Completion and blockers

The tool-enabled model calls `finish_autoresearch` alone after execution and
validation results have returned. `completed` requires nonempty workspace
deliverable files, actual tool evidence IDs and a validation report. A workspace
manifest can reference authorized outputs stored elsewhere. The runtime checks
file existence, size, workspace containment and evidence ID validity; it does
**not** independently certify scientific correctness or that every requested
requirement is satisfied. Task-specific validation remains essential.

Plans, unexecuted scripts, intermediate artifacts and submitted background jobs
must not be presented as completion of an execution task. An audited negative
result, or an empty strict-novelty selection, can be a finished deliverable. Do
not relax novelty/evidence gates, relabel known relations or fabricate data to
force a positive result.

A `blocked` report identifies the missing requirement, checks/alternatives tried
and relevant tool evidence. Missing local patient data is not a blocker for an
idea-only literature task. Missing indispensable data, inaccessible resources,
required authorization, exhausted budget or an unrecoverable execution failure
can be genuine blockers. Authorization/budget blockers need not first execute an
unauthorized or over-budget tool action.

Four consecutive text-only responses without tool execution, or six identical
failed tool attempts in the recent failure window, stop as **stalled/incomplete**,
not as scientific completion or proof that the requested analysis is impossible.
Unchanged successful job polling does not trip the failed-attempt guard. A
non-retryable provider/runtime failure is **interrupted/incomplete**.

## Cancellation and records

HTTP send and resend carry a unique request ID. The Stop button sends
`/api/chat/cancel` for that ID and aborts the browser request; HTTP disconnect also
signals cancellation. WebSocket AutoResearch uses the same tool-enabled runtime,
with `{"type":"cancel"}` and disconnect handling, rather than text-only streaming.
Cancellation stops subsequent model/tool dispatch and retry backoff, and attempts
to terminate the owned foreground shell process tree. An already in-flight
provider request may still return before it can drain; provider-side cancellation
is not claimed. Unrelated jobs and historical results are untouched.

Each run atomically updates
`.neurodiscovery/autoresearch/<unique-run-id>/run.json` in its workspace. It records
mode, configured provider/model, status, iteration budget/count, tool evidence
metadata, delivery paths and final validation/blocker information. Runtime
receipts are rejected as research deliverables. Tool arguments, provider exception
text, prompts, credentials and raw command outputs are not copied into this
receipt. The existing client worklog retains its bounded tool outputs and now
includes AutoResearch state and evidence IDs. Partial files are preserved after
cancellation/interruption; the receipt is not an automatic restart mechanism or
a full conversation archive. Inspect the saved work before resuming.

The persistent tool protocol requires a tool-capable backend, including the
OpenAI-compatible and native Anthropic adapters. The local text-only adapter
reports this limitation without pretending an experiment ran. Benchmark execution
remains on its existing path; this change does not dispatch or modify scientific
campaigns.

## Offline regression tests

```bash
python -m pytest core/agent/test_autoresearch_execution.py core/test_autoresearch.py core/web/test_autoresearch_execution.py core/web/test_autoresearch_help.py core/web/test_hypothesis_controls.py
```

Tests use synthetic replies and temporary files, plus a harmless local sleep
process for cancellation. They do not call a live model or run a scientific
experiment. Existing novelty-policy tests also verify the three selection modes.
