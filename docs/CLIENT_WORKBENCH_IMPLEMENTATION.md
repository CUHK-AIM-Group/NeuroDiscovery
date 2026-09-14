# Client workbench implementation

Scope: bring the desktop/browser workbench's interaction and state management up
to the local DSH reference without replacing NeuroDiscovery's research workflows.
No experiment, provider switch, credential migration, release, or Git push is part
of this work.

## Acceptance checklist

- [x] Request-bound, resumable execution events; real provider-visible reasoning
  and text, tool lifecycle, cancellation, and terminal outcomes. Never render
  encrypted reasoning, signatures, or invented thinking.
- [x] Persistent per-upstream-call usage ledger, including retries, auxiliary
  calls, and delegated calls. Preserve unreported usage and normalize cache
  accounting without double counting. Separate local usage from account balances.
- [x] Per-turn/session usage and a filterable settings dashboard with export.
- [x] Durable workspace/session state with safe legacy migration; canonical
  workspace paths, pin/archive/move, and remove-workspace without deleting history.
- [x] Unified accessible dialogs and buttons, keyboard/focus behavior, inline
  validation, and guards against moving/deleting active work.
- [x] Offline backend and frontend regressions, plus isolated UI validation with
  synthetic responses and no API keys or scientific data.

## Implemented behavior

- `core/web/workbench.py` stores versioned history, recovery copies, public run
  snapshots and one ledger row per actual model-call attempt in SQLite. Default:
  the user's `.neurodiscovery/workbench.sqlite3`; tests/isolated runtimes can set
  `NEURODISCOVERY_WORKBENCH_DB`. The existing explicit application reset includes
  this data, and both reset confirmations now say so. Removing a workspace or
  deleting a conversation does **not** delete disk files or usage records.
- The current client starts a turn once, then reads request/chat-bound sequenced
  events from `/api/chat/runs/{request_id}`. Reloading resumes observation, not
  model execution. A runtime restart is shown as interrupted, with partial output
  and any reported usage retained. A browser disconnection does not cancel work;
  the Stop control requests cancellation and waits for the real terminal state.
- Native AgentSession tools, AutoResearch stopping/evidence rules and subagents
  remain in charge of execution. Provider-visible reasoning summaries/text and
  tool events are presented in expandable cards. Opaque/encrypted reasoning and
  Anthropic signatures are never rendered. Stream transport failures after any
  upstream event are not automatically replayed. SDK-hidden retries are disabled
  only on the observed client copy; framework retry attempts are individually
  counted. Title generation is counted separately and cannot run tools.
- Settings → Usage filters by period, provider, model and workspace, and exports
  all matching rows as CSV (the display shows the newest 200). Conversation and
  turn counters are available beside the composer and responses. Cache details
  are not added twice; reasoning tokens are an output subset. Missing fields
  remain unknown, not zero. Costs/account balances are explicitly unavailable
  when not reported, rather than estimated from invented prices.
- Workspace paths are validated against real existing directories. Pin, archive,
  move and safe workspace removal retain conversation identity, drafts and
  checkpoint scope. Moving to another workspace changes future execution's cwd;
  detached/independent conversations keep their previous cwd. UI actions that
  would move/archive/delete active conversations are disabled and rechecked.
- Server history is authoritative after the first legacy-browser migration.
  Compare-and-swap revisions prevent stale-window overwrites. On a conflict,
  the independent conflicting snapshot is saved for download and further saves
  pause until the user resolves it. A browser backup is still retained where
  browser storage is available; it is not the sole history store.
- Native dialogs provide inline errors, busy controls, Escape dismissal and
  focus restoration. Sidebar redraws preserve menu focus. Streaming and final
  responses respect an existing scroll position when the user is reading older
  messages. An interrupted regeneration retains its real partial execution.

## Verification — 2026-09-15

```console
python -m pytest core/web core/agent/test_autoresearch_execution.py core/llm/tests core/subagent desktop/tests -q --tb=short
node --test core/web/static/tests/client-workbench.test.cjs
node --check core/web/static/client-workbench.js
```

462 pytest cases passed, including the Node behavior-suite wrapper; the Node
suite contains 11 passing subtests (not 11 additional pytest cases). The only
warning is the existing Starlette/AnyIO BlockingPortal deprecation. Tracked-file
whitespace checks and JavaScript syntax checks passed.

An isolated local preview with synthetic responses verified live public summary,
text and tool cards, reload/reconnection without duplicated calls, explicit Stop,
per-turn totals and the settings dashboard. The two fixture calls remained two
ledger rows across reloads. One turn was stopped during its tool phase **after**
its model call completed; the UI correctly distinguishes a cancelled turn/tool
from a completed API call. Moving the fixture chat preserved both turns while
changing its workspace. Menu opening, dialog field focus and Escape return-focus
were checked against the actual browser accessibility tree; the usage layout was
visually checked at desktop size. No provider credentials, patient datasets or
running research campaigns were used.

## Explicit boundaries

- Usage covers requests through this workbench, not unrelated scripts or the
  whole provider account. Existing usage before this feature is not reconstructed.
- Public reasoning appears only when the provider supplies it. The legacy
  `local` Ollama path is still non-streaming: its content and actual
  `prompt_eval_count` / `eval_count` are observed when its response arrives.
- Live in-memory events are bounded; durable snapshots retain at most one million
  characters per text/reasoning block. This is a UI replay bound, not a change to
  the native scientific protocol or model output cap. Server restart does not
  resume in-flight execution automatically.
- The original websocket compatibility route remains legacy; the current UI
  uses the instrumented HTTP start/poll path. No new multi-connection credential
  manager, account-price estimates or right-hand terminal IDE was added.
- These are source changes. No installer was rebuilt and no commit/push was made.

## Working-tree boundary

`core/web/server.py` already contained a discovery-study route registration before
this task. Preserve it and all other pre-existing changes. New client modules must
be opt-in from the web workbench; frozen scientific campaigns remain untouched.

## Reference

The local DSH checkout's `session-workspaces.js`, `claude-sidebar.js`, `claude.js`,
and `api-usage.js` informed the behavior. Implementations remain native to the
NeuroDiscovery Python/HTML client. In particular, do not copy automatic key
failover, credential fingerprints, or misleading cache/context labels.
