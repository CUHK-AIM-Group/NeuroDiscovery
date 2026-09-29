# NeuroDiscovery Harness — incremental refactor

Latest correction: see `DESKTOP_ENGINE_AUDIT_20260923.md` for actual engineering
parity and Markdown loading. Subsequent UI requests removed all visible HARNESS
badges and sidebar research actions except chat/graph. The shared palette now
also covers graph/evaluation documents. Earlier visual descriptions below are
historical, not claims that Camellia's conversation engine was ported.

## Current desktop direction — 2026-09-23

The user clarified that the target is an **Electron desktop client with
Camellia's design**, not a browser-only preview. The existing Electron host now
loads `/harness` by default (`cd desktop`, `npm start`). `--legacy-ui` remains a
comparison/rollback route; browser `/` is unchanged. This supersedes the initial
opt-in/default-interface statements in the first-slice record below.

The desktop interface now uses Camellia-style full-height sidebar/brand, HARNESS
badge, centered 768px composer, 22px input card, neutral light/dark backgrounds,
blue accents and a compact header. Domain actions live in the sidebar rather
than an extra top toolbar. Labels, selected views and AutoResearch scope badges
follow the existing language/runtime state. Existing workspace/session behavior
and native Electron menus, file dialogs and preload remain in use.

Reference layout: Camellia `src/renderer/chat/claude.html`, `claude.css` and
`src/renderer/shared/ui-theme.css`. This ports presentation patterns and selected
style values; it does not claim to have integrated Camellia's five engines or
shared-conversation implementation. Graph/evaluation inner documents still use
their existing presentation. The new client is not a separately implemented
agent runtime and does not start research on navigation.

Verification: 120 Python and 20 Node regressions passed. A hidden real Electron
window additionally checked actual renderer layout, scope selection, graph and
evaluation navigation, sidebar collapse, narrow-screen drawer, and no research
dispatch against synthetic APIs. Light/dark/narrow screenshots and a receipt are
in `tmp/harness-electron-preview`; light/dark and narrow screenshots were visually
inspected. No real provider, graph, study dataset or production service was used.
The actual desktop host/backend launch and a rebuilt installer still need their
own integration acceptance; no installer has been rebuilt or installed.

## Initial migration record

## Direction

Package NeuroDiscovery as a research harness: a reusable workbench shell with
NeuroRuntime as the execution adapter and neuroscience capabilities registered
separately. CLI and GUI consume the same capability manifest. Do not replace the
scientific runtime with a second agent loop or equate UI reuse with validation.

This is the first migration slice, **not a completed Camellia port**. `/harness`
is an opt-in workbench using the existing client, history and service APIs. `/`
and the normal desktop launch remain available. No history import or reset is
performed. Both interfaces share the existing history conflict protection.

## Try it

From the repository, using the existing Python environment:

```console
python -m core.cli capabilities
python -m core.cli gui --port 7083
python -m core.cli graph --port 7083 --open
python -m core.cli evaluate --port 7083 --open
python -m core.cli autoresearch data --language zh
python -m core.cli chat --workspace .
```

After installing the project, `neurodiscovery` is the equivalent console entry
point. The GUI command serves the backend; open `http://127.0.0.1:7083/harness`.
Graph/evaluate print links to that backend and only open a browser with `--open`.
The default port deliberately differs from existing production port 7080.

Desktop development: `cd desktop` then `npm run dev:harness`. This requires the
updated backend; compatibility checks reject older backends without `/api/harness`
and the existing launcher selects another available port. Do not stop
a scientific/production service to test the UI. Use a separate configured port.
No installer or packaged runtime has been rebuilt by this change.

The AutoResearch button opens the existing scope selector. It does not submit a
message or start a run. `autoresearch <scope>` prints the existing offline scope
checklist; `chat --autoresearch <scope>` selects that mode for a user-driven
interactive session. Model execution begins through the existing runtime after
the user submits a research message, subject to its existing limits and gates.

## Camellia reuse audit

Reference: local `Camellia`, HEAD `7054d1ce18763c486332b1fc29122d1e7c62db61`.
The inspected working-tree theme SHA-256 was
`7ef26c567c9d2c6fb870b20e53ec4a5b694070dc74be9a687267d3f7fd933601`.
Reference files were read only; no Camellia files or application state changed.

| Area | Reference | Current disposition |
| --- | --- | --- |
| Visual tokens | `src/renderer/shared/ui-theme.css` | Selected palette/font variables reused in `harness.css`; scoped to the opt-in surface, not the legacy graph/evaluation documents |
| Shared workbench | `src/engines/shared-conversations.js` | Architecture reference only; tightly coupled to five engines, history, Goal and Tasks; not copied or advertised as integrated |
| Desktop host | `src/main/main.js`, `desktop-views.js` | Keep NeuroDiscovery's existing secure preload/backend lifecycle; opt-in route added |
| Durable state | `src/shared/json-store.js` | Retain NeuroDiscovery's existing SQLite/CAS history instead of creating a second incompatible store |
| Research | NeuroDiscovery's existing runtime/APIs | Graph, AutoResearch, Human Evaluation, ranking and results remain domain adapters |

No live filesystem dependency on the neighboring Camellia checkout is introduced.
Camellia accounts, provider settings, credentials, conversations and engine
installation directories are not imported. Goal/Tasks/conversation automation
are not enabled by the refactor. Embedded graph and evaluation pages retain
their current presentation and authentication; complete theme unification is
not claimed.

## Remaining migration

1. Extract Camellia's renderer/session host behind explicit injected interfaces
   (conversation CRUD, workspace, event subscription, cancel, usage, permission
   requests, settings). Avoid copying its monolithic main process wholesale.
2. Implement a NeuroRuntime adapter to the existing request-ID/sequence event
   stream and SQLite history; verify reload/cancel/partial-output behavior with
   synthetic sessions before switching defaults. Current CLI uses the existing
   REPL, not yet the GUI's durable conversation store.
3. Move NeuroDiscovery capability panels to that shared host. Add bilingual
   labels, active-view state, keyboard tests and cross-panel theme consistency.
   Current capability toolbar labels are English; existing panels retain their
   language settings. Keep evaluation submissions separate from research chat.
4. Independently validate the replacement with offline fixtures, then package
   and test fresh installs/upgrades. Only then change the default GUI/desktop
   entry. Do not claim production readiness from structural tests alone.

## Boundaries and checks

This refactor does not extract papers, rebuild/update the KG, consume heldout
assignments, run experiments, invoke models, migrate E:, start background tasks,
or write scientific results. Navigating to graph does not auto-download it.
Existing study authentication and demo-build exclusions remain authoritative.

```console
python -m pytest core/web/test_harness.py core/web/test_workspace_theme.py core/web/test_oracle_workspace.py core/web/test_study_workspace.py desktop/tests/test_window_chrome.py -q
node --test core/web/static/tests/harness.test.cjs
```

### Verification — 2026-09-23

120 Python tests passed across the harness, existing workspace/graph/evaluation
presentation, AutoResearch scope/execution and desktop chrome suites. 19 Node
tests passed across the harness and existing client-workbench behavior suites.
JavaScript syntax and tracked-change whitespace checks passed. Only existing
Starlette/AnyIO deprecation warnings were reported. Tests used offline fixtures;
no full Electron/browser visual acceptance, live research run or installer build
was performed. The default interface has not been switched.
