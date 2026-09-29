# NeuroDiscovery Desktop

Electron desktop client for NeuroDiscovery, the closed-loop framework for
evidence-grounded neuroimaging autoresearch. NeuroRuntime provides the execution
backend; the graph explorer and hypothesis-generation interface retain the
NeuroOracle name.

## Offline five-scenario demo

`npm run prepare:demo:showcase` then `npm run dev:demo` opens the keyless,
offline Chat / Idea / Data / Experiment / Full demonstration. It streams
scripted replies, simulated tool steps and results, and includes loadable
synthetic data. No Python backend or model service starts. See
[DEMO_NATIVE_PROMPTS_zh.md](DEMO_NATIVE_PROMPTS_zh.md) for the five prompts and
AutoResearch settings. The demo uses the production frontend unchanged; there
is no separate scene picker or demo layout. Windows packages: `dist-demo-native/`.

## Camellia-style Electron client

`cd desktop` then `npm start` launches the Electron window with the new harness
interface by default. It uses the existing local Python backend, native menu,
file/folder dialogs, configuration and secure preload; no browser launch is
required. Configure the local runtime as before. An older backend without
`/api/harness` is not reused; the launcher can select a different available port.

New configurations default to Built-in Python. `npm start -- --development-runtime`
uses `desktop/runtime/python` with the live source checkout, not `.venv`, and
keeps its environment file separate from the repository configuration. The bundled
requirements already include `openai>=1.0,<3`; the development runtime's OpenAI
2.53.0 import and offline client construction were verified on 2026-09-23.

The layout follows Camellia's sidebar, centered 768px conversation/composer,
neutral light/dark palette, blue accents and HARNESS identity. NeuroDiscovery's
graph, AutoResearch scopes, Human Evaluation, ranking and results are explicit
sidebar entries. Selecting a scope does not run an experiment.

`npm run dev:legacy` opens the previous interface for comparison. Browser `/`
remains unchanged. Existing installed executables are not updated by a source
change: stage the current backend and rebuild before distributing an installer.

`npm run test:harness` exercises the renderer in a hidden Electron window with
synthetic local APIs, blocks external network requests, and saves screenshots to
`tmp/harness-electron-preview`. It does not load user credentials, real graph or
study records. This checks the renderer, not packaged-runtime installation.

## macOS build

```bash
cd desktop
npm ci
npm run prepare:runtime:mac
npm run dist:mac
```

The macOS runtime script stages a relocatable conda prefix in `desktop/runtime/python`
and the NeuroRuntime backend in `desktop/runtime/backend`. It uses the Python minor
version from the `neuroclaw` conda environment by default.

The desktop Settings page stores the same LLM fields on Windows and macOS:
provider, model, base URL, API key, and optional API key environment variable.
Those values are written into the bundled or local backend configuration on
next launch.

Pass build-time options after `--`:

```bash
npm run prepare:runtime:mac -- --conda-exe "$HOME/miniconda3/bin/conda" --conda-env neuroclaw
```

Architecture-specific packages are also available:

```bash
npm run dist:mac:arm64
npm run dist:mac:x64
```

## Windows build

### Demo distribution (without Human Evaluation)

```powershell
cd desktop
npm run dist:demo:win
```

Reuses the prepared `runtime/python` interpreter and stages current backend
source separately in `runtime-demo`. Produces a Windows x64 portable executable
and ZIP in `dist-demo`. Staging refuses to overwrite an existing demo runtime.
Human Evaluation menus, routes and study materials are excluded; the normal
distribution retains evaluation. Demo settings and runtime caches use the
separate `NeuroDiscovery-Demo` user-data directory. No personal credentials,
chat history, local datasets or knowledge graph are included. Configure a model
in Settings for live inference; graph/data demonstrations require separately
supplied data. This Windows build does not produce a macOS application.

### Standard distribution

```powershell
cd desktop
npm ci
npm run prepare:runtime:win
npm run dist:win
```
