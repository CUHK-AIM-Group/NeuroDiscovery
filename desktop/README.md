# NeuroDiscovery Desktop

Electron desktop client for NeuroDiscovery, the closed-loop framework for
evidence-grounded neuroimaging autoresearch. NeuroRuntime provides the execution
backend; the graph explorer and hypothesis-generation interface retain the
NeuroOracle name.

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
