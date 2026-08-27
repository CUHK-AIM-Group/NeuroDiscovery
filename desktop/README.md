# NeuroDiscovery Desktop

Electron desktop client for NeuroDiscovery. The NeuroOracle knowledge graph is
available as a dedicated graph and hypothesis-generation feature.

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

```powershell
cd desktop
npm ci
npm run prepare:runtime:win
npm run dist:win
```
