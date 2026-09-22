# Current graph inputs and version-pinned experiments

Ordinary graph readers now resolve the current **accepted publication**, not a
date-named directory or the most recently modified JSON file:

1. Use the campaign configured by `NEUROCLAW_KG_CAMPAIGN` or
   `neurooracle/configs/graph_explorer.json`. Check its completed publication,
   acceptance receipt, graph binding and file fingerprint.
2. If no campaign is configured, use the canonical graph recorded in
   `neurooracle/data/full_v2/CURRENT_STATE.json`.
3. A configured but unavailable, changing or unaccepted publication is an error.
   There is no silent fallback to `full_snapshot_v1`, `full_snapshot_v2`, a small
   display export, or an empty graph.

Inspect the resolved input without loading the large graph:

```bash
python -m neurooracle.current_graph
```

Normal Python callers use `load_graph()`; hypothesis CLI and query commands
resolve the same default. An explicit `--graph /path/to/frozen/graph.json` (or an
explicit path to `load_graph`) keeps experiments pinned to their original input.
This change does not republish graph data or relabel historical results.

## Writes and model assets

- `save_graph` requires an explicit output path. Ingestion templates, marker
  ingestion and optimization commands require a separate `--output`.
- Fresh batch-extraction builds use `data/build_artifacts/claim_extraction`, or
  their explicitly supplied output directory. Creating a new build is separate
  from selecting a published graph for research.
- KGE plausibility requires an explicitly supplied checkpoint. Selecting a new
  graph does not make an old graph's KGE checkpoint compatible. This change does
  not train models or certify user-supplied checkpoint compatibility.
- Shell runners accept `NEUROCLAW_GRAPH_PATH` for an explicit frozen input and
  `NEUROCLAW_KGE_CHECKPOINT` for the corresponding model. Otherwise their graph
  input follows the current publication. Set `NEUROCLAW_NOVELTY_CACHE` separately
  if an existing literature-query cache should be reused.

Legacy abstract caches, original extracted claims and marker/hypothesis exports
are **source or experiment assets**, not the published graph selector. Their
historical audit references are deliberately retained until a separately
verified migration or cleanup. Frozen implementations and completed experiment
manifests must not be rewritten to point to a newer graph.

The existing accepted-claim web interface already follows the campaign and its
accepted evidence layer. The legacy packaged display export is a separate UI
artifact, not a substitute for the current research graph. Rebuild desktop
runtime bundles after source changes; do not patch sealed experiment copies.
