# cognitive_decoding — `{IM}->Tk`

**Description**: Brain activity → stimulus / mental-state label.

**Example**: Visual cortex BOLD pattern → seen-image category

## Atom composition

- Inputs: imaging_marker
- Output: cognitive_task
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: cognitive_decoding
canonical_signature: "{IM}->Tk"
inputs:
  - atom: <one of ['imaging_marker']>
    kg_anchor: <node id or domain query>
output:
  atom: cognitive_task
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
