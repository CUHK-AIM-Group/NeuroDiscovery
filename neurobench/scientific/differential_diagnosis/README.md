# differential_diagnosis — `{IM}->D[contrastive]`

**Description**: Distinguish disease A from disease B using imaging features.

**Example**: rs-fMRI features distinguish bipolar from major depression

## Atom composition

- Inputs: imaging_marker
- Output: disease
- Modifier: `contrastive`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: differential_diagnosis
canonical_signature: "{IM}->D[contrastive]"
inputs:
  - atom: <one of ['imaging_marker']>
    kg_anchor: <node id or domain query>
output:
  atom: disease
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
