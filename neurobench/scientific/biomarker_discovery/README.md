# biomarker_discovery — `{IM}->D`

**Description**: Identify imaging markers that distinguish or predict a disease.

**Example**: Hippocampal volume → Alzheimer's Disease

## Atom composition

- Inputs: imaging_marker
- Output: disease
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: biomarker_discovery
canonical_signature: "{IM}->D"
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
