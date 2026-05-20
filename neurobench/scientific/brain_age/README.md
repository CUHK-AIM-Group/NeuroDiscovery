# brain_age — `{IM}->Idv`

**Description**: Imaging features → biological age estimate.

**Example**: Cortical thickness + GM volume → predicted age

## Atom composition

- Inputs: imaging_marker
- Output: individual_data
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: brain_age
canonical_signature: "{IM}->Idv"
inputs:
  - atom: <one of ['imaging_marker']>
    kg_anchor: <node id or domain query>
output:
  atom: individual_data
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
