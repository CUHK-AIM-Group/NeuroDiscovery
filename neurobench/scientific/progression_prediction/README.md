# progression_prediction — `{IM}->D[longitudinal]`

**Description**: Baseline imaging → future disease conversion.

**Example**: MCI baseline imaging → AD conversion within 24 months

## Atom composition

- Inputs: imaging_marker
- Output: disease
- Modifier: `longitudinal`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: progression_prediction
canonical_signature: "{IM}->D[longitudinal]"
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
