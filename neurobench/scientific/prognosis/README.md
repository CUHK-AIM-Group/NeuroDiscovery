# prognosis — `{D,IM}->O[longitudinal]`

**Description**: Acute / baseline state + disease → outcome at follow-up.

**Example**: Stroke acute lesion → 6-month NIHSS

## Atom composition

- Inputs: disease, imaging_marker
- Output: outcome
- Modifier: `longitudinal`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: prognosis
canonical_signature: "{D,IM}->O[longitudinal]"
inputs:
  - atom: <one of ['disease', 'imaging_marker']>
    kg_anchor: <node id or domain query>
output:
  atom: outcome
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
