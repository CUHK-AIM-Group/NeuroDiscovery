# drug_response_prediction — `{D,Rx,IM}->O`

**Description**: Patient profile + drug → expected response magnitude.

**Example**: Baseline DaTSCAN + levodopa → ΔMDS-UPDRS at 12 months

## Atom composition

- Inputs: disease, drug, imaging_marker
- Output: outcome
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: drug_response_prediction
canonical_signature: "{D,Rx,IM}->O"
inputs:
  - atom: <one of ['disease', 'drug', 'imaging_marker']>
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
