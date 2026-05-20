# personalised_treatment — `{D,IM,Idv}->Rx`

**Description**: Patient profile → recommended drug.

**Example**: MCI + APOE-ε4 + low hippocampal volume → cholinesterase inhibitor

## Atom composition

- Inputs: disease, imaging_marker, individual_data
- Output: drug
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: personalised_treatment
canonical_signature: "{D,IM,Idv}->Rx"
inputs:
  - atom: <one of ['disease', 'imaging_marker', 'individual_data']>
    kg_anchor: <node id or domain query>
output:
  atom: drug
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
