# drug_repurposing — `{Rx}->D`

**Description**: Existing drug → novel indication via mechanism path.

**Example**: Memantine → traumatic brain injury (via NMDA modulation)

## Atom composition

- Inputs: drug
- Output: disease
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: drug_repurposing
canonical_signature: "{Rx}->D"
inputs:
  - atom: <one of ['drug']>
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
