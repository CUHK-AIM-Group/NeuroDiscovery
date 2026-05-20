# adverse_event_prediction — `{Rx}->O`

**Description**: Drug → adverse-event likelihood (off-target effects).

**Example**: Levodopa long-term → impulse control disorder

## Atom composition

- Inputs: drug
- Output: outcome
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: adverse_event_prediction
canonical_signature: "{Rx}->O"
inputs:
  - atom: <one of ['drug']>
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
