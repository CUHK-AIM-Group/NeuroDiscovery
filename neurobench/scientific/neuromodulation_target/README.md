# neuromodulation_target — `{D,Tk}->IM`

**Description**: Disease + symptom → optimal stimulation site.

**Example**: Treatment-resistant depression + anhedonia → DMN node target

## Atom composition

- Inputs: cognitive_task, disease
- Output: imaging_marker
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: neuromodulation_target
canonical_signature: "{D,Tk}->IM"
inputs:
  - atom: <one of ['cognitive_task', 'disease']>
    kg_anchor: <node id or domain query>
output:
  atom: imaging_marker
  kg_anchor: <node id or domain query>
dataset: <e.g. ADNI, HCP, PPMI>
modality: <e.g. T1w, rs-fMRI, EEG>
---
```

Followed by free-form natural-language instructions for the agent.

## Status

Currently empty. Populate as scientific benchmark instances roll out.
