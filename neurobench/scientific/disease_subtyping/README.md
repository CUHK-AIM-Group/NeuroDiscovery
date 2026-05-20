# disease_subtyping — `{IM,Idv}->D[subtype]`

**Description**: Multivariate features → disease subtype label.

**Example**: DMN connectivity + age → AD-typical vs AD-hippocampal-sparing

## Atom composition

- Inputs: imaging_marker, individual_data
- Output: disease
- Modifier: `subtype`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: disease_subtyping
canonical_signature: "{IM,Idv}->D[subtype]"
inputs:
  - atom: <one of ['imaging_marker', 'individual_data']>
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
