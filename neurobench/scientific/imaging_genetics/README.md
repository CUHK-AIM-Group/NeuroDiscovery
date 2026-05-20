# imaging_genetics — `{G}->IM`

**Description**: Genotype / molecular target → brain phenotype.

**Example**: APOE-ε4 → hippocampal atrophy

## Atom composition

- Inputs: gene_target
- Output: imaging_marker
- Modifier: `(none)`

## Adding a benchmark instance

Create `<this_dir>/<instance_id>/task.md` with the standard front-matter:

```yaml
---
canonical_task: imaging_genetics
canonical_signature: "{G}->IM"
inputs:
  - atom: <one of ['gene_target']>
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
