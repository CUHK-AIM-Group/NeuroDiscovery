---
name: model-interpretability
description: "Export trained-model attention weights, target-specific Integrated Gradients or Grad-CAM, patient-level feature attributions and spatial NIfTI maps with numerical QA and required-output validation. Use for post-training explanations or attention/attribution maps, not causal effect estimation."
license: MIT
layer: base
skill_type: tool
dependencies:
  - claw-shell
  - nibabel-skill
---
# Model Interpretability Skill

Produce requested explanatory artifacts in addition to predictions and metrics.
Read [the adapter and output contract](references/adapter-contract.md) before execution.
This supports trained PyTorch models through a small explicit adapter, including
tuple-output and multimodal models. It does not make every architecture
automatically explainable or turn explanations into clinical evidence.

## Select what the user asked for

- **Input/feature attribution:** use Integrated Gradients (`ig`). Select a named
  class logit or regression output and an explicit reference baseline after the
  exact training-fitted preprocessing. Keep signed attributions and per-patient
  completeness residuals. Zero is not a neutral baseline for every modality.
- **CNN localization:** use `gradcam` with an explicitly selected 2D/3D activation
  layer and spatial input. It exports original ReLU Grad-CAM, not voxel-level
  causal effects. A zero map can be a genuine method result; inspect diagnostics.
- **Attention:** use `attention` only when the model actually produces normalized
  attention weights. Preserve layer, heads/averaging, token/ROI identity and
  padding labels. BNT's `get_attention_weights()` is supported on the selected
  executed module; adapters must enable real weight return for other models.
  Attention is not generally target-specific or equivalent to attribution.

Ask for missing model/checkpoint, target or reference-space information when it
changes interpretation. Do not retrain a frozen model, rerun a clinical study,
upload patient artifacts, or invent a missing explainer to satisfy the request.
Dependencies: `torch numpy pandas nibabel matplotlib` (`clinical-outputs` extra).

```bash
python skills/model-interpretability/scripts/explain.py --adapter adapter.py --trust-adapter --checkpoint checkpoint.pt --inputs heldout.npz --metadata inputs.json --method ig --target 1 --output-kind logit --baseline reference.npz --output-dir explanations_ig

python skills/model-interpretability/scripts/explain.py --adapter adapter.py --trust-adapter --checkpoint checkpoint.pt --inputs heldout.npz --metadata inputs.json --method gradcam --target 1 --output-kind logit --layer encoder.4 --cam-input inputs --output-dir explanations_cam

python skills/model-interpretability/scripts/explain.py --adapter adapter.py --trust-adapter --checkpoint checkpoint.pt --inputs heldout.npz --metadata inputs.json --method attention --target 1 --output-kind logit --layer attention_layer --output-dir explanations_attention
```

Layer names above are illustrative: inspect `model.named_modules()` and select
the actual relevant layer. `--trust-adapter` executes that specific local Python
file; review it before opting in. Checkpoints use `weights_only=True` and strict
state-dict matching; do not weaken loading to accept an untrusted pickle.

## Completion means delivered outputs

The script writes per-subject arrays, `diagnostics.csv`, `qc.json`, a preview figure
and `run_manifest.json`. With tabular feature names it also writes signed feature
tables; with explicit model-grid reference NIfTIs it writes per-subject spatial
maps. Original arrays remain unnormalized; the figure is a labeled first-subject
preview, not a cohort result. The manifest binds IDs, preprocessing, inputs,
checkpoint, adapter, parameters and output hashes. File names use subject indices
with an explicit ID mapping, avoiding patient identifiers in image filenames.

Before training/explaining, turn the user's requested deliverables into a short
checklist. Afterward, require **all** of them, even when a producer omitted an
optional metadata-driven output. For example:

```bash
python -m models.common.research_outputs explanations_ig/run_manifest.json --require attributions attribution_map diagnostics explanation_figure qc
python -m models.common.research_outputs explanations_attention/run_manifest.json --require attention diagnostics explanation_figure
```

An attention run cannot satisfy an attribution request. A missing map/reference,
unstated target or failed IG completeness check means the request is incomplete.
The script preserves failed-QA diagnostics without a completion manifest. Review
the baseline and numerical integration; if justified, retry with more `--steps`
in a new directory. Do not simply loosen tolerances to make a result pass.

For subtype outputs use `subject-subtyping`; for CTP-derived maps use `ctp-skill`.
Check each bundle against its required roles. For biomarker claims, plan explicit
perturbation/randomization, stability and external checks with the user; successful
export is not scientific validation and does not authorize those experiments.

Created At: 2026-09-14 14:43:06 HKT
Last Updated At: 2026-09-14 15:05:31.591 HKT
Author: chengwang96
