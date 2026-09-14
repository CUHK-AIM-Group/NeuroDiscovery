# Adapter and artifact contract

## Model boundary

`--adapter` is a reviewed local Python file defining `build_model()` and optionally
`forward_scores(model, inputs)`. The first returns the exact architecture used
for the supplied checkpoint. The second accepts a dictionary of preprocessed
floating tensors and returns `[batch,outputs]` scores (or `[batch]` for scalar
regression with target 0). Return logits when explaining logits; do not silently
softmax or change the prediction function. It must preserve gradients for IG/CAM,
be deterministic in evaluation mode, and use only the intended patient input.

The CLI imports this adapter, loads a state dict (or a dict with `state_dict`),
and calls `load_state_dict(strict=True)`. The caller explicitly trusts adapter
execution. Loading arbitrary pickled whole-model objects is unsupported. Adapter
code should not perform training, network requests or file writes at import time.
The default forward is `model(inputs["inputs"])`.

Minimal tabular example (replace dimensions/architecture to match the checkpoint):

```python
from torch import nn

def build_model():
    return nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))
```

Tuple-output model example:

```python
def forward_scores(model, inputs):
    logits, assignments = model(inputs["fc"])
    return logits
```

Graph models need an architecture-specific adapter that binds the actual graph
topology and returns differentiable scores; this interface does not fabricate a
generic graph/edge explainer. Discrete token IDs and integer graph indices should
be fixed adapter context, not integrated as floating clinical features. Document
and fingerprint any additional bound model resources in the study's provenance.

## Data and identity

`--inputs` is an NPZ created without object arrays. Keys are Python identifiers;
each value is finite float `[N,...]` with the same subject axis. Execution is one
subject at a time and input tensors are converted to float32. Models whose prediction depends on other batch members need a
separately validated adapter; do not treat this as full-batch reproduction.

```json
{
  "subject_ids": ["subject-001", "subject-002"],
  "preprocessing": "training-fold imputer/scaler; held-out inputs transformed without refit",
  "feature_names": {"inputs": ["roi_volume", "cbf_mean", "age_at_scan"]}
}
```

Supply only authorized, prespecified features. Metadata does not perform or
validate training preprocessing: verify the adapter's predictions agree with the
original model before interpreting its explanations. Baseline NPZ keys/shapes
must exactly match inputs; duplicate a prespecified reference for all subjects
explicitly when appropriate. `--zero-baseline` is an explicit alternative.

### Spatial attribution

Spatial input layout is **`[N,C,X,Y,Z]`**, with the last three indices matching the
reference NIfTI's voxel axes exactly. A reference describes the **processed model
grid**, not an arbitrary native scan. Supply `spatial_references` in metadata:

```json
{
  "subject_ids": ["subject-001", "subject-002"],
  "preprocessing": "registered model-grid volumes; frozen training normalization",
  "spatial_references": {"inputs": ["processed_001.nii.gz", "processed_002.nii.gz"]}
}
```

Paths are relative to the metadata file (absolute paths also work). A single
string explicitly declares a shared template. Shape, finite nonsingular affine
and mm units are checked. IG preserves individual channels (fourth NIfTI axis,
not time); Grad-CAM produces one resampled spatial channel. Affine retention does
not undo cropping, normalization or registration: invert transforms separately
only if a native-space output was requested and transformations are available.

### Attention

The selected module must be called once and either return `(output, weights)`
or expose `get_attention_weights()` for that forward pass. Export preserves
`[N,Q,K]` (already averaged) or `[N,H,Q,K]`; only the preview averages heads.
Weights must be finite, nonnegative and row-normalized. No hidden fallback to
gradients or random/synthetic weights is allowed.

Metadata needs `attention.query_labels` and `attention.key_labels`, each a list
matching its axis, including explicitly labeled padding if present. Example:

```json
{"attention": {"query_labels": ["ROI_A", "ROI_B"], "key_labels": ["ROI_A", "ROI_B"]}}
```

Merge these fields with subject/preprocessing metadata. Variable per-subject
token identities require separate compatible batches or an explicit adapter
mapping; the reference exporter uses one common label order. `target` records the
prediction score, but does **not** make attention class-conditioned.

## Numerical interpretation

IG integrates input gradients with a trapezoidal rule and tests signed attribution
sums against the selected input-minus-baseline score difference. Grad-CAM uses
spatially averaged target gradients, a weighted activation sum, ReLU and spatial
upsampling. Neither guarantees localization accuracy. Visual scale changes never
replace the raw array. Completeness is a numerical check, not model faithfulness
or a causal/clinical validity test.

- [Integrated Gradients and completeness (Captum)](https://captum.ai/docs/extension/integrated_gradients).
- [Layer Grad-CAM definition and output conventions (Captum)](https://captum.ai/api/layer.html).
- [Original Integrated Gradients paper](https://arxiv.org/abs/1703.01365).
