"""Target-specific PyTorch IG/Grad-CAM and actual attention export.

Adapters are explicitly trusted local Python; checkpoints load as weights only.
No arbitrary pickle loading and no invented attention for models without it.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.common.research_outputs import OutputBundle, read_json, subject_ids, write_json


@contextmanager
def evaluation(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, state in states:
            module.training = state


def selected_scores(output, n, target, output_kind):
    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise ValueError("Adapter must return finite tensor scores, not a tuple/dictionary")
    if output.ndim == 1:
        if output_kind != "regression" or target != 0 or len(output) != n:
            raise ValueError("1D scores require regression and target=0")
        scores = output
    elif output.ndim == 2 and output.shape[0] == n and 0 <= target < output.shape[1]:
        scores = output[:, target]
    else:
        raise ValueError("Expected [batch,outputs] and a valid target index")
    if output_kind == "probability" and ((output < 0).any() or (output > 1).any()):
        raise ValueError("Declared probabilities are outside [0,1]")
    return scores


def integrated_gradients(model, forward, inputs, baselines, target, output_kind, steps=128):
    """Trapezoidal IG. Process one patient per call (no batch-coupling assumption)."""
    if steps < 2 or set(inputs) != set(baselines):
        raise ValueError("IG needs >=2 integration steps and matching baselines")
    for key, value in inputs.items():
        if value.shape != baselines[key].shape or not value.is_floating_point():
            raise ValueError("IG baselines must exactly match floating input shapes")
        if not torch.isfinite(value).all() or not torch.isfinite(baselines[key]).all():
            raise ValueError("Nonfinite IG input/baseline")
        if value.shape[0] != 1:
            raise ValueError("Call IG one subject at a time to avoid batch-dependent explanations")
    sums = {key: torch.zeros_like(value) for key, value in inputs.items()}
    with evaluation(model), torch.enable_grad():
        for step in range(steps + 1):
            alpha = step / steps
            point = {key: (baselines[key] + alpha * (value - baselines[key])).detach().requires_grad_(True)
                     for key, value in inputs.items()}
            score = selected_scores(forward(model, point), 1, target, output_kind)
            gradients = (torch.autograd.grad(score.sum(), tuple(point.values()), allow_unused=True)
                         if score.requires_grad else [None] * len(point))
            weight = (0.5 if step in (0, steps) else 1.0) / steps
            for key, gradient in zip(point, gradients):
                if gradient is not None:
                    sums[key] += weight * gradient.detach()
        attrs = {key: (value - baselines[key]) * sums[key] for key, value in inputs.items()}
        with torch.no_grad():
            score = selected_scores(forward(model, inputs), 1, target, output_kind)
            baseline_score = selected_scores(forward(model, baselines), 1, target, output_kind)
        total = sum(value.reshape(1, -1).sum(dim=1) for value in attrs.values())
        delta = total - (score - baseline_score)
    return attrs, score.detach(), baseline_score.detach(), delta.detach()


def grad_cam(model, forward, inputs, target, output_kind, layer, spatial_input):
    """Original ReLU Grad-CAM from a selected 2D/3D activation tensor."""
    if spatial_input not in inputs or inputs[spatial_input].ndim not in (4, 5):
        raise ValueError("Grad-CAM needs --cam-input with shape [1,C,H,W] or [1,C,X,Y,Z]")
    activations = []
    hook = model.get_submodule(layer).register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        with evaluation(model), torch.enable_grad():
            point = {key: value.detach().requires_grad_(True) for key, value in inputs.items()}
            score = selected_scores(forward(model, point), 1, target, output_kind)
            if len(activations) != 1 or not isinstance(activations[0], torch.Tensor):
                raise ValueError("Select a layer invoked exactly once with tensor activations")
            activation = activations[0]
            if activation.ndim not in (4, 5) or activation.shape[0] != 1:
                raise ValueError("Grad-CAM layer must have [1,channels,spatial...] activations")
            gradient = torch.autograd.grad(score.sum(), activation)[0]
            weights = gradient.mean(dim=tuple(range(2, activation.ndim)), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
            if cam.ndim != inputs[spatial_input].ndim:
                raise ValueError("CAM and selected spatial input dimensions differ")
            cam = F.interpolate(cam, size=inputs[spatial_input].shape[2:],
                                mode="bilinear" if cam.ndim == 4 else "trilinear", align_corners=False)
            return cam.detach(), score.detach()
    finally:
        hook.remove()


def attention_weights(model, forward, inputs, layer, target, output_kind):
    """Capture actual returned weights or the selected module's explicit getter."""
    module = model.get_submodule(layer)
    captured = []
    hook = module.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        with evaluation(model), torch.no_grad():
            scores = selected_scores(forward(model, inputs), 1, target, output_kind)
            if len(captured) != 1:
                raise ValueError("Attention layer must execute exactly once for this subject")
            # BNT returns DEC assignments as tuple[1]; prefer its explicit
            # attention getter so assignments cannot masquerade as attention.
            if callable(getattr(module, "get_attention_weights", None)):
                weights = module.get_attention_weights()
            elif isinstance(captured[0], tuple) and len(captured[0]) > 1:
                weights = captured[0][1]
            else:
                weights = None
            if not isinstance(weights, torch.Tensor) or weights.ndim not in (3, 4) or weights.shape[0] != 1:
                raise ValueError("No [1,Q,K] or [1,heads,Q,K] attention; adapter must request real weights")
            if (not torch.isfinite(weights).all() or (weights < 0).any()
                    or not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4, rtol=1e-4)):
                raise ValueError("Attention must be finite normalized nonnegative post-softmax weights")
            return weights.detach().clone(), scores.detach()
    finally:
        hook.remove()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not arrays:
        raise ValueError("Input NPZ is empty")
    for key, data in arrays.items():
        if not key.isidentifier() or data.ndim < 2 or not np.issubdtype(data.dtype, np.floating) or not np.isfinite(data).all():
            raise ValueError("NPZ keys must be identifiers; values finite float arrays [subject,...]")
    return arrays


def spatial_references(meta, metadata_path, arrays):
    import nibabel as nib
    references, paths = {}, []
    for key, names in meta.get("spatial_references", {}).items():
        if key not in arrays or arrays[key].ndim != 5:
            raise ValueError("Spatial references require [N,C,X,Y,Z] inputs")
        n = arrays[key].shape[0]
        if isinstance(names, str):
            names = [names] * n
        if not isinstance(names, list) or len(names) != n:
            raise ValueError("Provide one reference image per subject, or an explicitly shared template")
        references[key] = []
        for name in names:
            path = (metadata_path.parent / name).resolve()
            img = nib.load(path)
            if (img.shape != arrays[key].shape[2:] or not np.isfinite(img.affine).all()
                    or abs(np.linalg.det(img.affine[:3, :3])) < 1e-10
                    or img.header.get_xyzt_units()[0] != "mm"):
                raise ValueError("Spatial reference must match model voxel grid, with valid mm affine")
            references[key].append(img)
            paths.append(path)
    return references, list(dict.fromkeys(paths))


def export_array(bundle, key, array, role, ids, references=None):
    name = f"{key}.npy"
    np.save(bundle.root / name, array, allow_pickle=False)
    bundle.add(name, role, "npy", shape=list(array.shape), subject_ids=ids)
    if references is not None:
        import nibabel as nib
        for i, reference in enumerate(references):
            # Preserve channels as the fourth NIfTI dimension; never call them time.
            data = np.moveaxis(array[i], 0, -1)
            if data.shape[-1] == 1:
                data = data[..., 0]
            if data.shape[:3] != reference.shape:
                raise ValueError("Explanation and reference grids differ")
            header = reference.header.copy()
            header.set_data_dtype(np.float32)
            header.set_intent("none")
            header.set_xyzt_units("mm", "unknown")
            header.set_slope_inter(1.0, 0.0)
            img = nib.Nifti1Image(data.astype(np.float32), reference.affine, header)
            filename = f"{key}_subject_{i:04d}.nii.gz"
            nib.save(img, bundle.root / filename)
            bundle.add(filename, "attribution_map", "nifti", shape=list(data.shape),
                       affine=reference.affine.tolist(), subject_id=ids[i],
                       channel_axis="fourth dimension when multichannel; not time")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--trust-adapter", action="store_true", help="Allow executing the specified local adapter Python")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ of preprocessed model inputs, subject axis first")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--method", choices=["ig", "gradcam", "attention"], required=True)
    parser.add_argument("--output-kind", choices=["logit", "regression", "probability"], required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--zero-baseline", action="store_true")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--delta-atol", type=float, default=1e-3)
    parser.add_argument("--delta-rtol", type=float, default=0.05)
    parser.add_argument("--layer")
    parser.add_argument("--cam-input")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.trust_adapter:
        raise ValueError("Review adapter code, then explicitly opt in with --trust-adapter")
    if args.method == "ig" and (bool(args.baseline) == args.zero_baseline):
        raise ValueError("Choose exactly one explicit baseline: --baseline or --zero-baseline")
    if args.method != "ig" and (args.baseline or args.zero_baseline):
        raise ValueError("Baselines apply only to IG")
    if args.method in {"gradcam", "attention"} and not args.layer:
        raise ValueError("This method requires --layer")
    if args.target < 0 or not np.isfinite([args.delta_atol, args.delta_rtol]).all() or min(args.delta_atol, args.delta_rtol) < 0:
        raise ValueError("Invalid target or completeness tolerances")
    arrays = load_arrays(args.inputs)
    n = len(next(iter(arrays.values())))
    if n == 0 or any(len(v) != n for v in arrays.values()):
        raise ValueError("All model inputs must share a nonempty subject axis")
    meta = read_json(args.metadata)
    ids = subject_ids(meta.get("subject_ids"), n)
    if not isinstance(meta.get("preprocessing"), str) or not meta["preprocessing"].strip():
        raise ValueError("Metadata must document the exact training-fitted preprocessing")
    baseline_arrays = load_arrays(args.baseline) if args.baseline else {k: np.zeros_like(v) for k, v in arrays.items()}
    if args.method == "ig" and (set(baseline_arrays) != set(arrays) or any(baseline_arrays[k].shape != arrays[k].shape for k in arrays)):
        raise ValueError("Baseline arrays must have the same keys and shapes as inputs")
    references, ref_paths = spatial_references(meta, args.metadata, arrays)
    spec = importlib.util.spec_from_file_location("nd_user_explanation_adapter", args.adapter)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the specified adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    model = adapter.build_model()
    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model() must return torch.nn.Module")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    forward = getattr(adapter, "forward_scores", None)
    if forward is None:
        if set(arrays) != {"inputs"}:
            raise ValueError("Default forward requires NPZ key 'inputs'; otherwise implement forward_scores")
        forward = lambda model, inputs: model(inputs["inputs"])
    collected, diagnostics = {}, []
    for i, identifier in enumerate(ids):
        inputs = {k: torch.tensor(v[i:i + 1], dtype=torch.float32, device=args.device) for k, v in arrays.items()}
        row = {"subject_id": identifier, "target": args.target, "output_kind": args.output_kind}
        if args.method == "ig":
            baselines = {k: torch.tensor(v[i:i + 1], dtype=torch.float32, device=args.device) for k, v in baseline_arrays.items()}
            attrs, score, baseline_score, delta = integrated_gradients(model, forward, inputs, baselines, args.target, args.output_kind, args.steps)
            passed = bool(abs(delta.item()) <= args.delta_atol + args.delta_rtol * abs((score - baseline_score).item()))
            row.update(score=score.item(), baseline_score=baseline_score.item(), completeness_delta=delta.item(), completeness_pass=passed)
            outputs = {f"ig_{key}": value.cpu().numpy() for key, value in attrs.items()}
        elif args.method == "gradcam":
            cam, score = grad_cam(model, forward, inputs, args.target, args.output_kind, args.layer, args.cam_input)
            row.update(score=score.item(), all_zero_map=bool(torch.count_nonzero(cam) == 0))
            outputs = {"gradcam": cam.cpu().numpy()}
        else:
            weights, score = attention_weights(model, forward, inputs, args.layer, args.target, args.output_kind)
            row.update(score=score.item())
            outputs = {"attention": weights.cpu().numpy()}
        for key, value in outputs.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite explanation; no completion manifest will be written")
            collected.setdefault(key, []).append(value)
        diagnostics.append(row)
    outputs = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    if args.method == "attention":
        q, k = outputs["attention"].shape[-2:]
        for axis, size in (("query_labels", q), ("key_labels", k)):
            labels = meta.get("attention", {}).get(axis)
            if not isinstance(labels, list) or len(labels) != size or any(not isinstance(x, str) or not x for x in labels):
                raise ValueError(f"Metadata attention.{axis} must identify all {size} tokens/ROIs (including padding)")
    feature_names = meta.get("feature_names", {})
    for key, names in feature_names.items():
        if (key not in arrays or arrays[key].ndim != 2 or not isinstance(names, list)
                or len(names) != arrays[key].shape[1] or any(not isinstance(x, str) or not x for x in names)
                or len(set(names)) != len(names)):
            raise ValueError("feature_names must identify each feature of a tabular input")
    config = {key: value for key, value in vars(args).items() if key not in {"trust_adapter"}}
    config["metadata"] = meta
    config["torch_version"] = torch.__version__
    config["evaluation_input_dtype"] = "float32"
    sources = [args.adapter, args.checkpoint, args.inputs, args.metadata, *ref_paths] + ([args.baseline] if args.baseline else [])
    bundle = OutputBundle(args.output_dir, f"model-{args.method}", config, sources)
    required = ["attributions" if args.method != "attention" else "attention", "explanation_figure", "diagnostics", "qc"]
    for key, data in outputs.items():
        input_key = key.removeprefix("ig_") if args.method == "ig" else args.cam_input
        refs = references.get(input_key) if args.method != "attention" else None
        export_array(bundle, key, data, required[0], ids, refs)
        if refs:
            required.append("attribution_map")
        if args.method == "ig" and input_key in feature_names:
            rows = [{"subject_id": ids[i], "feature": name, "attribution": float(data[i, j]), "target": args.target}
                    for i in range(n) for j, name in enumerate(feature_names[input_key])]
            filename = f"{key}_features.csv"
            pd.DataFrame(rows).to_csv(bundle.root / filename, index=False)
            bundle.add(filename, "feature_attributions", "csv", rows=n * len(feature_names[input_key]), columns=["subject_id", "feature", "attribution", "target"])
            required.append("feature_attributions")
    pd.DataFrame(diagnostics).to_csv(bundle.root / "diagnostics.csv", index=False)
    bundle.add("diagnostics.csv", "diagnostics", "csv", rows=n, subject_ids=ids, columns=["subject_id", "target", "score"])
    passed = all(row.get("completeness_pass", True) for row in diagnostics)
    qc = {"numerical_check_passed": passed, "n_subjects": n,
          "warnings": ["Attention is not target-specific attribution or causal evidence.",
                       "Explanations do not establish biomarker validity, clinical utility or causality.",
                       "Review baseline/target and use independent perturbation or randomization checks before scientific claims."],
          "attention_axes": (["subject", "heads", "query", "key"] if outputs.get("attention", np.zeros(0)).ndim == 4
                             else ["subject", "query", "key"]) if args.method == "attention" else None,
          "attention_target_conditioned": False if args.method == "attention" else None,
          "map_units": "selected score contribution" if args.method == "ig" else "model-dependent arbitrary units",
          "gradcam_relu": args.method == "gradcam"}
    write_json(bundle.root / "qc.json", qc)
    bundle.add("qc.json", "qc", "json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(outputs), figsize=(6 * len(outputs), 5), squeeze=False)
    for ax, (key, values) in zip(axes[0], outputs.items()):
        data = values[0]
        if args.method == "attention":
            if data.ndim == 3:
                data = data.mean(axis=0)
            handle = ax.imshow(data, cmap="viridis", aspect="auto")
            ax.set_xlabel("key index"); ax.set_ylabel("query index")
        elif data.ndim == 1:
            order = np.argsort(np.abs(data))[-min(25, len(data)):]
            ax.barh(range(len(order)), data[order])
            names = feature_names.get(key.removeprefix("ig_"), [str(j) for j in range(len(data))])
            ax.set_yticks(range(len(order)), [names[j] for j in order])
            handle = None
        else:
            if data.ndim == 4:
                # Descriptive channel sum and central voxel slice; originals stay unmodified.
                data = data.sum(axis=0)[:, :, data.shape[-1] // 2].T
            elif data.ndim == 3:
                data = data.sum(axis=0)
            scale = max(float(np.max(np.abs(data))), 1e-12)
            handle = ax.imshow(data, cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
        if handle is not None:
            fig.colorbar(handle, ax=ax, shrink=.7)
        ax.set_title(f"{key}: first subject, target {args.target}" + (" (attention not target-specific)" if args.method == "attention" else ""))
    fig.suptitle("Research explanation preview; full subject arrays are the primary output")
    fig.tight_layout()
    fig.savefig(bundle.root / "explanation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    bundle.add("explanation.png", "explanation_figure")
    if not passed:
        write_json(bundle.root / "FAILED_QC.json", {"reason": "IG completeness tolerance exceeded", "qc": qc})
        raise ValueError("IG completeness failed; diagnostics preserved without completion manifest. Review/retry explicitly in a new directory.")
    print(bundle.finish(sorted(set(required))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
