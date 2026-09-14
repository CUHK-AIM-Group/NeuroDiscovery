"""Synthetic numerical and CLI tests for clinical-research output skills.

No patient data, network requests, model-provider calls or frozen experiments.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

nib = pytest.importorskip("nibabel")
pytest.importorskip("matplotlib")

from models.common.research_outputs import OutputBundle, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ctp = load_script("ctp", "skills/ctp-skill/scripts/ctp.py")
explain = load_script("research_explain", "skills/model-interpretability/scripts/explain.py")
subtypes = load_script("research_subtypes", "skills/subject-subtyping/scripts/report_subtypes.py")


def nifti(path, data, affine=None):
    img = nib.Nifti1Image(np.asarray(data, np.float32), np.diag([2., 3., 4., 1.]) if affine is None else affine)
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, path)
    return path


@pytest.fixture
def ctp_case(tmp_path):
    shape, n, dt, flow = (4, 5, 3), 48, .5, 50.0
    aif = np.zeros(n)
    aif[4] = 80
    k = np.zeros(n)
    k[4:12] = flow * 1.04 / 6000
    tissue = np.convolve(aif, k)[:n] * dt
    hu = np.broadcast_to(40 + tissue, shape + (n,)).copy()
    hu[0, 0, 0] = 40 + aif
    nifti(tmp_path / "series.nii.gz", hu)
    mask = np.ones(shape); mask[0, 0, 0] = 0
    nifti(tmp_path / "mask.nii.gz", mask)
    nifti(tmp_path / "aif.nii.gz", 1 - mask)
    atlas = np.ones(shape); atlas[2:] = 2
    nifti(tmp_path / "atlas.nii.gz", atlas)
    write_json(tmp_path / "acquisition.json", {"signal_units": "HU", "motion_corrected": True,
               "baseline_frames": [0, 1], "frame_times_s": (np.arange(n) * dt).tolist()})
    return tmp_path, mask.astype(bool)


def compute_args(path, output="maps"):
    return ["compute", "--series", str(path / "series.nii.gz"), "--metadata", str(path / "acquisition.json"),
            "--mask", str(path / "mask.nii.gz"), "--aif-mask", str(path / "aif.nii.gz"),
            "--atlas", str(path / "atlas.nii.gz"), "--output-dir", str(path / output)]


def test_ctp_known_flow_volume_transit_and_delay(ctp_case):
    path, mask = ctp_case
    ctp.main(compute_args(path))
    expected = {"CBF": 50., "CBV": 50. * 4 / 60, "MTT": 4., "Tmax": 2.}
    for name, value in expected.items():
        img = nib.load(path / "maps" / f"{name}.nii.gz")
        assert np.allclose(img.get_fdata()[mask], value, atol=2e-4)
        assert np.isnan(img.get_fdata()[~mask]).all()
        assert np.allclose(img.affine, nib.load(path / "series.nii.gz").affine)
    manifest = validate_manifest(path / "maps/run_manifest.json", [*expected, "roi_summary", "curves"])
    assert manifest["research_only"]
    assert len(pd.read_csv(path / "maps/roi_summary.csv")) == 12


def test_ctp_vof_scaling_and_negative_delay():
    aif = np.zeros(24); aif[4] = 50
    residue = np.zeros(24); residue[2:6] = [.01, .009, .008, .007]
    tissue = np.convolve(aif, residue)[:24]
    maps, _ = ctp.deconvolve(tissue[None], aif, 1)
    scaled, _ = ctp.deconvolve(tissue[None], 2 * aif, 1)
    assert scaled["CBF"][0] == pytest.approx(maps["CBF"][0] / 2)
    assert scaled["CBV"][0] == pytest.approx(maps["CBV"][0] / 2)
    assert scaled["MTT"][0] == pytest.approx(maps["MTT"][0])
    early = np.roll(tissue, -4)
    result, _ = ctp.deconvolve(early[None], aif, 1)
    assert result["Tmax"][0] < 0


@pytest.mark.parametrize("field,value,match", [
    ("signal_units", "arbitrary", "HU"), ("motion_corrected", False, "motion"),
    ("baseline_frames", [0], "baseline"), ("baseline_frames", [0, 2], "baseline"),
    ("frame_times_s", [0] * 48, "frame_times"),
])
def test_ctp_invalid_metadata_rejected(ctp_case, field, value, match):
    path, _ = ctp_case
    meta = json.loads((path / "acquisition.json").read_text())
    meta[field] = value
    write_json(path / "acquisition.json", meta)
    with pytest.raises(ValueError, match=match):
        ctp.main(compute_args(path))
    assert not (path / "maps/run_manifest.json").exists()


def test_ctp_nonuniform_timing_is_explicit(ctp_case):
    path, _ = ctp_case
    meta = json.loads((path / "acquisition.json").read_text())
    meta["frame_times_s"][10] += .1
    write_json(path / "acquisition.json", meta)
    with pytest.raises(ValueError, match="Nonuniform"):
        ctp.main(compute_args(path))
    ctp.main(compute_args(path) + ["--resample-dt", "0.5"])
    validate_manifest(path / "maps/run_manifest.json")


def test_ctp_rejects_bad_mask_geometry_and_zero_aif(ctp_case):
    path, _ = ctp_case
    original = nib.load(path / "mask.nii.gz")
    affine = original.affine.copy(); affine[0, 3] += 10
    nifti(path / "mask.nii.gz", original.get_fdata(), affine)
    with pytest.raises(ValueError, match="affine"):
        ctp.main(compute_args(path))
    with pytest.raises(ValueError, match="positive enhancement"):
        ctp.deconvolve(np.ones((1, 8)), np.zeros(8), 1)
    with pytest.raises(ValueError, match="controls"):
        ctp.deconvolve(np.ones((1, 8)), np.ones(8), 1, cutoff=0)


def test_supplied_perfusion_maps_and_units(ctp_case):
    path, mask = ctp_case
    args = ["summarize", "--mask", str(path / "mask.nii.gz"), "--metadata", str(path / "units.json"),
            "--output-dir", str(path / "supplied")]
    for name in ctp.UNITS:
        source = nifti(path / f"source_{name}.nii.gz", np.ones(mask.shape))
        args.extend([f"--{name.lower()}", str(source)])
    write_json(path / "units.json", {"units": {**ctp.UNITS, "MTT": "ms"}})
    with pytest.raises(ValueError, match="units"):
        ctp.main(args)
    write_json(path / "units.json", {"units": ctp.UNITS})
    ctp.main(args)
    validate_manifest(path / "supplied/run_manifest.json", ["CBF", "Tmax"])


def test_motion_correction_identity_and_physical_coordinates(tmp_path):
    pytest.importorskip("SimpleITK")
    motion = load_script("ctp_motion", "skills/ctp-skill/scripts/motion_correct.py")
    grid = np.indices((12, 14, 10)).astype(float)
    phantom = (100 * np.exp(-sum((grid[i] - v) ** 2 for i, v in enumerate([4., 6., 3.])) / 10)
               + 40 * np.exp(-sum((grid[i] - v) ** 2 for i, v in enumerate([8., 9., 7.])) / 5))
    hu = np.repeat(phantom[..., None], 6, axis=3)
    affine = np.diag([-2., -3., 4., 1.]); affine[:3, 3] = [10, 20, 30]
    nifti(tmp_path / "motion.nii.gz", hu, affine)
    write_json(tmp_path / "meta.json", {"signal_units": "HU", "baseline_frames": [0, 1], "frame_times_s": list(range(6))})
    image = motion.sitk_image(phantom, affine)
    assert np.allclose(image.TransformIndexToPhysicalPoint((1, 2, 3)), [-8, -14, 42])
    motion.main(["--series", str(tmp_path / "motion.nii.gz"), "--metadata", str(tmp_path / "meta.json"),
                 "--iterations", "10", "--output-dir", str(tmp_path / "aligned")])
    result = nib.load(tmp_path / "aligned/motion_corrected.nii.gz")
    assert np.allclose(result.affine, affine)
    assert np.array_equal(result.get_fdata()[..., 0], hu[..., 0].astype(np.float32))
    assert not json.loads((tmp_path / "aligned/metadata_for_review.json").read_text())["motion_corrected"]
    validate_manifest(tmp_path / "aligned/run_manifest.json", ["motion_transform", "coverage_mask"])


def test_linear_ig_multi_input_completeness_and_training_state():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., -2.], [3., 4.]]))
    model.train()
    model.weight.grad = torch.ones_like(model.weight)
    inputs = {"left": torch.tensor([[2., 5.]]), "right": torch.tensor([[1., 1.]])}
    baselines = {key: torch.zeros_like(x) for key, x in inputs.items()}
    attrs, score, base, delta = explain.integrated_gradients(model, lambda m, x: m(x["left"] + x["right"]),
                inputs, baselines, 0, "logit", steps=16)
    assert torch.allclose(attrs["left"], torch.tensor([[2., -10.]]))
    assert torch.allclose(attrs["right"], torch.tensor([[1., -2.]]))
    assert abs(delta.item()) < 1e-6
    assert model.training
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight))


def test_constant_ig_and_invalid_targets():
    model = nn.Identity()
    attrs, _, _, delta = explain.integrated_gradients(model, lambda m, x: torch.ones(1),
        {"x": torch.ones(1, 2)}, {"x": torch.zeros(1, 2)}, 0, "regression", steps=4)
    assert torch.count_nonzero(attrs["x"]) == 0 and delta.item() == 0
    with pytest.raises(ValueError, match="target"):
        explain.selected_scores(torch.ones(1, 2), 1, 2, "logit")


@pytest.mark.parametrize("ndim", [2, 3])
def test_gradcam_known_localization_and_hook_cleanup(ndim):
    conv = nn.Conv2d(1, 1, 1, bias=False) if ndim == 2 else nn.Conv3d(1, 1, 1, bias=False)
    pool = nn.AdaptiveAvgPool2d(1) if ndim == 2 else nn.AdaptiveAvgPool3d(1)
    model = nn.Sequential(conv, pool, nn.Flatten())
    with torch.no_grad():
        conv.weight.fill_(1)
    x = torch.zeros((1, 1) + (4,) * ndim)
    x[(0, 0) + (2,) * ndim] = 10
    cam, _ = explain.grad_cam(model, lambda m, v: m(v["x"]), {"x": x}, 0, "regression", "0", "x")
    assert cam.shape == x.shape
    assert torch.allclose(cam, x / 4 ** ndim)
    assert not conv._forward_hooks
    with pytest.raises(ValueError):
        explain.grad_cam(model, lambda m, v: m(v["x"]), {"x": x}, 1, "regression", "0", "x")
    assert not conv._forward_hooks


def test_bnt_attention_is_real_weights_not_dec_assignments():
    from models.bnt.net.bnt import BrainNetworkTransformer
    torch.manual_seed(3)
    model = BrainNetworkTransformer(n_roi=8, sizes=[4], nhead=2, hidden_size=16, dropout=0.)
    forward = lambda m, v: m(v["fc"])[0]
    weights, _ = explain.attention_weights(model, forward, {"fc": torch.randn(1, 8, 8)}, "attention_list.0", 0, "logit")
    assert weights.shape == (1, 8, 8)  # not DEC [1,8,4]
    assert torch.allclose(weights.sum(-1), torch.ones(1, 8), atol=1e-5)


@pytest.fixture
def tabular_explanation(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("from torch import nn\ndef build_model():\n    return nn.Linear(3, 2, bias=False)\n", encoding="utf-8")
    model = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2., -3.], [2., 0., 4.]]))
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    np.savez(tmp_path / "inputs.npz", inputs=np.array([[1., 2., 3.], [2., 1., 1.]], np.float32))
    write_json(tmp_path / "inputs.json", {"subject_ids": ["001", "002"], "preprocessing": "synthetic identity",
                                         "feature_names": {"inputs": ["a", "b", "c"]}})
    args = ["--adapter", str(adapter), "--trust-adapter", "--checkpoint", str(tmp_path / "weights.pt"),
            "--inputs", str(tmp_path / "inputs.npz"), "--metadata", str(tmp_path / "inputs.json"),
            "--method", "ig", "--output-kind", "logit", "--target", "0", "--zero-baseline", "--steps", "16",
            "--output-dir", str(tmp_path / "ig")]
    return tmp_path, args


def test_explanation_cli_artifact_contract_and_tampering(tabular_explanation):
    path, args = tabular_explanation
    explain.main(args)
    values = np.load(path / "ig/ig_inputs.npy")
    assert np.array_equal(values, [[1, 4, -9], [2, 2, -3]])
    validate_manifest(path / "ig/run_manifest.json", ["attributions", "feature_attributions", "diagnostics"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_manifest(path / "ig/run_manifest.json", ["attribution_map"])
    with pytest.raises(ValueError, match="new or empty"):
        explain.main(args)
    (path / "ig/ig_inputs_features.csv").write_text("subject_id,feature,attribution,target\n001,a,999,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_manifest(path / "ig/run_manifest.json")


def test_explanation_requires_trust_and_baseline(tabular_explanation):
    path, args = tabular_explanation
    with pytest.raises(ValueError, match="trust-adapter"):
        explain.main([value for value in args if value != "--trust-adapter"])
    with pytest.raises(ValueError, match="baseline"):
        explain.main([value for value in args if value != "--zero-baseline"])
    assert not (path / "ig/run_manifest.json").exists()


def test_ig_failed_completeness_preserves_diagnostics_without_success(tmp_path):
    (tmp_path / "adapter.py").write_text(
        "import torch\nfrom torch import nn\nclass Cubic(nn.Module):\n"
        "    def forward(self, x):\n        return (x ** 3).sum(1)\n"
        "def build_model():\n    return Cubic()\n", encoding="utf-8")
    torch.save({}, tmp_path / "weights.pt")
    np.savez(tmp_path / "inputs.npz", inputs=np.array([[2.]], np.float32))
    write_json(tmp_path / "meta.json", {"subject_ids": ["s1"], "preprocessing": "synthetic identity"})
    args = ["--adapter", str(tmp_path / "adapter.py"), "--trust-adapter", "--checkpoint", str(tmp_path / "weights.pt"),
            "--inputs", str(tmp_path / "inputs.npz"), "--metadata", str(tmp_path / "meta.json"),
            "--method", "ig", "--output-kind", "regression", "--target", "0", "--zero-baseline", "--steps", "2",
            "--delta-atol", "0.0001", "--delta-rtol", "0", "--output-dir", str(tmp_path / "failed")]
    with pytest.raises(ValueError, match="completeness failed"):
        explain.main(args)
    assert (tmp_path / "failed/FAILED_QC.json").is_file()
    assert (tmp_path / "failed/diagnostics.csv").is_file()
    assert not (tmp_path / "failed/run_manifest.json").exists()


def test_gradcam_cli_emits_real_spatial_maps(tmp_path):
    (tmp_path / "adapter.py").write_text(
        "from torch import nn\ndef build_model():\n"
        "    return nn.Sequential(nn.Conv3d(1,1,1,bias=False),nn.AdaptiveAvgPool3d(1),nn.Flatten())\n", encoding="utf-8")
    model = nn.Sequential(nn.Conv3d(1, 1, 1, bias=False), nn.AdaptiveAvgPool3d(1), nn.Flatten())
    with torch.no_grad():
        model[0].weight.fill_(1)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    inputs = np.ones((2, 1, 3, 4, 5), np.float32)
    np.savez(tmp_path / "inputs.npz", inputs=inputs)
    nifti(tmp_path / "reference.nii.gz", np.ones((3, 4, 5)))
    write_json(tmp_path / "meta.json", {"subject_ids": ["s1", "s2"], "preprocessing": "synthetic identity",
                                         "spatial_references": {"inputs": "reference.nii.gz"}})
    explain.main(["--adapter", str(tmp_path / "adapter.py"), "--trust-adapter", "--checkpoint", str(tmp_path / "weights.pt"),
        "--inputs", str(tmp_path / "inputs.npz"), "--metadata", str(tmp_path / "meta.json"), "--method", "gradcam",
        "--layer", "0", "--cam-input", "inputs", "--target", "0", "--output-kind", "regression",
        "--output-dir", str(tmp_path / "cam")])
    manifest = validate_manifest(tmp_path / "cam/run_manifest.json", ["attributions", "attribution_map"])
    assert len([x for x in manifest["artifacts"] if x["role"] == "attribution_map"]) == 2
    assert np.allclose(nib.load(tmp_path / "cam/gradcam_subject_0000.nii.gz").get_fdata(), 1 / 60)


def test_attention_cli_preserves_heads_and_rejects_attribution_contract(tmp_path):
    (tmp_path / "adapter.py").write_text(
        "from torch import nn\nclass Net(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.attn=nn.MultiheadAttention(4,2,batch_first=True)\n"
        "    def forward(self,x):\n        y,_=self.attn(x,x,x,need_weights=True,average_attn_weights=False)\n        return y.mean(1)\n"
        "def build_model():\n    return Net()\n", encoding="utf-8")
    module = load_script("test_attention_adapter", tmp_path / "adapter.py")
    torch.manual_seed(11)
    model = module.build_model()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    np.savez(tmp_path / "inputs.npz", inputs=data)
    write_json(tmp_path / "meta.json", {"subject_ids": ["s1", "s2"], "preprocessing": "synthetic identity",
        "attention": {"query_labels": ["a", "b", "c"], "key_labels": ["a", "b", "c"]}})
    explain.main(["--adapter", str(tmp_path / "adapter.py"), "--trust-adapter", "--checkpoint", str(tmp_path / "weights.pt"),
        "--inputs", str(tmp_path / "inputs.npz"), "--metadata", str(tmp_path / "meta.json"), "--method", "attention",
        "--layer", "attn", "--target", "0", "--output-kind", "logit", "--output-dir", str(tmp_path / "attn")])
    weights = np.load(tmp_path / "attn/attention.npy")
    assert weights.shape == (2, 2, 3, 3)
    model.eval()
    with torch.no_grad():
        for i in range(2):
            x = torch.from_numpy(data[i:i + 1])
            expected = model.attn(x, x, x, need_weights=True, average_attn_weights=False)[1].numpy()
            assert np.allclose(weights[i:i + 1], expected)
    validate_manifest(tmp_path / "attn/run_manifest.json", ["attention"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_manifest(tmp_path / "attn/run_manifest.json", ["attributions"])


def test_spatial_export_affine_and_multichannel_identity(tmp_path):
    reference_path = nifti(tmp_path / "reference.nii.gz", np.zeros((3, 4, 5)))
    refs, _ = explain.spatial_references({"spatial_references": {"x": "reference.nii.gz"}}, tmp_path / "meta.json",
                                       {"x": np.ones((1, 2, 3, 4, 5))})
    data = np.arange(120, dtype=np.float32).reshape(1, 2, 3, 4, 5)
    bundle = OutputBundle(tmp_path / "spatial", "test", {})
    explain.export_array(bundle, "ig_x", data, "attributions", ["001"], refs["x"])
    bundle.finish(["attributions", "attribution_map"])
    result = nib.load(tmp_path / "spatial/ig_x_subject_0000.nii.gz")
    assert result.shape == (3, 4, 5, 2)
    assert np.array_equal(result.get_fdata(), np.moveaxis(data[0], 0, -1))
    assert np.array_equal(result.affine, nib.load(reference_path).affine)
    assert result.header.get_xyzt_units() == ("mm", "unknown")


@pytest.fixture
def subtype_case(tmp_path):
    frame = pd.DataFrame({"subject_id": ["001", "002", "003", "004", "005", "006"],
                          "f1": [0., .1, .2, 5., 5.1, np.nan], "f2": [1., 1., 1.2, 6., 6.2, 6.],
                          "outcome_not_a_feature": [0, 0, 1, 0, 1, 1]})
    frame.to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"subject_id": frame.subject_id[::-1], "subtype": ["B"] * 3 + ["A"] * 3}).to_csv(tmp_path / "assignments.csv", index=False)
    write_json(tmp_path / "names.json", ["f1", "f2"])
    args = ["--features", str(tmp_path / "features.csv"), "--feature-columns", str(tmp_path / "names.json"),
            "--assignments", str(tmp_path / "assignments.csv"), "--output-dir", str(tmp_path / "report")]
    return tmp_path, frame, args


def test_subtype_report_joins_by_id_and_reports_missing_stability(subtype_case):
    path, frame, args = subtype_case
    subtypes.main(args)
    rows = pd.read_csv(path / "report/subtype_assignments.csv", dtype={"subject_id": str})
    assert rows.subject_id.tolist() == frame.subject_id.tolist()
    assert rows.subtype.tolist() == ["A"] * 3 + ["B"] * 3
    assert rows.missing_feature_count.tolist() == [0, 0, 0, 0, 0, 1]
    qc = json.loads((path / "report/qc.json").read_text())
    assert qc["stability_status"] == "not_evaluated"
    profiles = pd.read_csv(path / "report/subtype_profiles.csv")
    assert set(profiles.feature) == {"f1", "f2"}
    validate_manifest(path / "report/run_manifest.json", ["subtype_profiles", "subtype_figure"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_manifest(path / "report/run_manifest.json", ["stability"])


def test_subtype_replicate_label_permutation_and_mismatched_ids(subtype_case):
    path, frame, args = subtype_case
    replicate = pd.DataFrame({"subject_id": frame.subject_id, "subtype": ["Y"] * 3 + ["X"] * 3})
    replicate.to_csv(path / "replicate.csv", index=False)
    subtypes.main(args + ["--replicate", str(path / "replicate.csv")])
    stability = pd.read_csv(path / "report/stability.csv")
    assert (stability.jaccard == 1).all() and (stability.adjusted_rand_index == 1).all()
    validate_manifest(path / "report/run_manifest.json", ["stability"])
    replicate.loc[0, "subject_id"] = "not-a-patient"
    replicate.to_csv(path / "replicate.csv", index=False)
    with pytest.raises(ValueError, match="subject set"):
        subtypes.main(args + ["--replicate", str(path / "replicate.csv")])


def test_output_contract_blocks_traversal_and_incomplete_bundle(tmp_path):
    bundle = OutputBundle(tmp_path / "bundle", "test", {})
    write_json(bundle.root / "qc.json", {"ok": True})
    with pytest.raises(ValueError, match="escapes"):
        bundle.add("../outside.json", "outside", "json")
    bundle.add("qc.json", "qc", "json")
    with pytest.raises(ValueError, match="Missing required"):
        bundle.finish(["attributions"])
    assert not (bundle.root / "run_manifest.json").exists()


def test_new_skills_discovered_without_missing_dependencies():
    from core.skill_loader.validate_dag import load_skill_graph, find_missing_deps, find_cycles
    graph, metadata = load_skill_graph(ROOT / "skills")
    for name in ("ctp-skill", "model-interpretability", "subject-subtyping"):
        assert name in metadata
        assert name not in find_missing_deps(graph)
    assert not any({"ctp-skill", "model-interpretability"}.intersection(cycle) for cycle in find_cycles(graph))
