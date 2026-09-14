---
name: ctp-skill
description: "Process 4D CT perfusion (CTP) time series and supplied Tmax, CBF, CBV and MTT maps, including rigid motion correction, AIF/VOF-based deconvolution, spatial/unit validation and ROI/QC outputs. Use for CT perfusion, not ASL or DSC perfusion MRI."
license: MIT
layer: base
skill_type: tool
dependencies:
  - nibabel-skill
  - claw-shell
---
# CT Perfusion Skill

Research workflows for NeuroDiscovery. Read [the input and method contract](references/input-contract.md) before processing data. These reference implementations are not clinically validated diagnostic software; do not label an ischemic core, penumbra, treatment eligibility or patient outcome from their maps.

## Choose the input route

- **4D CT:** accept a NIfTI HU series and acquisition timing, reviewed precontrast frames, brain mask and arterial input function (AIF) mask. For DICOM, first use a validated converter with rescale slope/intercept and per-frame acquisition timing preserved; never guess slice ordering or time units. Shuttle/per-slice asynchronous data need explicit upstream temporal reconstruction.
- **Unaligned frames:** use `scripts/motion_correct.py`, inspect motion and all-frame coverage, then save a reviewed metadata file. Do not declare registration good just because the optimizer stopped. Register/select all masks in the fixed frame.
- **Already derived maps:** use `scripts/ctp.py summarize`. Require units and a common affine/grid; never infer vendor scaling or replace missing-value codes with meaningful zero perfusion.

Dependencies: `numpy scipy pandas nibabel matplotlib`; motion correction additionally needs `SimpleITK`. They are available through the project's `clinical-outputs` optional dependency group.

```bash
python skills/ctp-skill/scripts/motion_correct.py --series ctp.nii.gz --metadata acquisition.json --output-dir ctp_motion

python skills/ctp-skill/scripts/ctp.py compute --series aligned_ctp.nii.gz --metadata reviewed_acquisition.json --mask brain.nii.gz --aif-mask aif.nii.gz --atlas atlas_in_ct_space.nii.gz --output-dir ctp_maps

python skills/ctp-skill/scripts/ctp.py summarize --cbf cbf.nii.gz --cbv cbv.nii.gz --mtt mtt.nii.gz --tmax tmax.nii.gz --metadata map_units.json --mask brain.nii.gz --output-dir ctp_summary
```

For irregular acquisition times, explicitly choose `--resample-dt` (seconds) after reviewing sampling; the script otherwise rejects them. Optional `--vof-mask` applies venous area-based AIF scaling. Record and justify `--svd-cutoff`, `--density`, and any `--hematocrit-factor`; defaults are disclosed research assumptions, not scanner calibration. No automatic AIF selection is performed.

## Deliver and check

Return `CBF.nii.gz`, `CBV.nii.gz`, `MTT.nii.gz`, `Tmax.nii.gz`, `analysis_mask.nii.gz`, `roi_summary.csv`, `maps_qc.png`, `qc.json` and `run_manifest.json`; time-series processing also returns `curves.csv` and `curves_qc.png`. Preserve NaNs for invalid/outside-mask voxels and report valid counts. Review bolus truncation, residuals, negative Tmax and mask coverage before downstream analysis. Plots are labeled voxel-index slices, not a clinical radiological viewer.

```bash
python -m models.common.research_outputs ctp_maps/run_manifest.json --require CBF CBV MTT Tmax roi_summary map_figure qc
```

For cohort modeling, join map/ROI features to pseudonymous subject IDs with an explicit atlas and feature schema. Route patient grouping to `subject-subtyping` and trained-model explanations to `model-interpretability`; processing one scan does not authorize a cohort experiment or external upload.

Created At: 2026-09-14 14:38:01 HKT
Last Updated At: 2026-09-14 15:05:31.197 HKT
Author: chengwang96
