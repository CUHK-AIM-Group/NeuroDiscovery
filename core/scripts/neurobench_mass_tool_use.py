from __future__ import annotations

"""Mass batch: tool_use tasks T139-T148 (10 tasks)."""

from neurobench_taskkit import body, std_eval

CAT = "tool_use"


def _t(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, ev, save_to=std_eval(folder)))


TASKS = [
    _t(139, "afni_skullstrip", "AFNI 3dSkullStrip Brain Extraction",
       "Extract the brain from a T1w volume using AFNI `3dSkullStrip`, then "
       "visually and quantitatively compare the mask against a reference mask "
       "if one is provided.",
       ins=["Subject T1w NIfTI file (required)",
            "Reference brain mask (optional, for Dice comparison)"],
       cons=["Use AFNI `3dSkullStrip` (document the AFNI version).",
             "Report mask volume in ml."],
       outs=["`brain.nii.gz` (skull-stripped T1w)",
             "`brain_mask.nii.gz`",
             "`qc_overlay.png` (mask edges over T1w, 3 orthogonal slices)",
             "Dice score vs. reference mask if provided (in metadata JSON)"]),
    _t(140, "afni_3ddeconvolve", "AFNI 3dDeconvolve Task GLM",
       "Run a single-subject task-fMRI GLM with AFNI `3dDeconvolve` using the "
       "provided event timings, producing per-condition beta and t-stat maps.",
       ins=["Preprocessed task fMRI 4D NIfTI (required)",
            "Event timing files per condition (AFNI format, required)",
            "Motion parameter file (required)"],
       cons=["Use `3dDeconvolve` with `-polort A` and motion regressors; "
             "censoring above FD 0.5 mm if a censor file is provided.",
             "Model conditions with the BLOCK or GAM hemodynamic basis and "
             "document the choice."],
       outs=["`stats_bucket.nii.gz` (betas + t-stats)",
             "`design_matrix.xmat` + rendered design matrix PNG",
             "`fitts` and `errts` time series"]),
    _t(141, "ants_n4_biasfield", "ANTs N4 Bias Field Correction",
       "Correct intensity inhomogeneity in a T1w image using ANTs "
       "`N4BiasFieldCorrection` and quantify the improvement in intensity "
       "uniformity within a brain mask.",
       ins=["Subject T1w NIfTI file (required)",
            "Brain mask (optional)"],
       cons=["Use `N4BiasFieldCorrection -d 3`; document shrink factor and "
             "spline distance if changed from defaults.",
             "Report coefficient of variation (CV) of white-matter intensity "
             "before vs. after correction when a mask is available."],
       outs=["`t1w_n4.nii.gz` (bias-corrected image)",
             "`bias_field.nii.gz` (estimated field)",
             "`n4_report.json` (parameters + CV before/after)"]),
    _t(142, "ants_apply_transforms", "ANTs Apply Transforms: Atlas to Subject Space",
       "Warp a volumetric atlas (e.g. Harvard-Oxford or AAL) from MNI space "
       "into subject T1w space using an existing ANTs transform chain, with "
       "label-preserving interpolation.",
       ins=["Atlas NIfTI in MNI space (required)",
            "Subject T1w NIfTI (required)",
            "Existing warp + affine from a prior registration (required)"],
       cons=["Use `antsApplyTransforms -d 3` with "
             "`GenericLabel` (nearest-neighbour) interpolation for labels.",
             "Verify ROI count is preserved after warping."],
       outs=["`atlas_in_subject.nii.gz`",
             "`roi_count_check.json` (ROI labels before vs. after)"],
       ev=["Warped atlas must contain the same label set as the input atlas.",
           "This test case is manually evaluated."]),
    _t(143, "mrtrix_dwi2response", "MRtrix dwi2response (Dhollander)",
       "Estimate fibre response functions from a preprocessed DWI dataset "
       "using MRtrix3 `dwi2response dhollander`, suitable for multi-tissue "
       "CSD.",
       ins=["Preprocessed DWI (`dwi.mif`) with bval/bvec (required)",
            "Brain mask (required)"],
       cons=["Use `dwi2response dhollander`; document MRtrix3 version.",
             "Report the number of voxels selected for WM/GM/CSF responses."],
       outs=["`response_wm.txt`, `response_gm.txt`, `response_csf.txt`",
             "`response_voxels.mif` (selected-voxel mask)",
             "`response_plot.png` (response function profiles)"]),
    _t(144, "mrtrix_mrdegibbs", "MRtrix mrdegibbs Ringing Removal",
       "Remove Gibbs ringing artefacts from a raw DWI dataset using MRtrix3 "
       "`mrdegibbs` and quantify residual ringing on a high-b-value shell.",
       ins=["Raw DWI NIfTI/MIF with bval/bvec (required)"],
       cons=["Run `mrdegibbs` BEFORE any other preprocessing step (state this "
             "ordering in the log).",
             "Keep original data untouched; write corrected output as a new "
             "file."],
       outs=["`dwi_degibbs.mif`",
             "`degibbs_qc.png` (before/after axial slice comparison)",
             "`mrdegibbs_log.txt`"]),
    _t(145, "fslmaths_roi_mask", "FSL fslmaths ROI Mask Construction",
       "From a probabilistic atlas map, construct a binary ROI mask at a "
       "given probability threshold using FSL `fslmaths`, and report mask "
       "size and overlap with a subject-specific mask if provided.",
       ins=["Probabilistic atlas NIfTI (4D or single map, required)",
            "Probability threshold (default 0.25, configurable)",
            "Subject mask (optional)"],
       cons=["Use `fslmaths -thr` + `-bin`; document exact command line.",
             "Threshold must be stated in the output filenames."],
       outs=["`roi_thr{p}.nii.gz` (binary mask)",
             "`mask_stats.json` (voxel count, volume ml, overlap metrics)"]),
    _t(146, "nilearn_qc_report", "Nilearn fMRI QC Report Generation",
       "Generate a one-page HTML quality-control report for a preprocessed "
       "resting-state fMRI run using nilearn plotting: carpet plot, mean "
       "image, motion parameters, and FD time series.",
       ins=["Preprocessed resting-state fMRI 4D NIfTI (required)",
            "Confounds TSV with motion + framewise displacement (required)",
            "Brain mask (required)"],
       cons=["Use nilearn (`nilearn.plotting`, `nilearn.image`) only; no "
             "external QC packages.",
             "Report mean FD and number of FD > 0.5 mm volumes."],
       outs=["`qc_report.html` (self-contained)",
             "`qc_summary.json` (mean FD, scrubbed volumes, tSNR)"]),
    _t(147, "spm12_segment", "SPM12 Unified Segmentation",
       "Run SPM12 unified segmentation on a T1w image (via MATLAB, Octave, or "
       "nipype's SPM interface) producing GM/WM/CSF tissue probability maps.",
       ins=["Subject T1w NIfTI file (required)"],
       cons=["Use SPM12 `spm_preproc` (unified segmentation) with default "
             "tissue priors; document SPM version and runtime environment.",
             "Native-space outputs only; no DARTEL normalization."],
       outs=["`c1*.nii` GM, `c2*.nii` WM, `c3*.nii` CSF maps",
             "`tissue_volumes.csv` (per-tissue volume in ml)",
             "`segment_qc.png` (tissue map overlays)"]),
    _t(148, "dcm2bids_conversion", "dcm2bids Single-Session Conversion",
       "Convert one session of DICOMs to BIDS using `dcm2bids` with a "
       "written configuration file, then validate the output.",
       ins=["DICOM directory for one session (required)"],
       cons=["Write and keep the `dcm2bids` config JSON in the output "
             "directory.",
             "Run `dcm2bids_helper` first and record the helper output used "
             "to build the config.",
             "Output must pass `bids-validator` with no errors."],
       outs=["BIDS session tree (`sub-*/ses-*`)",
             "`dcm2bids_config.json`",
             "`bids_validation_report.txt`"]),
]
