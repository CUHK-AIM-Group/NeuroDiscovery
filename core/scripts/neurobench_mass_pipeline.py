from __future__ import annotations

"""Mass batch: pipeline_execution tasks T149-T202 (54 tasks).

Families of end-to-end / multi-step pipelines: denoising-strategy variants,
reconstruction-spec variants, dataset variants, modality-specific chains and
stage splits of larger pipelines.
"""

from neurobench_taskkit import body, std_eval

CAT = "pipeline_execution"


def _t(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, ev, save_to=std_eval(folder)))


BIDS_IN = ["BIDS dataset directory (required)",
           "FreeSurfer license file (required)"]
DERIV_IN = ["fMRIPrep derivatives directory for the subject(s) (required)"]

TASKS = []

# --- A. xcp_d denoising strategy variants (T149-T154) -----------------------
_XCPD = [
    ("36p", "36P", "36-parameter model (24 motion + WM/CSF + global signal "
     "with derivatives)"),
    ("36p_nogsr", "36P without GSR", "36-parameter model WITHOUT global "
     "signal regression"),
    ("acompcor", "aCompCor", "aCompCor (5 WM + 5 CSF components) + 24 motion "
     "parameters"),
    ("acompcor_gsr", "aCompCor+GSR", "aCompCor + 24 motion parameters + "
     "global signal regression"),
    ("aroma", "ICA-AROMA", "ICA-AROMA non-aggressive denoising + 6 motion "
     "parameters"),
    ("aroma_gsr", "ICA-AROMA+GSR", "ICA-AROMA non-aggressive denoising + "
     "6 motion parameters + GSR"),
]
for i, (slug, name, model) in enumerate(_XCPD):
    TASKS.append(_t(
        149 + i, f"xcpd_denoise_{slug}", f"XCP-D Denoising: {name}",
        f"Post-process fMRIPrep derivatives with XCP-D using the {model} "
        "strategy. Produce denoised BOLD, confound-filtering QC, and "
        "framewise-displacement statistics.",
        ins=DERIV_IN,
        cons=["Use the `xcp_d` BIDS app with the matching confound strategy "
              "documented in the command log.",
              "Band-pass filter 0.01-0.1 Hz; FD threshold 0.5 mm unless the "
              "strategy dictates otherwise.",
              "Report percent of volumes censored."],
        outs=["Denoised BOLD (NIfTI + CIFTI if available)",
              "`qc_report.html` (pre/post denoising carpet plots)",
              "`fd_summary.json` (mean FD, censored fraction)"]))

# --- B. QSIRecon reconstruction-spec variants (T155-T158) -------------------
_QSIR = [
    ("mrtrix_ss3t", "MRtrix SS3T CSD",
     "`--recon-spec mrtrix_multishell_msmt` with SS3T CSD + 1M-streamline "
     "tracking"),
    ("amico_noddi", "AMICO NODDI",
     "`--recon-spec amico_noddi` producing NDI/ODI/ISOVF maps"),
    ("dipy_dki", "DIPY DKI",
     "`--recon-spec dipy_dki` producing kurtosis tensor metrics"),
    ("dsistudio_gqi_act", "DSI Studio GQI + ACT",
     "`--recon-spec dsi_studio_gqi` with anatomically-constrained tracking"),
]
for i, (slug, name, spec) in enumerate(_QSIR):
    TASKS.append(_t(
        155 + i, f"qsirecon_{slug}", f"QSIRecon Reconstruction: {name}",
        f"Run QSIRecon on QSIPrep derivatives using {spec}. Produce the "
        "reconstructed outputs plus a per-stage log.",
        ins=["QSIPrep derivatives directory for one subject (required)"],
        cons=["Container execution (Docker or Singularity) with pinned "
              "version.",
              "Document the exact recon-spec JSON/YAML used."],
        outs=["Reconstruction outputs per spec (tractogram and/or scalar "
              "maps)",
              "`recon_spec_used.json`",
              "`qsirecon_log.txt` (runtime per stage)"]))

# --- C. fMRIPrep dataset variants (T159-T163) -------------------------------
_FPR = [
    ("adni", "ADNI", "an ADNI subject (longitudinal T1w handling noted if "
     "multiple sessions)"),
    ("oasis3", "OASIS-3", "an OASIS-3 subject"),
    ("abide", "ABIDE", "an ABIDE subject (document the site)"),
    ("multiecho_tedana", "multi-echo + tedana", "a multi-echo fMRI subject "
     "with multi-echo ICA (tedana) enabled"),
    ("fmapless_syn", "fieldmap-less SyN-SDC", "a subject WITHOUT fieldmaps "
     "using `--use-syn-sdc` distortion correction"),
]
for i, (slug, name, what) in enumerate(_FPR):
    TASKS.append(_t(
        159 + i, f"fmriprep_{slug}", f"fMRIPrep Pipeline: {name}",
        f"Run the full fMRIPrep anatomical + functional workflow for {what}. "
        "Produce standard fMRIPrep derivatives and the HTML QC report.",
        ins=BIDS_IN,
        cons=["fMRIPrep LTS version pinned in the log; output spaces "
              "MNI152NLin2009cAsym + T1w (add fsnative for the multi-echo "
              "variant).",
              "Document any non-default flags and justify them."],
        outs=["fMRIPrep derivatives tree (anat + func)",
              "`sub-*.html` QC report",
              "`run_summary.json` (version, flags, wall time)"]))

# --- D. MRIQC group QC batches (T164-T166) ----------------------------------
_MRIQC = [
    ("smri", "structural (T1w)", "T1w"),
    ("fmri", "functional (BOLD)", "BOLD"),
    ("dwi", "diffusion (DWI)", "DWI"),
]
for i, (slug, what, mod) in enumerate(_MRIQC):
    TASKS.append(_t(
        164 + i, f"mriqc_{slug}_group", f"MRIQC Group QC: {what}",
        f"Run MRIQC participant + group level over all {mod} scans in a BIDS "
        "dataset, producing per-scan IQMs and the group report, then flag "
        "outlier scans.",
        ins=["BIDS dataset directory (required)"],
        cons=["MRIQC container with pinned version.",
              "Outlier rule: any IQM more than 1.5x IQR from the group "
              "distribution; document the IQM(s) used."],
        outs=["Per-participant IQM JSON/TSV + group TSV",
              "`group_{mod}.html` report",
              "`outliers.csv` (flagged scans + reason)"]))

# --- E. Ciftify (T167-T168) --------------------------------------------------
TASKS.append(_t(
    167, "ciftify_fmri", "ciftify fMRI CIFTI Workflow",
    "Run ciftify `ciftify_recon_all` + `ciftify_subject_fmri` to map one "
    "subject's fMRI to CIFTI grayordinates from FreeSurfer recon-all output.",
    ins=["BIDS dataset with fMRI (required)",
         "FreeSurfer recon-all output for the subject (required)"],
    cons=["Use ciftify (HCP-derived surfaces, 32k fsLR).",
          "Document the HCP templates version."],
    outs=["`*.dtseries.nii` CIFTI time series",
          "Surface QC scene/PNG",
          "`ciftify_log.txt`"]))
TASKS.append(_t(
    168, "ciftify_clean", "ciftify Clean + Dense Connectome",
    "From ciftify fMRI output, run `ciftify_clean_img` denoising and compute "
    "a dense functional connectome with `cifti_conn_matrix`.",
    ins=["ciftify fMRI output for one subject (required)"],
    cons=["Cleaning: detrend, band-pass 0.01-0.1 Hz, 24-motion + WM/CSF "
          "regression; document the cleaning config.",
          "Smoothing FWHM must be stated."],
    outs=["`cleaned.dtseries.nii`",
          "`dconn.nii` or dense correlation matrix",
          "`clean_config.json`"]))

# --- F. CPAC (T169-T170) ------------------------------------------------------
TASKS.append(_t(
    169, "cpac_default_preproc", "C-PAC Default Preprocessing Pipeline",
    "Run C-PAC's default preprocessing pipeline (anatomical + functional, "
    "nuisance regression with the default strategy) for one subject.",
    ins=BIDS_IN,
    cons=["Use the C-PAC container with the default pipeline configuration; "
          "keep the generated pipeline YAML.",
          "Document the C-PAC version."],
    outs=["C-PAC output directory (anat + func derivatives)",
          "`pipeline_config_used.yaml`",
          "`cpac_run_log.txt`"]))
TASKS.append(_t(
    170, "cpac_custom_strategy", "C-PAC Custom Nuisance Strategy",
    "Run C-PAC with a custom pipeline configuration: aCompCor nuisance "
    "regression, no GSR, band-pass 0.01-0.1 Hz, scrubbing at FD 0.5 mm.",
    ins=BIDS_IN + ["Base pipeline YAML to modify (optional)"],
    cons=["Start from a default C-PAC pipeline config and edit ONLY the "
          "nuisance/filter/scrubbing sections; show a diff of the YAML.",
          "Scrubbing must be reported as percent volumes removed."],
    outs=["C-PAC output directory",
          "`pipeline_config_diff.txt`",
          "`scrubbing_summary.json`"]))

# --- G. ASL (T171-T172) -------------------------------------------------------
TASKS.append(_t(
    171, "oxford_asl_pipeline", "Oxford ASL (BASIL) Perfusion Pipeline",
    "Run oxford_asl/BASIL on a pCASL dataset: motion correction, calibration "
    "with the M0 image, and CBF quantification in native + MNI space.",
    ins=["ASL NIfTI (label/control pairs, required)",
         "M0 calibration image (required)",
         "T1w NIfTI for registration (required)"],
    cons=["Use `oxford_asl` with structural + calibration inputs; document "
          "acquisition parameters (PLD, labeling duration) from the "
          "BIDS sidecar.",
          "Register CBF to MNI152 and report both spaces."],
    outs=["`cbf_native.nii.gz`, `cbf_mni.nii.gz`",
          "`basil_report.txt` (parameters + fit statistics)",
          "`cbf_qc.png` overlay"]),
)
TASKS.append(_t(
    172, "aslprep_pipeline", "ASLPrep End-to-End Pipeline",
    "Run ASLPrep on a BIDS dataset with ASL data: full preprocessing "
    "including CBF computation, BASIL partial-volume correction, and QC "
    "report.",
    ins=BIDS_IN,
    cons=["ASLPrep container with pinned version; default options unless "
          "justified in the log.",
          "Note whether pCASL or PASL and handle accordingly."],
    outs=["ASLPrep derivatives (CBF maps, QC HTML)",
          "`run_summary.json`"]))

# --- H. PET (T173) ------------------------------------------------------------
TASKS.append(_t(
    173, "petprep_pipeline", "PETPrep End-to-End Pipeline",
    "Run PETPrep on a BIDS dataset with PET data: motion correction, "
    "coregistration to T1w, and uptake-ratio images with the reference "
    "region documented.",
    ins=BIDS_IN + ["Reference-region specification (required)"],
    cons=["PETPrep container with pinned version.",
          "State the reference region (e.g. cerebellum) and SUVr formula."],
    outs=["PETPrep derivatives (coregistered PET, SUVr maps)",
          "`petprep_report.html`"]))

# --- I. EEG (T174-T176) -------------------------------------------------------
TASKS.append(_t(
    174, "mne_rest_full_pipeline", "MNE Resting-State EEG Full Pipeline",
    "Full resting-state EEG pipeline with MNE-Python: preprocessing "
    "(filter, ICA), sensor-level spectral analysis (PSD + band power), and "
    "connectivity (wPLI) between regions.",
    ins=["Resting-state EEG recording (required)",
         "Channel montage/positions (required)"],
    cons=["MNE-Python throughout; band-pass 1-40 Hz.",
          "Report PSD per canonical band (delta..gamma) per channel group.",
          "Connectivity: wPLI on cleaned epochs; state epoch length."],
    outs=["`psd_by_band.csv`",
          "`wpli_connectome.npy` + matrix PNG",
          "`pipeline_report.html` (MNE Report)"]))
TASKS.append(_t(
    175, "mne_erp_full_pipeline", "MNE ERP Full Pipeline",
    "Event-related pipeline with MNE-Python: preprocessing, epoching, "
    "artifact rejection, evoked responses per condition, and a "
    "between-condition contrast with cluster-corrected sensor statistics.",
    ins=["Task EEG recording (required)",
         "Event definitions per condition (required)"],
    cons=["Document rejection thresholds and trial counts kept per "
          "condition.",
          "Statistics: spatio-temporal cluster permutation test."],
    outs=["`evoked_conditions.fif` + butterfly plots",
          "`contrast_stats.png` (significant clusters)",
          "`trial_counts.json`"]))
TASKS.append(_t(
    176, "mne_source_localization", "MNE EEG Source Localization (stage beyond T124)",
    "Source-localize cleaned epoched EEG: BEM forward model from the "
    "subject's FreeSurfer recon, dSPM inverse solution, and cortical "
    "activation maps per condition.",
    ins=["Cleaned epochs (FIF, e.g. from T124) (required)",
         "FreeSurfer subject directory (required)",
         "Coregistration fiducials (required)"],
    cons=["3-layer BEM; document conductivity values.",
          "Inverse: dSPM with loose=0.2, depth=0.8 unless justified."],
    outs=["`fwd.fif`, `inv.fif`, per-condition `stc` files",
          "`source_activation.png` (inflated brain, per condition)"]))

# --- J. MEG (T177) ------------------------------------------------------------
TASKS.append(_t(
    177, "meg_maxfilter_pipeline", "MEG MaxFilter + Preprocessing Pipeline",
    "Preprocess raw MEG with MaxFilter (SSS/tSSS + head-position "
    "correction), then filter, annotate bad segments, and produce a sensor "
    "QC summary.",
    ins=["Raw MEG FIF recording (required)",
         "Empty-room recording for noise covariance (required)"],
    cons=["MaxFilter via `mne.preprocessing.maxwell_filter` or Elekta "
          "software; document which.",
          "tSSS parameters and bad-channel detection logged."],
    outs=["`sss_raw.fif` (cleaned recording)",
          "`noise_cov.fif` from empty room",
          "`bad_channels.json` + PSD QC plot"]))

# --- K. FSL FEAT higher level (T178-T179) --------------------------------------
TASKS.append(_t(
    178, "feat_group_flame", "FSL FEAT Group-Level FLAME",
    "Run a group-level mixed-effects analysis (FLAME 1+2) in FSL FEAT over "
    "first-level COPE directories, with a two-group or one-sample design.",
    ins=["First-level FEAT directories for all subjects (required)",
         "Group design matrix + contrasts (required)"],
    cons=["FLAME 1+2; cluster threshold z>3.1, corrected p<0.05 unless the "
          "design dictates otherwise.",
          "Document the exact `design.fsf` settings."],
    outs=["Group FEAT directory with thresholded zstat maps",
          "`rendered_zstat.png` (MNI slices)",
          "`design.png` + `design.mat`"])),
TASKS.append(_t(
    179, "feat_firstlevel_blocked", "FSL FEAT First-Level Blocked Design",
    "Full first-level FEAT analysis for one subject: pre-stats (motion "
    "correction, smoothing), FILM GLM with a blocked design, registration "
    "to structural + MNI.",
    ins=["Raw task fMRI 4D NIfTI (required)",
         "T1w structural (required)",
         "3-column event files (required)"],
    cons=["FEAT GUI-equivalent `fsf` file kept in outputs; FWHM 5 mm.",
          "Registration: BBR to T1w, 12-dof to MNI."],
    outs=["`*.feat` directory (zstats, cope/varcope)",
          "`report.html` from FEAT",
          "`design.png`"]))

# --- L. Nilearn pipelines (T180-T181) ------------------------------------------
TASKS.append(_t(
    180, "nilearn_mvpa_searchlight", "Nilearn MVPA Searchlight Pipeline",
    "Run a whole-brain searchlight decoding analysis with nilearn: decode "
    "the given condition labels from task fMRI with an SVM, cross-validated "
    "per subject.",
    ins=["Task fMRI 4D NIfTI (required)",
         "Condition labels / events file (required)",
         "Brain mask (required)"],
    cons=["`nilearn.decoding.SearchLight` with radius 6 mm, SVC linear.",
          "Report chance level and permutation-based significance map."],
    outs=["`searchlight_accuracy.nii.gz`",
          "`permutation_threshold.json`",
          "`searchlight_report.md`"])),
TASKS.append(_t(
    181, "nilearn_group_ica_canica", "Nilearn Group ICA (CanICA) Pipeline",
    "Run CanICA group ICA over a set of resting-state subjects, extract the "
    "20-component atlas, and back-reconstruct subject maps for downstream "
    "analysis.",
    ins=["List of preprocessed resting-state 4D NIfTIs (>= 5 subjects, "
         "required)"],
    cons=["`nilearn.decomposition.CanICA` with 20 components, standard "
          "smoothing 6 mm.",
          "Label components against a reference atlas (Yeo 7-network "
          "overlap table)."],
    outs=["`canica_components.nii.gz` + component montage PNG",
          "`yeo_overlap_table.csv`",
          "Per-subject back-reconstructed maps"]))

# --- M. FreeSurfer longitudinal (T182) ------------------------------------------
TASKS.append(_t(
    182, "fs_longitudinal_pipeline", "FreeSurfer Longitudinal Pipeline",
    "Run the FreeSurfer longitudinal stream for one subject with 2+ time "
    "points: cross-sectional recon-all per TP, unbiased base template, then "
    "longitudinal runs, and extract longitudinal volume/thickness changes.",
    ins=["T1w NIfTIs for >= 2 time points (required)",
         "FreeSurfer license (required)"],
    cons=["Use `recon-all -base` + `-long` stream; document FreeSurfer "
          "version.",
          "Do NOT average time points cross-sectionally."],
    outs=["Base + longitudinal recon directories",
          "`longitudinal_change.csv` (per-region annualized change)",
          "`qc_snapshots.png`"]))

# --- N. VBM (T183-T184) ---------------------------------------------------------
TASKS.append(_t(
    183, "cat12_vbm_pipeline", "CAT12 VBM Pipeline",
    "Run a CAT12 VBM analysis for a group of T1w images: segmentation, "
    "DARTEL normalization, modulation, smoothing, and a two-group GMV "
    "comparison.",
    ins=["T1w NIfTIs for both groups (>= 10 per group, required)",
         "Group assignment file (required)"],
    cons=["CAT12 via SPM batch (document CAT12 release).",
          "Smoothing 8 mm FWHM; TIV as covariate; threshold with TFCE or "
          "FWE correction."],
    outs=["Modulated normalized GM maps",
          "`group_diff_tstat.nii` + rendered PNG",
          "`cat12_qc_report.pdf`"])),
TASKS.append(_t(
    184, "fsl_vbm_pipeline", "FSL-VBM Pipeline",
    "Run the FSL-VBM protocol: brain extraction, tissue segmentation, "
    "study-specific GM template, nonlinear normalization, modulation, "
    "smoothing, and permutation stats with `randomise`.",
    ins=["T1w NIfTIs for the group (>= 10, required)",
         "Design matrix + contrasts for `randomise` (required)"],
    cons=["Follow the standard fslvbm_1/2/3 steps; document FSL version.",
          "Smoothing 3 mm (sigma-equivalent); 5000 permutations with TFCE."],
    outs=["`stats/` randomise outputs (TFCE corrected)",
          "`gm_template.nii.gz`",
          "`vbm_report.md`"]))

# --- O. DARTEL (T185) -----------------------------------------------------------
TASKS.append(_t(
    185, "spm_dartel_template", "SPM DARTEL Study Template",
    "Build a study-specific DARTEL template from segmented GM/WM images, "
    "normalize subjects to MNI via the template, and report template "
    "sharpness across iterations.",
    ins=["SPM segmentation outputs (c1/c2) for >= 10 subjects (required)"],
    cons=["`spm_dartel_template` with default 6 outer iterations.",
          "Visualize template evolution across iterations."],
    outs=["`Template_0..6.nii` series",
          "Flow fields + MNI-normalized tissue maps",
          "`template_evolution.png`"]))

# --- P. Hippocampal subfields (T186-T187) ----------------------------------------
TASKS.append(_t(
    186, "fs_hippo_subfields", "FreeSurfer Hippocampal Subfields",
    "Segment hippocampal subfields (and amygdala nuclei if T2 is available) "
    "with FreeSurfer `segmentHA_T1.sh`/`segmentHA_T2.sh`, extract volumes, "
    "and QC against expected ranges.",
    ins=["FreeSurfer recon-all output for the subject (required)",
         "T2w NIfTI (optional, improves subfield accuracy)"],
    cons=["Document FreeSurfer version (subfield labels differ across "
          "versions).",
          "Volumes in mm^3, corrected for eTIV in the report table."],
    outs=["`[lr]h.hippoSfVolumes.txt` parsed to CSV",
          "`subfield_qc.png` overlays",
          "`volume_range_flags.json`"])),
TASKS.append(_t(
    187, "ashs_hippo_segmentation", "ASHS Hippocampal Subfield Segmentation",
    "Run ASHS automatic segmentation of hippocampal subfields using a "
    "high-resolution T2 template package, extract volumes, and export "
    "label QC snapshots.",
    ins=["T1w + high-resolution T2w (oblique coronal) NIfTIs (required)",
         "ASHS template package (required)"],
    cons=["Document ASHS version and atlas package.",
          "Keep intermediate registration QC."],
    outs=["ASHS label maps (native space)",
          "`subfield_volumes.csv`",
          "`ashs_qc.pdf`"]))

# --- Q. Lesion (T188-T189) --------------------------------------------------------
TASKS.append(_t(
    188, "fsl_lesion_filling", "FSL Lesion Filling Pipeline",
    "Fill white-matter lesions in a T1w image using FSL `lesion_filling` "
    "prior to downstream volumetric analysis, given a lesion mask from "
    "FLAIR segmentation.",
    ins=["T1w NIfTI (required)",
         "Lesion mask (e.g. from T71 WMH segmentation) (required)"],
    cons=["Use FSL `lesion_filling`; document FSL version.",
          "QC: filled-region intensity should match surrounding WM "
          "(report mean intensities)."],
    outs=["`t1w_lesion_filled.nii.gz`",
          "`fill_qc.json` (intensity stats)",
          "`fill_overlay.png`"])),
TASKS.append(_t(
    189, "lst_lesion_segmentation", "LST Lesion Segmentation (SPM)",
    "Run SPM/LST lesion prediction (LPA) on FLAIR images for a small cohort "
    "and produce per-subject lesion maps + volumes.",
    ins=["FLAIR NIfTIs for the cohort (required)",
         "T1w NIfTIs (optional, used by LGA variant)"],
    cons=["LST toolbox via SPM batch; LPA unless T1w provided (then "
          "document LGA).",
          "Threshold lesions at the recommended probability; state it."],
    outs=["Per-subject `ples_*.nii` maps",
          "`lesion_volumes.csv`",
          "`lst_batch_log.txt`"]))

# --- R. Tractography -> connectome -> graph (T190-T191) --------------------------
TASKS.append(_t(
    190, "dwi_full_connectome_graph", "DWI Tractography-Connectome-Graph Pipeline",
    "Full MRtrix chain from preprocessed DWI: response estimation, "
    "multi-shell CSD, ACT tractography (10M), SIFT2, connectome construction "
    "with a chosen atlas, and graph-metric extraction.",
    ins=["Preprocessed DWI + mask + 5TT (required)",
         "Parcellation atlas in subject space (required)"],
    cons=["MRtrix3 throughout; document every command in `pipeline_log.sh`.",
          "Connectome weighted by SIFT2 streamline counts (invnode "
          "volume normalisation documented)."],
    outs=["`connectome_sift2.csv`",
          "`graph_metrics.json` (degree, strength, betweenness, efficiency)",
          "`tractogram_qc.png`"])),
TASKS.append(_t(
    191, "mrtrix_sift2_stage", "MRtrix SIFT2 Stage (stage split of T190)",
    "Stage split of T190: given a whole-brain tractogram and FOD, run SIFT2 "
    "to produce streamline weights, with convergence and QC reporting.",
    ins=["Tractogram `.tck` (required)",
         "FOD image `wmfod.mif` (required)"],
    cons=["Use `tcksift2` default regularisation; document iterations.",
          "Report the proportionality-coefficient QC plots."],
    outs=["`weights.csv` (per-streamline weights)",
          "`sift2_mu.txt` + QC PNG",
          "`sift2_log.txt`"]))

# --- S. Stage splits of BIDS apps (T192-T195) ------------------------------------
_TASKS_S = [
    (192, "fmriprep_anat_only", "fMRIPrep Anatomical-Only Stage",
     "Run fMRIPrep with `--anat-only`: surface reconstruction and anatomical "
     "normalization only, stopping before functional preprocessing.",
     ["Use `--anat-only`; confirm no func outputs are produced.",
      "This is a stage split of the full fMRIPrep pipeline."],
     ["`anat/` derivatives tree",
      "Anatomical QC HTML"]),
    (193, "fmriprep_func_from_derivs", "fMRIPrep Functional Stage from Existing Anat",
     "Run fMRIPrep functional preprocessing reusing existing anatomical "
     "derivatives via `--anat-derivatives`.",
     ["Use `--anat-derivatives` pointing at a completed anat run.",
      "Verify anatomical outputs are NOT recomputed (log evidence)."],
     ["`func/` derivatives tree",
      "Functional QC HTML"]),
    (194, "hcp_icafix_only", "HCP ICA+FIX Stage Only",
     "Run only the ICA+FIX denoising stage of the HCP functional pipeline "
     "on a preprocessed rfMRI run, using a trained FIX classifier.",
     ["Start from PostFreeSurfer/fMRIVolume outputs.",
      "Document the FIX training file used and threshold."],
     ["Cleaned dtseries (FIX-denoised)",
      "`fix_classification_report.txt`"]),
    (195, "xcpd_custom_confounds", "XCP-D with Custom Confound File",
     "Run XCP-D with a user-supplied custom confounds TSV merged with the "
     "standard motion parameters.",
     ["Custom confound columns must be documented (name, origin, units).",
      "Verify the final regression design includes the custom columns."],
     ["Denoised BOLD + design matrix TSV",
      "`confound_merge_report.json`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_S:
    ins = DERIV_IN if num in (192, 193, 195) else [
        "HCP minimal-preprocessing outputs for the subject (required)",
        "FIX training file (required)"]
    TASKS.append(_t(num, slug, title, desc, ins=ins, cons=cons, outs=outs))

# --- T. Multi-modal / CIFTI (T196-T197) ------------------------------------------
TASKS.append(_t(
    196, "multimodal_qc_pipeline", "Multi-Modal QC Aggregation Pipeline",
    "Aggregate quality control across modalities for one BIDS dataset: run "
    "or collect MRIQC IQMs for T1w/BOLD/DWI, merge with pipeline success "
    "logs, and produce a single subject x modality QC matrix.",
    ins=["BIDS dataset directory (required)",
         "Existing pipeline logs (optional)"],
    cons=["One row per subject x modality; status in {pass, warn, fail}.",
          "Rules for warn/fail must be explicit and reproducible."],
    outs=["`qc_matrix.csv`",
          "`qc_dashboard.html` (sortable table + distributions)",
          "`exclusion_recommendations.md`"])),
TASKS.append(_t(
    197, "hcp_style_cifti_full", "HCP-Style CIFTI Full-Subject Pipeline",
    "Produce full CIFTI outputs for one subject from fMRIPrep + FreeSurfer "
    "results: map rfMRI to 91k grayordinates, denoise, and build a dense "
    "connectome, following HCP conventions.",
    ins=["fMRIPrep + FreeSurfer derivatives for the subject (required)"],
    cons=["Use ciftify or equivalent wb_command steps; 32k fsLR surfaces.",
          "Document every wb_command call in `cifti_steps.sh`."],
    outs=["`rfMRI.dtseries.nii` (91k grayordinates)",
          "`dense_connectome.dconn.nii` or ROI connectome",
          "`cifti_steps.sh`"]))

# --- U. Onboarding / incremental (T198-T199) --------------------------------------
TASKS.append(_t(
    198, "onboarding_dicom_to_bids_qc", "DICOM-to-BIDS Onboarding Pipeline",
    "Full onboarding chain for a new scanner shipment: DICOM sorting, "
    "dcm2bids conversion, bids-validator, MRIQC, and a final onboarding "
    "report with pass/fail.",
    ins=["DICOM tar/directory as delivered by the scanner (required)"],
    cons=["Every stage logs its tool + version.",
          "Any stage failure must quarantine the subject and continue "
          "with the rest."],
    outs=["BIDS tree + validation report",
          "MRIQC outputs",
          "`onboarding_report.md` (per-subject status table)"])),
TASKS.append(_t(
    199, "bids_incremental_update", "BIDS Incremental Update Pipeline",
    "Add newly acquired sessions to an existing BIDS dataset: convert, "
    "merge, re-validate, update participants/scans TSVs, and emit a "
    "changelog diff of what changed.",
    ins=["Existing BIDS dataset (required)",
         "New DICOM/NIfTI session directories (required)"],
    cons=["Existing subjects/sessions must remain byte-identical.",
          "Changelog lists added/modified files only."],
    outs=["Updated BIDS dataset",
          "`incremental_changelog.md`",
          "Fresh `bids_validation_report.txt`"]))

# --- V. QSM / myelin (T200-T201) ---------------------------------------------------
TASKS.append(_t(
    200, "qsm_pipeline", "QSM Reconstruction Pipeline",
    "Run quantitative susceptibility mapping from multi-echo GRE: phase "
    "unwrapping, background-field removal, dipole inversion, and QSM map "
    "in MNI space.",
    ins=["Multi-echo GRE magnitude + phase NIfTIs (required)",
         "T1w for registration (required)"],
    cons=["Use SEPIA or equivalent open pipeline; document each algorithm "
          "choice (unwrap, BFR, inversion).",
          "Report ROI susceptibility means for deep GM structures."],
    outs=["`qsm_mni.nii.gz`",
          "`roi_susceptibility.csv`",
          "`qsm_qc.png`"])),
TASKS.append(_t(
    201, "myelin_t1t2_ratio", "HCP-Style T1w/T2w Myelin Map Pipeline",
    "Compute HCP-style T1w/T2w-ratio myelin maps: bias-correct T1w and T2w, "
    "rigid-align, ratio on surfaces via FreeSurfer, and parcellate with a "
    "cortical atlas.",
    ins=["T1w + T2w NIfTIs (required)",
         "FreeSurfer recon-all output (required)"],
    cons=["Follow the Glasser & Van Essen T1w/T2w methodology.",
          "Cortical ribbon masking required; report mean ratio per "
          "HCP-MMP parcel."],
    outs=["`t1t2_ratio.midthickness.func.gii` (or NIfTI equivalent)",
          "`parcel_myelin.csv`",
          "`myelin_render.png`"]))

# --- W. MRS (T202) ------------------------------------------------------------------
TASKS.append(_t(
    202, "mrs_fitting_pipeline", "MRS Fitting Pipeline (FSL-MRS)",
    "Process a single-voxel MRS spectrum: format conversion (spec2nii), "
    "eddy-current correction, and basis-set fitting with FSL-MRS, reporting "
    "major metabolite concentrations with CRLBs.",
    ins=["Raw MRS data (Twix/DICOM) + water reference (required)",
         "Basis set and sequence parameters (required)"],
    cons=["Use spec2nii + fsl_mrs; document versions.",
          "Report SNR and FWHM QC metrics; state exclusion criteria."],
    outs=["`metabolite_concentrations.csv` (with CRLB)",
          "`fit_report.pdf` (per-metabolite fit plots)",
          "`mrs_qc.json`"]))

assert len(TASKS) == 54, f"pipeline batch must be 54 tasks, got {len(TASKS)}"
