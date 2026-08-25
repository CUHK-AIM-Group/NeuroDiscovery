from __future__ import annotations

"""Mass batch: data_orchestration tasks T203-T267 (65 tasks)."""

from neurobench_taskkit import body, std_eval

CAT = "data_orchestration"


def _t(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, ev, save_to=std_eval(folder)))


TASKS = []

# --- A. Dataset onboarding (download + stage + validate), T203-T214 ----------
_ONB = [
    ("hcpd_onboarding", "HCP-D", "HCP Development",
     "HCP-D (Lifespan Development) via ConnectomeDB/NDA"),
    ("hcpa_onboarding", "HCP-A", "HCP Aging",
     "HCP-A (Lifespan Aging) via ConnectomeDB/NDA"),
    ("adni3_onboarding", "ADNI-3", "ADNI-3",
     "ADNI-3 via the LONI IDA archive"),
    ("oasis4_onboarding", "OASIS-4", "OASIS-4",
     "OASIS-4 via the OASIS portal"),
    ("abide2_onboarding", "ABIDE-II", "ABIDE-II",
     "ABIDE-II via the FCON_1000/NITRC mirror"),
    ("ixi_onboarding", "IXI", "IXI",
     "IXI via the IXI download portal"),
    ("nkirs_onboarding", "NKI-RS", "NKI Rockland Sample",
     "NKI-RS via the FCON_1000/COINS mirror"),
    ("camcan_onboarding", "CamCAN", "Cam-CAN",
     "Cam-CAN via the Cam-CAN portal"),
    ("gsp_onboarding", "GSP", "Brain Genomics Superstruct Project",
     "GSP via the Harvard Dataverse"),
    ("corr_onboarding", "CoRR", "Consortium for Reliability and "
     "Reproducibility", "CoRR via NITRC"),
    ("adhd200_onboarding", "ADHD-200", "ADHD-200",
     "ADHD-200 via NITRC"),
    ("fcon1000_onboarding", "FCON-1000", "Functional Connectomes 1000",
     "FCON-1000 via NITRC"),
]
for i, (slug, short, name, source) in enumerate(_ONB):
    TASKS.append(_t(
        203 + i, slug, f"{short} Onboarding: Download, Stage, Validate",
        f"Download a subject subset of {name} ({source}), stage it into a "
        "BIDS-compliant directory, and produce a validation + inventory "
        "report.",
        ins=["Subject/session list file (required)",
             "Access credentials or pre-downloaded archive path (required)"],
        cons=["Organize as BIDS (sub-*/ses-*/<modality>/) with "
              "`dataset_description.json` and `participants.tsv`.",
              "Staged dataset must pass `bids-validator` with no errors.",
              "Downloads must be resumable; keep a per-file status log."],
        outs=["BIDS dataset tree",
              "`bids_validation_report.txt`",
              "`data_inventory.csv` (file, size, checksum)",
              "`download_log.json`"]))

# --- B. BIDS conversion variants, T215-T220 -----------------------------------
TASKS.append(_t(
    215, "heudiconv_pipeline", "HeuDiConv DICOM-to-BIDS Conversion",
    "Convert DICOMs to BIDS with HeuDiConv: write the heuristic file, run "
    "conversion for one study, and validate.",
    ins=["DICOM directories (DICOMDIR or tarballs, required)"],
    cons=["Keep the heuristic `.py` in the output directory.",
          "Reproducibility: second run must be idempotent (no dupes)."],
    outs=["BIDS tree", "`heuristic.py`", "`bids_validation_report.txt`",
          "`heudiconv_log.tsv`"])),
TASKS.append(_t(
    216, "bidskit_conversion", "BIDSKIT Conversion Workflow",
    "Convert a dataset to BIDS with BIDSKIT: generate the protocol "
    "translation file from a first pass, edit mappings, and run the final "
    "conversion.",
    ins=["DICOM or sourcedata NIfTI/JSON directories (required)"],
    cons=["`Protocol_Translator.json` must be versioned with the output.",
          "Document any series excluded and why."],
    outs=["BIDS tree", "`Protocol_Translator.json`",
          "`excluded_series.csv` + reason"])),
TASKS.append(_t(
    217, "dcm2bids_multi_session", "dcm2bids Multi-Session Batch",
    "Batch-convert multiple subjects x sessions with dcm2bids driven by a "
    "participant/session manifest, with per-session success tracking.",
    ins=["Manifest CSV (subject, session, dicom_dir) (required)",
         "dcm2bids config JSON (required)"],
    cons=["Per-session logs; failures quarantined, batch continues.",
          "Final summary distinguishes converted / failed / skipped."],
    outs=["BIDS tree", "`batch_summary.csv`", "Per-session logs directory"])),
TASKS.append(_t(
    218, "bids_fieldmap_intendedfor_fix", "BIDS Fieldmap IntendedFor Repair",
    "Audit and repair `IntendedFor` links in a BIDS dataset: every fieldmap "
    "must point to existing BOLD/DWI files, and every BOLD needing SDC must "
    "have an assigned fieldmap.",
    ins=["BIDS dataset directory (required)"],
    cons=["Report broken/dangling links before fixing.",
          "Do not modify raw NIfTI data; JSON sidecars only."],
    outs=["`intendedfor_audit.csv` (before/after)",
          "Fixed JSON sidecars", "`unmatched_bold_report.md`"])),
TASKS.append(_t(
    219, "bids_events_tsv_generation", "BIDS events.tsv Generation",
    "Generate BIDS-compliant `events.tsv` files from raw stimulus/presentation "
    "logs for all task runs in a dataset.",
    ins=["Raw event logs (Presentation/E-Prime/psychopy, required)",
         "Task fMRI BIDS tree (required)"],
    cons=["Columns onset/duration/trial_type mandatory; document any extra "
          "columns in a JSON sidecar.",
          "Onsets aligned to scan start (state the trigger convention)."],
    outs=["`*_events.tsv` per run", "`events_json_sidecars`",
          "`event_count_report.csv`"])),
TASKS.append(_t(
    220, "bids_scans_tsv_rebuild", "BIDS scans.tsv Rebuild",
    "Rebuild `sub-*/ses-*_scans.tsv` files across a BIDS dataset from the "
    "actual files on disk, with acquisition times recovered from JSON "
    "sidecars.",
    ins=["BIDS dataset directory (required)"],
    cons=["Do not invent acquisition times; missing times left as `n/a` "
          "and reported.",
          "Diff against existing scans.tsv must be reported."],
    outs=["Updated `*_scans.tsv` files", "`scans_diff_report.md`"]))

# --- C. Validation / curation, T221-T226 ---------------------------------------
_VC = [
    (221, "bids_validator_ci_report", "BIDS Validator CI-Style Report",
     "Run bids-validator across a dataset and produce a CI-style report "
     "suitable for gating merges: errors/warnings summarized with per-file "
     "detail.",
     ["Exit code semantics documented (0 = pass).",
      "Distinguish errors from warnings; do not silently ignore codes."],
     ["`validation_summary.json`", "`validation_report.md`"]),
    (222, "bids_derivatives_atlas_check", "BIDS-Derivatives Atlas Compliance Check",
     "Check a derivatives directory against BIDS-Derivatives conventions: "
     "required `dataset_description.json` fields (`GeneratedBy`), spatial "
      "reference metadata, and naming rules.",
     ["Use the BIDS-Derivatives spec; cite the rule for each finding.",
      "Validator-plus-custom-checks approach documented."],
     ["`derivatives_compliance.csv`", "`fix_suggestions.md`"]),
    (223, "bids_missing_data_report", "BIDS Missing-Data Report",
     "Build a subject x modality x session completeness matrix for a BIDS "
     "dataset and report missing/expected files per the study protocol.",
     ["Expected protocol defined in a small config file kept with output.",
      "Report both missing files and unexpected extras."],
     ["`completeness_matrix.csv`", "`missing_data_report.md`"]),
    (224, "bids_duplicates_detection", "BIDS Duplicate File Detection",
     "Detect duplicate content in a BIDS dataset: same checksum under "
     "different paths, near-duplicate NIfTIs (same hash after gzip "
     "normalization), and duplicated sessions.",
     ["Normalize .nii.gz before hashing (gunzip stream) so re-compressed "
      "duplicates are caught.",
      "Never delete; produce a quarantine list only."],
     ["`duplicates.csv` (path pairs + hash)",
      "`duplicate_resolution_plan.md`"]),
    (225, "bids_nifti_header_audit", "NIfTI Header Audit",
     "Audit NIfTI headers across a BIDS dataset: TR, voxel sizes, qform/sform "
     "consistency, dim mismatches between runs, and units fields.",
     ["Flag qform != sform cases explicitly.",
      "Per-modality expected values configurable in a small YAML."],
     ["`header_audit.csv`", "`header_anomalies.md`"]),
    (226, "bids_eeg_channels_check", "EEG-BIDS channels.tsv Check",
     "Validate EEG-BIDS `channels.tsv` against the recording: channel count "
     "match, reference electrode declared, and status column present.",
     ["Use the EEG-BIDS spec for required columns.",
      "Cross-check with the actual FIF/EDF channel list."],
     ["`channels_check.csv` per subject", "`channels_fixes.md`"]),
]
for num, slug, title, desc, cons, outs in _VC:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["BIDS dataset directory (required)"],
                    cons=cons, outs=outs))

# --- D. DICOM ops, T227-T231 ------------------------------------------------------
_DC = [
    (227, "dicom_sort_by_series", "DICOM Sort by Series",
     "Sort a flat DICOM dump into Patient/Study/Series directories using "
     "header fields, and emit a series-inventory table.",
     ["Sorting key: StudyInstanceUID/SeriesInstanceUID; keep original "
      "files untouched (copy or link)."],
     ["Sorted directory tree", "`series_inventory.csv`"]),
    (228, "dicom_header_phi_scrub", "DICOM Header PHI Scrub",
     "Scrub protected health information from DICOM headers per a defined "
     "whitelist/blacklist profile, and verify no PHI remains.",
     ["Use a documented profile (e.g. DICOM PS3.15 basic de-identification).",
      "Keep a mapping log linking original to scrubbed IDs, stored "
      "separately."],
     ["Scrubbed DICOM tree", "`phi_scrub_report.json`",
      "`verification_scan.txt`"]),
    (229, "dicom_series_selection_report", "DICOM Series Selection Report",
     "For a study, classify each DICOM series (T1w, T2w, BOLD task/rest, "
     "DWI, fmap, ASL, localizer, other) from header metadata and propose "
     "the BIDS mapping.",
     ["Classification rules kept as a readable config.",
      "Ambiguous series flagged for human review, not guessed."],
     ["`series_classification.csv`", "`proposed_bids_mapping.json`"]),
    (230, "dicom_integrity_check", "DICOM Transfer Integrity Check",
     "Verify a transferred DICOM study: instance counts per series vs. the "
     "sender manifest, readable headers, and no zero-byte files.",
     ["Compare against the sender manifest when provided; otherwise use "
      "internal consistency (SeriesNumber contiguous)."],
     ["`integrity_report.csv`", "`corrupted_instances.txt`"]),
    (231, "dicom_mosaic_detection", "DICOM Mosaic Format Detection",
     "Detect Siemens mosaic-format DWI/fMRI DICOMs in a dump and produce a "
     "conversion-readiness report (which series need mosaic-aware "
     "conversion).",
     ["Detect via ImageType/SIEMENS CSA headers; do not rely on filenames.",
      "Recommend the converter (dcm2niix version) per series."],
     ["`mosaic_series.csv`", "`conversion_readiness.md`"]),
]
for num, slug, title, desc, cons, outs in _DC:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["DICOM directory (required)"], cons=cons, outs=outs))

# --- E. Defacing / anonymization, T232-T234 ---------------------------------------
TASKS.append(_t(
    232, "pydeface_batch", "PyDeface Batch Defacing",
    "Batch-deface all T1w images in a BIDS dataset with pydeface, writing "
    "defaced copies to a new dataset tree (originals untouched) with QC "
    "renders.",
    ins=["BIDS dataset directory (required)"],
    cons=["Defaced dataset gets its own `dataset_description.json` noting "
          "the defacing step.",
          "QC: brain mask must survive (render 3-slice overlays)."],
    outs=["Defaced BIDS tree", "`deface_qc/*.png`", "`deface_log.csv`"])),
TASKS.append(_t(
    233, "mri_deface_fsl_batch", "FSL fsl_deface Batch",
    "Batch-deface T1w/T2w images with FSL `fsl_deface` and compare mask "
    "coverage against pydeface output on a sample.",
    ins=["BIDS dataset directory (required)"],
    cons=["Document FSL version.",
          "Comparison subset: at least 3 subjects if pydeface output is "
          "available."],
    outs=["Defaced NIfTIs", "`deface_comparison.md`", "`deface_log.csv`"])),
TASKS.append(_t(
    234, "anonymization_verification", "Anonymization Verification Report",
    "Verify a dataset is de-identified: scan DICOM/JSON/NIfTI headers and "
     "filenames for residual PHI (names, dates beyond year, MRN patterns).",
    ins=["Dataset directory (required)"],
    cons=["Pattern list (PHI fields + regexes) kept with the output.",
          "Findings graded critical/minor; no auto-fixing."],
    outs=["`phi_scan_report.csv`", "`remediation_plan.md`"]))

# --- F. Packaging / versioning, T235-T240 -------------------------------------------
_PKG = [
    (235, "datalad_dataset_publish", "DataLad Dataset Publish",
     "Publish a BIDS dataset as a DataLad dataset to a sibling (local path "
     "or S3), verifying clone-ability from the published sibling.",
     ["Test the round-trip: fresh `datalad clone` + `datalad get` of a "
      "subset."],
     ["Published sibling URL/path record", "`publish_log.txt`",
      "`clone_verification.md`"]),
    (236, "datalad_subdataset_nesting", "DataLad Subdataset Nesting",
     "Restructure a monolithic dataset into nested DataLad subdatasets "
     "(raw / derivatives / per-site), registering each in the parent.",
     ["Each subdataset gets its own `.gitmodules` entry and README.",
      "History preservation not required; document the cut."],
     ["Nested dataset structure", "`subdataset_map.md`"]),
    (237, "git_annex_s3_sync", "git-annex S3 Sync",
     "Configure a git-annex special remote on S3 (or local-path stand-in) "
     "and sync large files, verifying availability counts.",
     ["`git annex whereis` report before/after sync.",
      "Credentials via environment only; none in repo files."],
     ["`annex_sync_report.txt`", "`whereis_summary.csv`"]),
    (238, "data_release_packager", "Data Release Packager",
     "Package a dataset release: staging area, file manifest with SHA256, "
     "versioned tarball(s), and a RELEASE_NOTES file.",
     ["Tarballs split by modality if > 10 GB; state the rule used.",
      "Manifest covers every packaged file."],
     ["`release/` tree", "`MANIFEST.sha256`", "`RELEASE_NOTES.md`"]),
    (239, "datacite_metadata_generation", "DataCite Metadata Generation",
     "Generate a DataCite-compatible metadata record (JSON + YAML) for a "
     "dataset release: creators, ORCIDs, license, funding, related "
     "identifiers.",
     ["Validate against the DataCite schema (jsonschema).",
      "Pull author ORCIDs from the provided contributors file only."],
     ["`datacite.json`", "`datacite.yaml`", "`schema_validation.txt`"]),
    (240, "release_diff_changelog", "Release Diff Changelog",
     "Diff two dataset releases and produce a human-readable changelog: "
     "added/removed/modified files grouped by subject and modality.",
     ["Modified = same path, different checksum.",
      "Summary counts at top; detail tables below."],
     ["`CHANGELOG_vA_vB.md`", "`diff_detail.csv`"]),
]
for num, slug, title, desc, cons, outs in _PKG:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Dataset directory (required)",
                         "Config/metadata inputs as stated (required)"],
                    cons=cons, outs=outs))

# --- G. Cohort / subset ops, T241-T246 -------------------------------------------------
_TASKS_G = [
    (241, "cohort_subset_selector", "Cohort Subset Selector",
     "Apply inclusion/exclusion criteria (age range, sex, diagnosis, site, "
     "QC pass) to a participants table and emit the selected subject list "
     "with per-criterion attrition counts.",
     ["Criteria given as a small YAML file kept with output.",
      "Attrition reported in PRISMA style (n removed per criterion)."],
     ["`selected_subjects.txt`", "`attrition_report.md`",
      "`cohort_summary.csv`"]),
    (242, "stratified_split_manifest", "Stratified Split Manifest",
     "Create train/validation/test split manifests stratified by site, sex, "
     "and age-bin; deterministic with a fixed seed.",
     ["Stratification report shows group balance per split.",
      "Seed logged; re-run with same seed must reproduce identical splits."],
     ["`split_train.txt`, `split_val.txt`, `split_test.txt`",
      "`stratification_report.csv`", "`split_config.json`"]),
    (243, "participants_tsv_builder", "participants.tsv Builder",
     "Merge multiple phenotypic CSV exports into a single BIDS "
     "`participants.tsv`: join on subject ID, harmonize column names, and "
     "flag unmatched rows on both sides.",
     ["Join rules documented; no row silently dropped.",
      "Column dictionary emitted as participants.json."],
     ["`participants.tsv`", "`participants.json`",
      "`merge_conflicts.csv`"]),
    (244, "session_matching_report", "Cross-Modal Session Matching Report",
     "For each subject, check that required modality sessions (T1w, rest, "
     "DWI) exist within the same session label, and report unmatched "
     "modality-session pairs.",
     ["Matching rules in a config (which modalities required).",
      "Output distinguishes missing session vs. missing modality."],
     ["`session_matching.csv`", "`unmatched_report.md`"]),
    (245, "longitudinal_session_ordering", "Longitudinal Session Ordering",
     "Order longitudinal sessions chronologically from acquisition dates "
     "(DICOM/JSON metadata), relabel `ses-` labels if needed, and verify "
     "consistent intervals.",
     ["Relabeling performed only via copy/rename plan; original kept.",
      "Sessions without dates flagged, not guessed."],
     ["`session_order.csv`", "`relabel_plan.json`",
      "`interval_check.md`"]),
    (246, "exclusion_log_generator", "Exclusion Log Generator",
     "Consolidate QC failure lists from multiple pipeline stages into one "
     "exclusion log with stage, reason, and final-inclusion flag per "
     "subject.",
     ["Each stage's criteria quoted in the log header.",
      "Final flag = AND of all stages; conflicts flagged."],
     ["`exclusion_log.csv`", "`final_cohort.txt`",
      "`exclusion_summary.md`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_G:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Participants/phenotypic table(s) (required)",
                         "Criteria or config file (required)"],
                    cons=cons, outs=outs))

# --- H. Storage / transfer, T247-T251 ----------------------------------------------------
_TASKS_H = [
    (247, "rclone_s3_mirror", "Rclone S3 Mirror",
     "Mirror a dataset directory to S3-compatible storage with rclone: "
     "configure the remote, sync with checksum verification, and produce a "
     "transfer report.",
     ["`--checksum` verification; dry-run output saved before the real "
      "sync.",
      "No credentials in command logs (env/config redacted)."],
     ["`rclone_config_redacted.txt`", "`sync_report.txt`",
      "`dry_run_diff.txt`"]),
    (248, "rsync_hpc_transfer_report", "HPC rsync Transfer + Report",
     "Transfer a dataset to/from an HPC cluster with rsync: resumable, "
     "partial-dir friendly, with a post-transfer verification pass.",
     ["Use `rsync -a --partial --info=progress2` or equivalent.",
      "Verification: file count + byte size + spot checksums (>= 1% of "
      "files)."],
     ["`transfer_log.txt`", "`verification_report.md`"]),
    (249, "disk_usage_treemap_report", "Dataset Disk Usage Report",
     "Profile disk usage of a dataset: per-directory sizes, top-20 largest "
     "files, format breakdown (DICOM/NIfTI/derivatives), and a text or HTML "
     "treemap.",
     ["Read-only; never modify the dataset.",
      "Identify redundant candidates (e.g. uncompressed duplicates) as "
      "suggestions only."],
     ["`disk_usage_report.html`", "`largest_files.csv`",
      "`cleanup_suggestions.md`"]),
    (250, "nii_compress_batch", "Batch nii -> nii.gz Compression",
     "Compress uncompressed NIfTIs in a dataset to .nii.gz, updating BIDS "
     "references (scans.tsv, IntendedFor) and verifying data integrity.",
     ["Integrity: voxel data identical after round-trip (verify via "
      "checksum of decompressed stream).",
      "Update every referencing sidecar."],
     ["Compressed files", "`compression_log.csv`",
      "`reference_updates.md`"]),
    (251, "md5_manifest_full_dataset", "Full-Dataset Checksum Manifest",
     "Generate a SHA256 (or MD5) manifest for every file in a dataset and a "
     "verification script that can re-check the manifest later.",
     ["Manifest sorted by path; one line per file.",
      "Verification script standalone (single python/bash file)."],
     ["`MANIFEST.sha256`", "`verify_manifest.py` (or .sh)",
      "`generation_log.txt`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_H:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Dataset directory (required)",
                         "Remote/credentials where applicable (required)"],
                    cons=cons, outs=outs))

# --- I. QC inventory, T252-T257 ------------------------------------------------------------
_TASKS_I = [
    (252, "modality_coverage_matrix", "Modality Coverage Matrix",
     "Build a subject x modality coverage matrix (T1w/T2w/BOLD/DWI/ASL/PET/ "
     "EEG/MEG) for a dataset, with counts and a heatmap render.",
     ["Modalities detected from BIDS suffixes, not folder names.",
      "Heatmap exported as PNG."],
     ["`modality_matrix.csv`", "`modality_heatmap.png`"]),
    (253, "file_count_consistency", "File Count Consistency Check",
     "Check internal consistency: per-subject file counts by modality "
     "against the dataset median; flag outliers (missing runs, doubled "
     "runs).",
     ["Outlier rule: |count - median| > 0 flagged with detail.",
      "No modifications; report only."],
     ["`file_count_report.csv`", "`outlier_subjects.md`"]),
    (254, "derivative_completeness_report", "Derivative Completeness Report",
     "Verify fMRIPrep/XCP-D output completeness per subject: every expected "
     "derivative file present (per a manifest template), missing ones "
     "listed.",
     ["Expected-file template kept as config.",
      "Report coverage percent per subject and overall."],
     ["`derivative_completeness.csv`", "`missing_derivatives.md`"]),
    (255, "failed_conversion_quarantine", "Failed Conversion Quarantine",
     "Collect failed conversions from pipeline logs, move (copy + plan) the "
     "affected sourcedata into a quarantine area, and produce a retry list.",
     ["Copy, never move originals; emit a quarantine plan instead of "
      "acting if destructive.",
      "Retry list grouped by failure reason."],
     ["`quarantine/` directory or plan", "`retry_list.csv`",
      "`failure_reasons.md`"]),
    (256, "rerun_list_generator", "Re-run List Generator",
     "From QC and completeness reports, generate the definitive list of "
     "subject/session/modality units that must be re-run, formatted for the "
     "target pipeline (fMRIPrep participant labels, SLURM array indices).",
     ["Output formats: participant-label list + SLURM array spec.",
      "Deduplicated and sorted."],
     ["`rerun_participants.txt`", "`rerun_array_spec.txt`",
      "`rerun_rationale.csv`"]),
    (257, "inventory_dashboard_html", "Dataset Inventory Dashboard",
     "Generate a self-contained HTML dashboard summarizing a dataset: "
     "counts per modality/site/scanner, acquisition-date timeline, and "
     "storage footprint.",
     ["Single HTML file, no external JS/CSS dependencies.",
      "All numbers reproducible from the dataset alone."],
     ["`inventory_dashboard.html`", "`inventory_stats.json`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_I:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["BIDS dataset directory (required)"],
                    cons=cons, outs=outs))

# --- J. Metadata, T258-T262 ---------------------------------------------------------------------
_TASKS_J = [
    (258, "dataset_description_writer", "dataset_description.json Writer",
     "Author or repair `dataset_description.json` files for a dataset and "
     "its derivatives: correct BIDSVersion, Name, DatasetType, GeneratedBy "
     "blocks.",
     ["Derivatives must include GeneratedBy with tool + version.",
      "Validate with bids-validator afterwards."],
     ["Updated JSON files", "`description_diff.md`"]),
    (259, "dataset_card_writer", "Dataset Card Writer",
     "Write a dataset card (README) for a neuroimaging dataset: cohort "
     "description, acquisition parameters, provenance, license, ethics, "
     "citation.",
     ["Follow the dataset-cards convention (sections in fixed order).",
      "Every acquisition number traced to the data (no invented values)."],
     ["`README.md` dataset card", "`sources_checked.md`"]),
    (260, "license_compliance_check", "License Compliance Check",
     "Audit a dataset + derivatives for license compliance: license files "
     "present, redistribution terms compatible with planned release, and "
     "attribution strings collected.",
     ["Check each source component (atlases, templates) license too.",
      "Report conflicts as blockers, not suggestions."],
     ["`license_audit.md`", "`attribution_strings.txt`"]),
    (261, "citation_cff_generator", "CITATION.cff Generator",
     "Generate a valid CITATION.cff for the dataset/codebase and verify it "
     "with cffconvert.",
     ["Validate with `cffconvert --validate`.",
      "Authors/ORCIDs from the provided contributors file only."],
     ["`CITATION.cff`", "`cffconvert_validation.txt`"]),
    (262, "participants_summary_stats", "Participants Summary Statistics",
     "Compute cohort summary statistics from participants.tsv: N, age "
     "mean/range per group, sex balance, site distribution, formatted as a "
     "paper-ready demographics table.",
     ["Table in Markdown + CSV; counts must reconcile with the TSV.",
      "Rounding rules stated (1 decimal for means)."],
     ["`demographics_table.md`", "`demographics.csv`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_J:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["Dataset directory (required)",
                         "Contributors/criteria files where applicable"],
                    cons=cons, outs=outs))

# --- K. Multi-modal linking, T263-T267 ------------------------------------------------------------------
_TASKS_K = [
    (263, "pet_mri_session_linking", "PET-MRI Session Linking",
     "Link PET scans to their anatomically-matching MRI sessions per "
     "subject: same session preferred, nearest-date fallback with an "
     "interval threshold.",
     ["Threshold (days) configurable; fallback links explicitly flagged.",
      "Unlinkable PET scans reported, not silently dropped."],
     ["`pet_mri_links.csv`", "`linking_report.md`"]),
    (264, "eeg_mri_coreg_manifest", "EEG-MRI Coregistration Manifest",
     "Build a manifest pairing each EEG recording with the subject's T1w "
     "for coregistration, including fiducial availability status.",
     ["Fiducial status from the EEG sidecar files.",
      "Missing-fiducial recordings listed separately."],
     ["`eeg_mri_manifest.csv`", "`fiducial_status.md`"]),
    (265, "asl_m0_pairing_check", "ASL M0 Pairing Check",
     "Verify every ASL run has an associated M0 scan (separate file or "
     "embedded), per the ASL-BIDS specification.",
     ["Follow the ASL-BIDS M0 pairing rules.",
      "Report pairing type per run (separate/included/absent)."],
     ["`asl_m0_pairing.csv`", "`asl_bids_findings.md`"]),
    (266, "fieldmap_assignment_audit", "Fieldmap Assignment Audit",
     "Audit fieldmap-to-scan assignment: correct `IntendedFor` coverage, "
     "PE-direction pairing sanity (AP/PA), and per-scan SDC readiness.",
     ["PE direction read from JSON sidecars.",
      "Scans lacking SDC options listed explicitly."],
     ["`fmap_assignment.csv`", "`sdc_readiness_report.md`"]),
    (267, "meg_emptyroom_linking", "MEG Empty-Room Linking",
     "Link each MEG session to its empty-room noise recording by date "
     "proximity and BIDS `AssociatedEmptyRoom` fields; repair missing "
     "links.",
     ["Proximity window configurable (default same day).",
      "Repairs limited to JSON sidecar fields."],
     ["`emptyroom_links.csv`", "`meg_linking_report.md`"]),
]
for num, slug, title, desc, cons, outs in _TASKS_K:
    TASKS.append(_t(num, slug, title, desc,
                    ins=["BIDS dataset directory (required)"],
                    cons=cons, outs=outs))

assert len(TASKS) == 65, f"data_orch batch must be 65 tasks, got {len(TASKS)}"
