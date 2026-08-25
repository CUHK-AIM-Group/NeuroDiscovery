from __future__ import annotations

"""NeuroBench task expansion generator.

Creates new task folders (T<num>_<slug>/task.md) under neurobench/ and
registers them in neurobench/task_atlas.json. Idempotent: existing folders
are skipped and atlas entries are not duplicated.

Current content: sample batch T121-T138 plus the approved mass batch
T139-T500 (category modules neurobench_mass_*.py). Idempotent: re-running
skips folders that already exist.

Distribution plan (approved): tool_use 80 / pipeline 75 / data_orch 75 /
dev_env 70 / research 70 / model_train 70 / cross_model 60 = 500.
"""

import argparse
import json
from pathlib import Path

from neurobench_mass_tool_use import TASKS as MASS_TOOL_USE
from neurobench_mass_pipeline import TASKS as MASS_PIPELINE
from neurobench_mass_data_orch import TASKS as MASS_DATA_ORCH
from neurobench_mass_dev_env import TASKS as MASS_DEV_ENV
from neurobench_mass_research import TASKS as MASS_RESEARCH
from neurobench_mass_model_train import TASKS as MASS_MODEL_TRAIN
from neurobench_mass_cross_model import TASKS as MASS_CROSS_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "neurobench"
ATLAS_PATH = BENCH_DIR / "task_atlas.json"

# ---------------------------------------------------------------------------
# Sample task definitions: (number, slug, category, title, body)
# body follows the established task.md conventions (Input Requirement /
# Constraints / Expected Output / Evaluation, output dir benchmark_results/).
# ---------------------------------------------------------------------------

TASKS = [
    # ------------------------------------------------------------------ tool_use
    (121, "ants_registration", "tool_use", "ANTs SyN Registration to MNI152", """
## Task Description

Register a subject T1w image to the MNI152NLin2009cAsym template using ANTs
`antsRegistrationSyN.sh` (affine + SyN deformable), producing the warped brain
and the forward/inverse transform chain.

## Input Requirement

Required input(s):

- Subject T1w NIfTI file (required)
- MNI152NLin2009cAsym template path (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use ANTs `antsRegistrationSyN.sh` with `-d 3 -t s` (or the equivalent
  `antsRegistration` call with the same stages).
- Save all generated artifacts to:
  - benchmark_results/T121_ants_registration/
- Report the ANTs version (`antsRegistration --version`).

## Expected Output

Expected output artifact(s):

- `warped.nii.gz` (subject T1w in MNI space)
- `transform_0GenericAffine.mat`, `transform_1Warp.nii.gz`, `transform_1InverseWarp.nii.gz`
- Overlay QC image (template edges over warped brain, PNG)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (122, "mrtrix_tckgen_tractography", "tool_use", "MRtrix tckgen Streamline Tractography (stage split of T59)", """
## Task Description

Stage split of T59_dwi_mrtrix_tractography: run ONLY the tractography step.
Given a precomputed FOD image and a seed mask, generate a whole-brain
streamline tractogram with MRtrix3 `tckgen` using the iFOD2 algorithm.

## Input Requirement

Required input(s):

- FOD image (`wmfod.mif`, required)
- Seed mask (gray-matter/white-matter interface mask, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `tckgen -algorithm iFOD2` with 10M streamlines (`-select 10000000`).
- ACT is optional (`-act 5tt.mif`) and must be documented if used.
- Save all generated artifacts to:
  - benchmark_results/T122_mrtrix_tckgen_tractography/

## Expected Output

Expected output artifact(s):

- `tractogram_10M.tck`
- `tckgen_log.txt` (full command log with parameters)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    # ------------------------------------------------------ pipeline_execution
    (123, "qsiprep_qsirecon_pipeline", "pipeline_execution", "QSIPrep + QSIRecon Diffusion Pipeline", """
## Task Description

Run an end-to-end diffusion workflow for one subject: preprocess raw DWI with
QSIPrep, then reconstruct with QSIRecon (DSI Studio GQI + deterministic
tracking), producing a connectome-ready tractogram and scalar maps.

## Input Requirement

Required input(s):

- BIDS dataset directory with DWI + reverse phase-encoding data (required)
- FreeSurfer license file (required by QSIPrep)

If any required input is missing, return:

- Missing required input

## Constraints

- QSIPrep: default preprocessing with eddy motion/eddy-current correction.
- QSIRecon: `--recon-spec dsi_studio_gqi` tractography with 1M streamlines.
- Container execution (Docker or Singularity) with versions pinned in the log.
- Save all generated artifacts to:
  - benchmark_results/T123_qsiprep_qsirecon_pipeline/

## Expected Output

Expected output artifact(s):

- QSIPrep derivatives tree (preprocessed DWI, bval/bvec, QC report HTML)
- QSIRecon outputs (`streamlines.tck`, FA/MD/RD scalar maps)
- Pipeline runtime log (per-stage wall time)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (124, "mne_eeg_preproc_pipeline", "pipeline_execution", "MNE-Python EEG Preprocessing Chain (stage split of T70)", """
## Task Description

Stage split of T70_eeg_full_pipeline: preprocessing only. Load raw EEG, apply
band-pass filtering, run ICA-based artifact removal, epoch around events, and
compute the evoked response. No source localization or statistics.

## Input Requirement

Required input(s):

- Raw EEG recording (FIF/EDF/BrainVision, required)
- Event annotations or stim channel (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use MNE-Python throughout.
- Band-pass 1-40 Hz, notch at line frequency (50/60 Hz documented).
- ICA with the extended infomax method; document which components were
  rejected and why (EOG/ECG correlation criteria).
- Save all generated artifacts to:
  - benchmark_results/T124_mne_eeg_preproc_pipeline/

## Expected Output

Expected output artifact(s):

- `cleaned_epo.fif` (epoched, artifact-corrected data)
- `evoked-ave.fif` plus a butterfly-plot PNG
- `ica_report.html` (MNE Report with topomaps of rejected components)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    # ------------------------------------------------------- data_orchestration
    (125, "oasis_bids_staging", "data_orchestration", "OASIS-3 Download and BIDS Staging", """
## Task Description

Download an OASIS-3 imaging subset (T1w + T2w for a given subject list) and
stage it into a BIDS-compliant dataset directory.

## Input Requirement

Required input(s):

- OASIS-3 subject/session list file (required)
- OASIS-3 access credentials or pre-downloaded archive path (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Organize output as `sub-<label>/ses-<label>/anat/` with BIDS filenames.
- Generate `dataset_description.json` and `participants.tsv`.
- The staged dataset must pass `bids-validator` with no errors.
- Save all generated artifacts to:
  - benchmark_results/T125_oasis_bids_staging/

## Expected Output

Expected output artifact(s):

- BIDS dataset directory tree
- `bids_validation_report.txt` (validator output)
- `data_inventory.csv` (one row per staged file with size + checksum)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (126, "abide_download_organize", "data_orchestration", "ABIDE I Fetch and Per-Site Organization", """
## Task Description

Fetch ABIDE I preprocessed resting-state fMRI (CPAC pipeline, filt_global)
for a given list of acquisition sites, and organize the derivatives into a
per-site directory layout with a unified subject manifest.

## Input Requirement

Required input(s):

- Site list file (e.g. `sites.txt` with ABIDE site IDs, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `nilearn.datasets.fetch_abide_pcp` (or direct HTTP mirrors) with
  `pipeline='cpac'`, `band_pass_filtering=True`, `global_signal_regression=True`.
- Organize as `derivatives/site-<ID>/sub-<label>_ses-1_task-rest_...nii.gz`.
- Phenotypic data merged into `participants.tsv` (subject, site, dx, age, sex).
- Save all generated artifacts to:
  - benchmark_results/T126_abide_download_organize/

## Expected Output

Expected output artifact(s):

- Organized derivatives tree per site
- `participants.tsv` + `download_log.json` (per-file status, resume-safe)
- `manifest_summary.md` (counts per site, failed downloads listed)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (127, "openneuro_dataset_fetch", "data_orchestration", "OpenNeuro Dataset Fetch and Integrity Check", """
## Task Description

Fetch one OpenNeuro dataset (default `ds000114`, configurable) via DataLad or
the S3 mirror, materialize the required files, and verify integrity against
the dataset manifest.

## Input Requirement

Required input(s):

- OpenNeuro dataset accession (e.g. `ds000114`, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `datalad install` + `datalad get`, or `aws s3 sync --no-sign-request`.
- Do not download more than the requested modalities (document what was
  materialized vs. left as symlinks/annexed).
- Verify file count and, where available, checksums.
- Save all generated artifacts to:
  - benchmark_results/T127_openneuro_dataset_fetch/

## Expected Output

Expected output artifact(s):

- Local dataset clone with materialized files
- `integrity_report.json` (expected vs. actual file counts, failures)
- `dataset_card.md` (short summary: subjects, tasks, modalities, license)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    # ----------------------------------------------------------- dev_environment
    (128, "docker_neuroimage_build", "dev_environment", "Docker Image for FSL/AFNI/ANTs Toolchain", """
## Task Description

Author a Dockerfile for a reproducible neuroimaging analysis image containing
FSL, AFNI, and ANTs, build it, and smoke-test each tool inside the container.

## Input Requirement

- No interactive input.

## Constraints

- Pin base image and tool versions (no `latest` tags).
- Keep the image as small as practical (multi-stage build or neurodocker
  generated recipe preferred; document the choice).
- Smoke tests must run non-interactively: `flirt -version`, `afni -ver`,
  `antsRegistration --version` inside the built image.
- Save all generated artifacts to:
  - benchmark_results/T128_docker_neuroimage_build/

## Expected Output

Expected output artifact(s):

- `Dockerfile` (and neurodocker command if used)
- `build_log.txt` + final image size and tag
- `smoke_test.json` (per-tool: pass/fail + version string)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (129, "dvc_data_versioning", "dev_environment", "DVC Data Versioning for Derivatives", """
## Task Description

Initialize DVC in an existing analysis repository, place a derivatives
directory under version control with a configured remote, and demonstrate a
checkout/repro round-trip.

## Input Requirement

Required input(s):

- Path to the repository containing the derivatives directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- `dvc init`, `dvc add derivatives/`, configure a remote (local-path remote
  is acceptable for the benchmark), `dvc push`.
- Git-track the `.dvc` files and `.gitignore` changes; data itself must NOT
  be committed to git.
- Demonstrate `dvc pull` restoring the directory after a clean checkout.
- Save all generated artifacts to:
  - benchmark_results/T129_dvc_data_versioning/

## Expected Output

Expected output artifact(s):

- `derivatives.dvc` and updated `.gitignore`
- `dvc_remote_config.txt` + push/pull logs
- `roundtrip_report.json` (file count + hash before vs. after pull)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (130, "slurm_array_job", "dev_environment", "SLURM Array Job for Batch fMRIPrep", """
## Task Description

Write a SLURM array job that runs fMRIPrep over a list of BIDS subjects, one
array task per subject, with per-subject logs and a final aggregation step.

## Input Requirement

Required input(s):

- BIDS dataset path + subject list file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Array size derived from the subject list (`#SBATCH --array=...`).
- Per-subject resource requests (cpus/mem/time) justified in comments.
- Stdout/stderr captured per subject (`logs/sub-<ID>_%j.out`).
- A `--dry-run` mode that prints the commands without submitting must be
  supported for evaluation on machines without SLURM.
- Save all generated artifacts to:
  - benchmark_results/T130_slurm_array_job/

## Expected Output

Expected output artifact(s):

- `run_fmriprep_array.slurm`
- `aggregate_qc.slurm` (or equivalent post-step) that collects per-subject
  success/failure into `qc_summary.csv`
- `dry_run_output.txt` from the dry-run mode

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    # --------------------------------------------------------- research_tooling
    (131, "literature_search_dmn_aging", "research_tooling", "Academic Search: Default Mode Network and Aging", """
## Task Description

Search for the most recent papers related to **"default mode network aging"**
from multiple academic platforms:

- **arXiv**
- **PubMed**
- **Semantic Scholar**

### Constraints

- **Time Range:** Last 180 days
- **Results per Platform:** 20 papers each (60 total minimum)
- **Sorting:** Newest first
- **Deduplication:** Cross-platform duplicates removed by DOI/title match
- **No Input:** This test case requires no command-line input
- **Robustness:** The workflow should tolerate partial platform failures,
  access restrictions, and rate limits without failing the whole run

### Expected Output

Results should be saved to `benchmark_results/T131_literature_search_dmn_aging/`
as a JSON file with the following structure:

```json
{
  "metadata": {
    "query": "default mode network aging",
    "timestamp": "ISO-8601 format",
    "total_papers": 60,
    "duplicates_removed": 0
  },
  "arxiv": [
    {
      "title": "string",
      "authors": ["string"],
      "published": "ISO-8601 format",
      "url": "string",
      "abstract": "string",
      "doi": "string or null"
    }
  ],
  "pubmed": [],
  "semantic_scholar": []
}
```

## Evaluation

- This test case is manually evaluated.
"""),
    (132, "citation_network_mapping", "research_tooling", "Citation Network Mapping from a Seed Paper", """
## Task Description

Starting from a seed paper, build its one-hop citation network (references +
citing papers) using open scholarly APIs, and produce a structured graph plus
a short landscape summary.

## Input Requirement

Required input(s):

- Seed paper identifier (DOI or Semantic Scholar / OpenAlex ID, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use open APIs only (OpenAlex, Semantic Scholar, Crossref); respect rate
  limits.
- Cap the network at 200 nodes; selection criterion must be documented
  (e.g. top-cited citing works).
- Save all generated artifacts to:
  - benchmark_results/T132_citation_network_mapping/

## Expected Output

Expected output artifact(s):

- `citation_graph.json` (`nodes`: id/title/year/venue/citation_count,
  `edges`: source/target/type)
- `landscape_summary.md` (top venues, publication-year histogram in text
  form, 5-sentence narrative of the research landscape)
- Optional `citation_graph.png` visualization

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    (133, "prisma_screening_workflow", "research_tooling", "PRISMA-Style Systematic Screening", """
## Task Description

Execute a PRISMA-style screening workflow for a given review query: retrieve
candidate papers, apply staged inclusion/exclusion criteria, and report the
funnel counts required for a PRISMA flow diagram.

## Input Requirement

Required input(s):

- Review query string and inclusion/exclusion criteria file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Stage counts must be reported: identified -> deduplicated -> title/abstract
  screened -> full-text assessed -> included.
- Every exclusion at full-text stage needs a one-line reason.
- LLM-assisted screening is allowed but criteria must be applied verbatim
  from the criteria file.
- Save all generated artifacts to:
  - benchmark_results/T133_prisma_screening_workflow/

## Expected Output

Expected output artifact(s):

- `screened_papers.csv` (one row per paper: stage reached, decision, reason)
- `prisma_flow.json` (counts per stage, ready to plot)
- `included_summary.md` (table of included papers with key fields)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
"""),
    # ------------------------------------------------------------ model_training
    (134, "gat_train_eval", "model_training", "GAT Training and Evaluation", """
## Task Description

Train and evaluate a Graph Attention Network (GAT) baseline on preprocessed
functional connectivity (FC) graphs for two settings: HCP age regression and
ABIDE diagnosis classification. ROI-as-node graphs, attention coefficients
exported for interpretability.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas
  (e.g. `schaefer_200_7net`, `aal_116`)
- Subject list file (`ready_subjects.txt`)
- Labels CSV
  - HCP age: `data/hcp_age_labels.csv` (continuous age in years)
  - ABIDE dx: `data/abide_dx_labels.csv` (binary: ASD vs control)
- Atlas name and ROI count (must match FC dimension)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py --model gat` for both settings.
- 5-fold split with deterministic `--seed`; report fold 0..4 test metrics.
- 2-layer GAT, 4 attention heads, hidden dim 64 (document any deviation).
- Save artefacts under `models/benchmark_results/T134_gat/<setting>/`.
- Save checkpoints under `models/checkpoints/gat/<atlas>/fold{k}.pt`.

## Expected Output

- Per-fold test metrics CSV (regression: MAE / RMSE / R^2; classification:
  accuracy / AUC / F1)
- Aggregated 5-fold mean +/- std
- One attention-weight visualization (ROI x ROI attention matrix) per setting
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- HCP age MAE within a reasonable baseline range (<= ~6.0 years).
- ABIDE binary AUC >= 0.62 on at least one atlas.
- Manually scored for reproducibility (seed + atlas + fold all logged).
"""),
    (135, "gcn_train_eval", "model_training", "GCN Training and Evaluation", """
## Task Description

Train and evaluate a Graph Convolutional Network (GCN, Kipf & Welling)
baseline on preprocessed functional connectivity graphs for two settings:
HCP age regression and ABIDE diagnosis classification.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas
  (e.g. `schaefer_200_7net`, `aal_116`)
- Subject list file (`ready_subjects.txt`)
- Labels CSV
  - HCP age: `data/hcp_age_labels.csv` (continuous age in years)
  - ABIDE dx: `data/abide_dx_labels.csv` (binary: ASD vs control)
- Atlas name and ROI count (must match FC dimension)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py --model gcn` for both settings.
- 5-fold split with deterministic `--seed`; report fold 0..4 test metrics.
- 2-layer GCN, hidden dim 64, thresholded FC adjacency (document the
  threshold, e.g. top 10% edges).
- Save artefacts under `models/benchmark_results/T135_gcn/<setting>/`.
- Save checkpoints under `models/checkpoints/gcn/<atlas>/fold{k}.pt`.

## Expected Output

- Per-fold test metrics CSV (regression: MAE / RMSE / R^2; classification:
  accuracy / AUC / F1)
- Aggregated 5-fold mean +/- std
- Adjacency-threshold sensitivity note (one extra threshold re-run)
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- HCP age MAE within a reasonable baseline range (<= ~6.0 years).
- ABIDE binary AUC >= 0.62 on at least one atlas.
- Manually scored for reproducibility (seed + atlas + fold all logged).
"""),
    # ---------------------------------------------------- cross_model_evaluation
    (136, "combat_harmonization_protocol", "cross_model_evaluation", "ComBat Harmonization Across Sites", """
## Task Description

Apply ComBat batch-effect harmonization to multi-site FC feature matrices
(ABIDE sites), then quantify how site effects are reduced and how much
biological signal (diagnosis, age) is preserved.

## Input Requirement

Required input(s):

- Per-subject FC feature matrices (NPZ) with site labels (required)
- Covariates CSV (site, diagnosis, age, sex) (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use neuroComBat (Python port) or `neuroHarmonize`; protect diagnosis and
  age as biological covariates.
- Harmonization fitted on the training split only, then applied to test
  (no leakage).
- Save artefacts under `models/benchmark_results/T136_combat/`.

## Expected Output

- Pre/post-harmonization site-effect quantification (e.g. per-feature
  variance explained by site, or kBET-style metric) as CSV
- A downstream ABIDE dx model (logistic regression) evaluated pre vs. post
  harmonization: accuracy / AUC / F1 comparison table
- `harmonization_report.md` with method, parameters, and interpretation
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- Site-effect metric must decrease post-harmonization.
- Downstream diagnosis performance must not collapse (AUC drop <= 0.05).
- Leakage-free protocol must be evident from the logs.
"""),
    (137, "leave_one_site_out", "cross_model_evaluation", "Leave-One-Site-Out Generalization", """
## Task Description

Run a leave-one-site-out (LOSO) evaluation on ABIDE: train a model on all
sites but one, test on the held-out site, rotate over all sites, and produce
a per-site generalization matrix.

## Input Requirement

Required input(s):

- Per-subject FC matrices (NPZ) with site labels (required)
- Labels CSV (`data/abide_dx_labels.csv`) (required)
- Choice of model (default `gcn`, configurable)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py` with a LOSO split mode (or an equivalent
  documented script); deterministic seed shared with T101/T134/T135.
- Sites with fewer than 20 subjects after filtering are excluded and logged.
- Save artefacts under `models/benchmark_results/T137_loso/<model>/`.

## Expected Output

- Per-site test metrics CSV (accuracy / AUC / F1 per held-out site)
- Site x site generalization summary (mean/std across held-out sites)
- Comparison table vs. random-split performance from the same model
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- All eligible sites covered (no silently skipped sites).
- LOSO performance must be reported honestly even when below random split.
- Discussion of site heterogeneity (sample size, scanner) included.
"""),
    (138, "atlas_ranking_consistency", "cross_model_evaluation", "Cross-Atlas Model Ranking Consistency", """
## Task Description

Quantify whether model rankings are stable across parcellations: evaluate
three models (braingnn, gcn, roi_mlp_baseline) on three atlases
(`aal_116`, `schaefer_200_7net`, `schaefer_400_7net`) for ABIDE
classification, then compute ranking-consistency statistics across atlases.

## Input Requirement

Required input(s):

- Per-atlas FC matrices (NPZ) for the three atlases (required)
- Subject list and labels CSV (required)

If any required input is missing, return:

- Missing required input

## Constraints

- 5-fold deterministic split shared with T101/T134/T135.
- Same hyperparameters per model across atlases (only input dimension
  changes).
- Save artefacts under `models/benchmark_results/T138_atlas_consistency/`.

## Expected Output

- Model x atlas performance matrix CSV (mean AUC across folds)
- Kendall's W (or Spearman rank correlation per atlas pair) for model
  rankings, with a short interpretation
- Heatmap PNG of the model x atlas matrix
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- All 9 (model, atlas) cells completed.
- Ranking statistics computed and interpreted correctly.
- Manually scored for completeness and correctness of the protocol.
"""),
]

# Mass batch T139-T500 (approved distribution), defined in neurobench_mass_*.
TASKS = (
    TASKS
    + MASS_TOOL_USE
    + MASS_PIPELINE
    + MASS_DATA_ORCH
    + MASS_DEV_ENV
    + MASS_RESEARCH
    + MASS_MODEL_TRAIN
    + MASS_CROSS_MODEL
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing.")
    args = parser.parse_args()

    atlas = json.loads(ATLAS_PATH.read_text(encoding="utf-8"))
    categories = atlas["operational_layer"]["categories"]

    created, skipped = [], []
    for num, slug, category, title, body in TASKS:
        folder = f"T{num}_{slug}"
        task_dir = BENCH_DIR / folder
        md = f"# {folder}: {title}\n{body.lstrip()}"
        if task_dir.exists():
            skipped.append(folder)
        else:
            if not args.dry_run:
                task_dir.mkdir(parents=True)
                (task_dir / "task.md").write_text(md, encoding="utf-8")
            created.append(folder)
        tasks = categories[category]["tasks"]
        if folder not in tasks and not args.dry_run:
            tasks.append(folder)

    if not args.dry_run:
        ATLAS_PATH.write_text(
            json.dumps(atlas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"created: {len(created)}  skipped(existing): {len(skipped)}")
    for f in created:
        print("  +", f)
    for f in skipped:
        print("  =", f)


if __name__ == "__main__":
    main()
