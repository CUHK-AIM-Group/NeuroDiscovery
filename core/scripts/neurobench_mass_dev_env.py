from __future__ import annotations

"""Mass batch: dev_environment tasks T268-T330 (63 tasks)."""

from neurobench_taskkit import body, std_eval

CAT = "dev_environment"


def _t(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, ev, save_to=std_eval(folder)))


TASKS = []

# --- A. Conda environments, T268-T273 ------------------------------------------
_ENV = [
    (268, "conda_fmriprep_support_env", "Conda Env: fMRIPrep Support Tools",
     "Create a fresh conda env with the support tooling needed around "
     "fMRIPrep runs (bids-validator, pybids, pandas, niworkflows client "
     "libs) and smoke-test imports.",
     ["Pinned versions via explicit spec file.", "Smoke test: python import "
      "check + validator version."],
     ["`environment.yml`", "`smoke_test.json`"]),
    (269, "conda_mrtrix_build_env", "Conda Env: MRtrix3 Build Environment",
     "Create an environment capable of building MRtrix3 from source "
     "(compiler toolchain, Eigen, Qt) and document the build.",
     ["Document compiler versions.", "If build is skipped, provide the "
      "binary-install alternative and say why."],
     ["`environment.yml`", "`build_log.txt`"]),
    (270, "conda_r_neuro_env", "Conda Env: R Neuro Statistics",
     "Create an R environment with neuroimaging statistics packages "
     "(tidyverse, lme4, emmeans, neurobase/oro.nifti) and verify loading.",
     ["R version pinned.", "Library load test captured."],
     ["`environment.yml`", "`r_package_versions.csv`",
      "`load_test.txt`"]),
    (271, "conda_neuro_base_py", "Conda Env: Python Neuro Base",
     "Create a pinned base environment for neuroimaging analysis: numpy, "
     "scipy, nibabel, nilearn, dipy, mne, pandas, matplotlib.",
     ["Exact pins (==) for all packages.",
      "Import test + version dump to JSON."],
     ["`environment.yml`", "`import_test.json`"]),
    (272, "conda_lock_export", "Conda-Lock Reproducible Lockfile",
     "Generate a fully reproducible conda-lock lockfile from an existing "
     "environment YAML and demonstrate a clean-room reinstall.",
     ["Use conda-lock; lockfile per platform if applicable.",
      "Clean-room test: new env from lockfile only."],
     ["`conda-lock.yml`", "`reinstall_log.txt`"]),
    (273, "conda_env_conflict_diagnosis", "Conda Conflict Diagnosis",
     "Given a failing environment specification, diagnose the dependency "
     "conflict, explain the cause, and produce a working minimal fix.",
     ["Explain the conflict in plain language (which packages clash on "
      "which constraint).",
      "Minimal change principle: smallest possible edit set."],
     ["`conflict_analysis.md`", "`environment_fixed.yml`",
      "`verification_log.txt`"]),
]
for num, slug, title, desc, cons, outs in _ENV:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- B. Containers, T274-T279 -----------------------------------------------------
_CTR = [
    (274, "singularity_pull_convert", "Singularity/Apptainer Image Conversion",
     "Pull a BIDS-app Docker image and convert it to a Singularity/Apptainer "
     "SIF for HPC use, then verify execution.",
     ["Pin the Docker tag by digest.",
      "Run `--version` inside the SIF as verification."],
     ["`image.sif` reference (or pull command record)",
      "`conversion_log.txt`", "`sif_exec_test.txt`"]),
    (275, "apptainer_gpu_binding", "Apptainer GPU Binding Test",
     "Verify GPU passthrough into an Apptainer container: nvidia-smi inside "
     "the container, PyTorch CUDA availability test, and driver/runtime "
     "compatibility notes.",
     ["Use `--nv`; document host driver version.",
      "Report CUDA devices visible inside the container."],
     ["`gpu_binding_report.md`", "`torch_cuda_test.json`"]),
    (276, "neurodocker_full_stack", "Neurodocker Full-Stack Image",
     "Generate a Dockerfile with neurodocker containing FSL + AFNI + ANTs + "
     "MRtrix3 + converted dcm2niix, build it, and smoke-test every tool.",
     ["Neurodocker command kept; versions pinned.",
      "Smoke test per tool (version command)."],
     ["`Dockerfile`", "`neurodocker_command.txt`",
      "`smoke_test.json`"]),
    (277, "container_registry_push", "Push Image to GHCR",
     "Tag and push a built analysis image to GitHub Container Registry with "
     "semantic version tags and a usage README.",
     ["Tags: version + sha; no `latest` only.",
      "README documents run command for the image."],
     ["Push log", "`IMAGE_README.md`", "`tags.txt`"]),
    (278, "container_size_audit", "Container Size Audit with dive",
     "Audit a large analysis image with `dive`: identify the biggest layers, "
     "wasted space, and produce a slimming plan.",
     ["Report per-layer sizes.", "Plan must keep functionality; estimate "
      "savings per suggestion."],
     ["`dive_report.txt`", "`slimming_plan.md`"]),
    (279, "sandbox_debug_workflow", "Apptainer Sandbox Debug Workflow",
     "Debug a failing containerized pipeline by rebuilding the SIF as a "
     "writable sandbox, applying a fix interactively, and re-sealing the "
     "image, documenting the whole flow.",
     ["Document every interactive change.",
      "Final image rebuilt cleanly (no sandbox leftovers)."],
     ["`debug_notes.md`", "`changes.diff`", "`rebuild_log.txt`"]),
]
for num, slug, title, desc, cons, outs in _CTR:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- C. CI/CD, T280-T285 --------------------------------------------------------------
_CI = [
    (280, "gha_pytest_matrix", "GitHub Actions: Pytest Matrix",
     "Add a GitHub Actions workflow running the test suite across Python "
     "versions (3.10-3.12) and OS (ubuntu, macOS), with caching.",
     ["Matrix defined explicitly; pip caching enabled.",
      "Badge added to README."],
     ["`.github/workflows/tests.yml`", "`ci_run_summary.md`"]),
    (281, "gha_container_build", "GitHub Actions: Container Build + Push",
     "CI workflow that builds the project Dockerfile on tag pushes and "
     "publishes to a registry with the tag as image version.",
     ["Build only on version tags.", "Push logins via secrets; document "
      "required secrets."],
     ["Workflow YAML", "`secrets_checklist.md`"]),
    (282, "ci_bids_validation_gate", "CI Gate: BIDS Validation",
     "Add a CI job that runs bids-validator on a small reference dataset "
     "and fails the build on errors.",
     ["Reference dataset < 50 MB or stub-based.",
      "Exit-code handling documented."],
     ["Workflow YAML", "`gate_behavior.md`"]),
    (283, "precommit_hooks_setup", "Pre-Commit Hooks Setup",
     "Configure pre-commit with black/ruff (or flake8), trailing-whitespace "
     "and large-file checks, and run it across the repo.",
     ["Config committed as `.pre-commit-config.yaml`.",
      "One full-repo pass; fixes committed separately from config."],
     ["`.pre-commit-config.yaml`", "`first_run_report.txt`"]),
    (284, "codecov_integration", "Coverage Reporting with Codecov",
     "Wire coverage measurement (pytest-cov) into CI and upload to Codecov "
     "with a threshold badge.",
     ["Coverage config (omit tests/, scripts/).",
      "Badge + PR comment behavior documented."],
     ["CI diff", "`coverage_baseline.json`"]),
    (285, "release_changelog_automation", "Release Changelog Automation",
     "Automate changelog generation from conventional commits "
     "(git-cliff or similar), wired into the tag workflow.",
     ["Config committed; changelog groups by type (feat/fix/docs).",
      "Dry-run on existing history included."],
     ["`cliff.toml` (or equivalent)", "`CHANGELOG.md` sample"]),
]
for num, slug, title, desc, cons, outs in _CI:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- D. Workflow engines, T286-T291 ---------------------------------------------------------
_WF = [
    (286, "snakemake_bidsapp_workflow", "Snakemake BIDS-App Workflow",
     "Wrap a BIDS app (e.g. fMRIPrep) in a Snakemake workflow: one rule per "
     "subject, wildcards from the BIDS tree, logs per rule.",
     ["Subject list discovered from the BIDS dataset (pybids or glob).",
      "`snakemake -n` dry-run output included."],
     ["`Snakefile`", "`dry_run.txt`", "`dag.png`"]),
    (287, "nextflow_nfcore_mriqc", "Nextflow: nf-core/mriqc Run",
     "Run the nf-core/mriqc pipeline with Nextflow on a test dataset, "
     "including a site config for the available executor.",
     ["Profile/config file kept.", "Report nextflow + pipeline versions."],
     ["`nextflow.config`", "`run_report.html`", "`timeline.html`"]),
    (288, "snakemake_slurm_profile", "Snakemake SLURM Profile",
     "Create a Snakemake SLURM profile (resources per rule, job naming, log "
     "paths) and demonstrate submission of a small workflow.",
     ["Profile as `slurm/config.yaml`.", "Per-rule mem/time in the "
      "profile, not inline."],
     ["Profile directory", "`submission_log.txt`"]),
    (289, "nextflow_hpc_config", "Nextflow HPC Executor Config",
     "Write a Nextflow config for an institutional SLURM/PBS cluster: "
     "queue selection, per-process resources, singularity enabled.",
     ["Document queue names/partitions as comments.",
      "Test with `nextflow run -profile` hello-level pipeline."],
     ["`nextflow.config`", "`test_run_log.txt`"]),
    (290, "workflow_provenance_tracking", "Workflow Provenance Tracking",
     "Add provenance capture to a workflow: record tool versions, input "
     "hashes, and parameters per run in W3C-PROV-inspired JSON.",
     ["One `prov.json` per run directory.", "Hashes of inputs included."],
     ["`prov.json` example", "`provenance_reader.py`"]),
    (291, "make_pipeline_driver", "Makefile Pipeline Driver",
     "Provide a Makefile that drives the common local pipeline operations "
     "(setup, convert, qc, clean-derivatives) with documented targets.",
     ["Every target has a `## comment` so `make help` works.",
      "No destructive target without a confirm prompt."],
     ["`Makefile`", "`make_help_output.txt`"]),
]
for num, slug, title, desc, cons, outs in _WF:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- E. HPC ops, T292-T297 ---------------------------------------------------------------------
_HPC = [
    (292, "slurm_qos_limits_report", "SLURM QoS / Limits Report",
     "Inventory the cluster's SLURM limits relevant to neuroimaging jobs: "
     "partitions, QoS, max walltime, memory per node, and produce a "
     "job-sizing cheat sheet.",
     ["Use `sacctmgr`/`sinfo` read-only commands.",
      "Cheat sheet as Markdown table."],
     ["`slurm_limits.md`", "`sinfo_snapshot.txt`"]),
    (293, "slurm_array_qsiprep", "SLURM Array: QSIPrep Batch",
     "SLURM array job running QSIPrep over a subject list, one subject per "
     "array task, with per-subject logs and failure collection.",
     ["Array size derived from subject list file.",
      "Dry-run mode prints commands without submitting."],
     ["`run_qsiprep_array.slurm`", "`dry_run_output.txt`"]),
    (294, "slurm_dependency_chain", "SLURM Dependency Chain",
     "Compose a 3-stage SLURM workflow with dependencies: preprocessing -> "
     "denoising -> statistics, each stage starting only after the previous "
     "completes successfully.",
     ["Use `--dependency=afterok:`.", "Each stage idempotent; document "
      "re-run behavior."],
     ["Three .slurm scripts", "`submit_chain.sh`", "`dag.md`"]),
    (295, "pbs_to_slurm_port", "Port PBS Scripts to SLURM",
     "Port existing PBS/Torque job scripts to SLURM equivalents, preserving "
     "resources and array semantics, with a side-by-side mapping table.",
     ["Every PBS directive mapped or explicitly dropped with reason.",
      "Semantics (array indexing, env vars) verified."],
     ["Ported .slurm scripts", "`pbs_slurm_mapping.md`"]),
    (296, "seff_efficiency_report", "SLURM seff Efficiency Report",
     "Analyze completed jobs with `seff`/`sacct`: CPU efficiency, memory "
     "efficiency, and walltime accuracy, then recommend right-sized "
     "requests per job type.",
     ["Group by job name; percentiles reported.",
      "Recommendations in a table (old vs. new request)."],
     ["`efficiency_report.md`", "`sacct_dump.csv`"]),
    (297, "quota_watchdog_script", "Disk Quota Watchdog",
     "Write a watchdog script that checks user/group quota, warns at "
     "configurable thresholds, and logs to a rotating file; wire it to cron "
     "or a systemd timer.",
     ["Read-only on the filesystem.", "Thresholds configurable at the top "
      "of the script."],
     ["`quota_watchdog.sh`", "`crontab_snippet.txt`",
      "`sample_alert.log`"]),
]
for num, slug, title, desc, cons, outs in _HPC:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- F. GPU, T298-T301 ----------------------------------------------------------------------------
_GPU = [
    (298, "cuda_pytorch_smoke", "CUDA + PyTorch Smoke Test",
     "Verify the GPU stack end-to-end: nvidia-smi, CUDA version vs. PyTorch "
     "build compatibility, tensor op on GPU, and a 1-epoch micro training "
     "run.",
     ["Report torch.version.cuda vs. nvidia-smi CUDA.",
      "Micro training must show decreasing loss (5 steps)."],
     ["`gpu_smoke_report.json`", "`micro_train_log.txt`"]),
    (299, "nvidia_container_toolkit_check", "NVIDIA Container Toolkit Check",
     "Verify GPU containers work: `docker run --gpus all` nvidia-smi, plus a "
     "PyTorch CUDA test inside the container, with troubleshooting notes.",
     ["Document toolkit + driver versions.",
      "Include the common failure table (driver mismatch, no --gpus)."],
     ["`container_gpu_report.md`", "`troubleshooting_table.md`"]),
    (300, "gpu_queue_policy_doc", "GPU Queue Usage Policy Doc",
     "Write a team policy document for shared GPU nodes: when to use "
     "interactive vs. batch, memory limits, etiquette for long jobs, and "
     "monitoring commands.",
     ["Concrete commands for every rule.",
      "One page; Markdown."],
     ["`GPU_POLICY.md`"]),
    (301, "cuda_compatibility_matrix", "CUDA Compatibility Matrix",
     "Build a compatibility matrix for the deep-learning stack: driver vs. "
     "CUDA toolkit vs. PyTorch/TensorFlow versions used by the project, "
     "with the verified-good combinations marked.",
     ["Mark the exact combo verified on the target machine.",
      "Include container tags that pin each combo."],
     ["`cuda_matrix.md`"]),
]
for num, slug, title, desc, cons, outs in _GPU:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- G. Sync / session tooling, T302-T307 ------------------------------------------------------------
_SYNC = [
    (302, "rclone_remote_setup", "Rclone Remote Configuration",
     "Configure rclone remotes for the lab storage (S3 + SFTP), with "
     "obscured passwords and a transfer cheat sheet.",
     ["Config file permissions 600.", "No plaintext secrets anywhere."],
     ["`rclone.conf` template", "`cheat_sheet.md`"]),
    (303, "ssh_hpc_jump_setup", "SSH Jump-Host Setup for HPC",
     "Set up SSH config for reaching the cluster through a jump host: "
     "key-based auth, ProxyJump, keepalive, and a connection test log.",
     ["`~/.ssh/config` snippet provided.", "No passwords stored."],
     ["`ssh_config_snippet`", "`connection_test_log.txt`"]),
    (304, "tmux_pipeline_session", "Tmux Pipeline Session Script",
     "Script a tmux session layout for pipeline babysitting: panes for job "
     "watch (squeue), logs tail, GPU watch, and scratch shell.",
     ["One script creates the full layout.", "Document key bindings used."],
     ["`pipeline_tmux.sh`", "`layout_screenshot.txt` (or description)"]),
    (305, "cron_pipeline_monitor", "Cron Pipeline Completion Monitor",
     "Cron-driven monitor that checks a SLURM job list and a derivatives "
     "directory, then emails/writes a digest of completed/failed subjects.",
     ["Digest format: counts + failed list.", "Idempotent; no duplicate "
      "alerts for the same state."],
     ["`pipeline_monitor.sh`", "`sample_digest.md`"]),
    (306, "git_lfs_migration", "Git LFS Migration for Large Assets",
     "Migrate large non-data assets (figures, small atlases, model weights "
     "< 100 MB) from plain git to Git LFS, rewriting tracking without "
     "breaking clones.",
     ["`.gitattributes` patterns documented.",
      "Verify fresh clone pulls LFS objects correctly."],
     ["`.gitattributes`", "`migration_log.txt`"]),
    (307, "git_submodule_pinning", "Git Submodule Pinning Audit",
     "Audit submodules in the analysis monorepo: pin each to a commit, "
     "document the upstream URL + pinned SHA, and add update instructions.",
     ["Table of submodule -> SHA -> purpose.",
      "Include the update workflow commands."],
     ["`submodules.md`", "`.gitmodules` updates"]),
]
for num, slug, title, desc, cons, outs in _SYNC:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- H. Reproducibility, T308-T313 ---------------------------------------------------------------------------
_REPRO = [
    (308, "repro_env_capture", "Full Environment Capture Report",
     "Capture the complete computational environment of an analysis "
     "machine: OS, kernel, Python/R/conda packages, container versions, GPU "
     "stack, into one reproducibility report.",
     ["Single script generates everything.", "Report includes capture "
      "timestamp + hostname."],
     ["`env_capture.sh`", "`environment_report.md`"]),
    (309, "pip_constraints_file", "pip Constraints File Generation",
     "Produce a constraints file that pins the transitive dependency tree "
     "of the analysis environment, and verify a clean venv install from "
     "it.",
     ["Constraints include hashes if pip-tools available.",
      "Clean venv verification log."],
     ["`constraints.txt`", "`clean_install_log.txt`"]),
    (310, "r_renv_lockfile", "R renv Lockfile Setup",
     "Initialize renv for the R analysis code, generate renv.lock, and "
     "verify `renv::restore()` in a clean library.",
     ["Snapshot type documented.", "Restore verification log."],
     ["`renv.lock`", "`restore_log.txt`"]),
    (311, "matlab_toolbox_inventory", "MATLAB Toolbox Inventory",
     "Inventory the MATLAB environment used by SPM/CAT12 analyses: MATLAB "
     "version, toolbox list + versions, SPM/CAT12 revisions, and license "
     "type.",
     ["`ver` output captured.", "SPM/CAT12 revision from their VERSION "
      "files."],
     ["`matlab_inventory.md`"]),
    (312, "jupyter_kernel_registry", "Jupyter Kernel Registry",
     "Register project conda environments as named Jupyter kernels with "
     "display names, and verify each kernel starts and imports its key "
     "package.",
     ["Kernel names match env names.", "Startup test per kernel logged."],
     ["`kernel_install_log.txt`", "`kernel_test.json`"]),
    (313, "devcontainer_setup", "VS Code Devcontainer Setup",
     "Add a `.devcontainer` configuration that reproduces the analysis "
     "environment in VS Code: image, extensions, post-create installs, "
     "mounts.",
     ["postCreateCommand installs project deps.",
      "Document how to open + rebuild."],
     ["`.devcontainer/devcontainer.json`", "`DEVCONTAINER.md`"]),
]
for num, slug, title, desc, cons, outs in _REPRO:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- I. Docs / writing tooling, T314-T317 ------------------------------------------------------------------------
_DOCS = [
    (314, "overleaf_git_sync", "Overleaf Git Sync Workflow",
     "Set up two-way git sync between a local paper repository and "
     "Overleaf: clone, push conventions, conflict-avoidance rules, and CI "
     "compile check.",
     ["Document the exact remote setup.", "Rules for who edits where."],
     ["`OVERLEAF_SYNC.md`", "`sync_test_log.txt`"]),
    (315, "latex_ci_compile", "LaTeX CI Compile",
     "CI job that compiles the manuscript with tectonic (or latexmk) on "
     "push, uploads the PDF artifact, and fails on LaTeX errors.",
     ["Warnings summarized in the job log.",
      "Artifact retention documented."],
     ["Workflow YAML", "`compile_log_sample.txt`"]),
    (316, "zotero_betterbib_ci", "Zotero Better-BibTeX CI Export",
     "Automate a citation .bib export from Zotero (Better-BibTeX) into the "
     "paper repo, with a CI check that all \\cite keys resolve.",
     ["Key format documented.", "CI check greps \\cite keys vs. bib."],
     ["`references.bib` update flow", "`cite_check.sh`"]),
    (317, "mkdocs_site_deploy", "MkDocs Documentation Site",
     "Stand up an MkDocs (Material) documentation site for the project: "
     "structure, API reference stubs, GitHub Pages deploy workflow.",
     ["Nav mirrors the repo layout.", "Deploy via gh-pages workflow."],
     ["`mkdocs.yml`", "`docs/` skeleton", "`deploy_log.txt`"]),
]
for num, slug, title, desc, cons, outs in _DOCS:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- J. Testing / QA infra, T318-T322 ----------------------------------------------------------------------------------
_QA = [
    (318, "pytest_synthetic_nifti", "Pytest Fixtures: Synthetic NIfTI",
     "Create pytest fixtures that synthesize small NIfTI/BIDS-like data for "
     "unit tests: deterministic 4D volumes, sidecar JSONs, temp BIDS trees.",
     ["Deterministic seeds.", "Fixtures in conftest.py with docstrings."],
     ["`conftest.py` additions", "`test_fixture_demo.py`"]),
    (319, "regression_test_hashes", "Regression Tests via Output Hashes",
     "Add regression tests that hash key pipeline outputs on a tiny fixture "
     "dataset and fail when numerical outputs change unexpectedly.",
     ["Hash tolerance policy documented (exact vs. near).",
      "Baseline hashes versioned."],
     ["`test_regression.py`", "`baseline_hashes.json`"]),
    (320, "smoke_suite_bidsapps", "Smoke Suite for BIDS Apps",
     "A single smoke-test script that verifies all containerized BIDS apps "
     "used by the lab start and respond (--version/--help) with expected "
     "versions.",
     ["Expected versions in a config file.", "One command runs all "
      "checks."],
     ["`smoke_bidsapps.sh`", "`expected_versions.yaml`",
      "`smoke_report.json`"]),
    (321, "markdown_lint_precommit", "Markdown Linting in Pre-Commit",
     "Add markdownlint (or pymarkdown) to pre-commit with a tuned rule set "
     "for the repo's docs and fix the existing violations.",
     ["Rule config committed.", "Fixes in a separate commit from the "
      "config."],
     ["`.markdownlint.yaml`", "`lint_fix_summary.md`"]),
    (322, "ci_caching_strategy", "CI Caching Strategy",
     "Optimize CI runtime with a caching strategy: pip/conda caches, docker "
     "layer caches, and dataset fixture caches; measure before/after.",
     ["Cache keys documented.", "Before/after timings in the report."],
     ["CI YAML changes", "`cache_benchmark.md`"]),
]
for num, slug, title, desc, cons, outs in _QA:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- K. Security / compliance, T323-T326 -----------------------------------------------------------------------------------
_SEC = [
    (323, "credential_audit_repo", "Credential Handling Audit",
     "Audit the repository for credential risks: scan history for secrets "
     "(gitleaks), inventory where credentials are used, and write handling "
     "rules.",
     ["Use gitleaks or equivalent on full history.",
      "Rules doc covers env vars, .env files, CI secrets."],
     ["`gitleaks_report.json`", "`CREDENTIAL_POLICY.md`"]),
    (324, "freesurfer_license_mgmt", "FreeSurfer License Management",
     "Document and script correct FreeSurfer license handling across "
     "local/Docker/Singularity runs: where the file lives, how it is "
     "mounted, and how CI handles it.",
     ["Never commit the license file.", "Mount examples per runtime."],
     ["`FREESURFER_LICENSE.md`", "`mount_examples.sh`"]),
    (325, "phi_scan_git_history", "PHI Scan of Git History",
     "Scan the repository git history for accidentally committed subject "
     "identifiers or PHI patterns, and produce a remediation plan "
     "(BFG/filter-repo) if found.",
     ["Pattern list documented.", "No history rewriting in this task; "
      "plan only."],
     ["`phi_history_scan.json`", "`remediation_plan.md`"]),
    (326, "access_log_review_script", "Data Access Log Review Script",
     "Script a review of dataset access/download logs: who accessed what "
     "when, anomaly flags (bulk downloads), formatted for the data-use "
     "agreement audit.",
     ["Input log format documented.", "Anomaly thresholds configurable."],
     ["`access_review.py`", "`sample_review.md`"]),
]
for num, slug, title, desc, cons, outs in _SEC:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

# --- L. Monitoring / tracking, T327-T330 + M. misc T331? -> numbering fixed below ---------------
_MON = [
    (327, "wandb_project_setup", "Weights & Biases Project Setup",
     "Set up W&B experiment tracking for the model-training code: project "
     "config, metric logging, artifact versioning for checkpoints, and a "
     "sample run.",
     ["API key via env only.", "Sample run uploaded with config + "
      "metrics."],
     ["`wandb/` sample run", "`WANDB_SETUP.md`"]),
    (328, "mlflow_tracking_server", "MLflow Tracking Server Setup",
     "Deploy a local MLflow tracking server (sqlite backend + artifact "
     "store), point the training scripts at it, and log a sample run.",
     ["Server start script + env config.", "Sample run visible in UI "
      "(screenshot or export)."],
     ["`mlflow_server.sh`", "`sample_run_export.json`"]),
    (329, "slack_pipeline_notifier", "Pipeline Slack/Webhook Notifier",
     "Add a notifier that posts pipeline completion/failure digests to a "
     "webhook (Slack/Teams/ntfy), callable from SLURM scripts and local "
     "runs.",
     ["Webhook URL via env/config, never committed.",
      "Message format: job, status, duration, failed subjects."],
     ["`notify.sh`", "`sample_message.json`"]),
    (330, "tensorboard_launcher", "TensorBoard Launcher for Training Runs",
     "Script a TensorBoard launcher that discovers all run directories "
     "under models/, starts TensorBoard with the right logdir structure, "
     "and prints the URL.",
     ["Discovers nested run dirs automatically.",
      "Port configurable; handles occupied ports."],
     ["`launch_tensorboard.sh`", "`discovered_runs.txt`"]),
]
for num, slug, title, desc, cons, outs in _MON:
    TASKS.append(_t(num, slug, title, desc, ins=None, cons=cons, outs=outs))

assert len(TASKS) == 63, f"dev_env batch must be 63 tasks, got {len(TASKS)}"
