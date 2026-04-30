---
name: hcp-skill
description: "Use this skill whenever the user wants an end-to-end workflow for the HCP Young Adult dataset, including dataset download, organization, and multimodal processing of sMRI, fMRI, and DTI. Triggers include: 'HCP Young Adult', 'HCP1200', 'process HCP data', 'HCP sMRI fMRI DTI', or any request to run the full HCP multimodal pipeline. This is the NeuroClaw modality-orchestration layer: it plans WHAT to do and delegates execution to other skills."
license: MIT License (NeuroClaw custom skill - freely modifiable within the project)
---

# HCP Skill (Dataset-Orchestration Layer)

## Overview
`hcp-skill` is the NeuroClaw orchestration skill for the **HCP Young Adult (HCP1200)** dataset.

It coordinates a fixed three-phase workflow:
1. Download HCP data using NeuroSTORM download scripts.
2. Prepare/validate data organization for downstream processing.
3. Delegate modality pipelines to `smri-skill`, `fmri-skill`, and `dwi-skill`.

This skill follows NeuroClaw hierarchy:
- Defines **WHAT to do**, not low-level implementation details.
- Does **not** execute direct shell commands itself.
- Delegates all execution via `claw-shell` to base/tool skills.

**Research use only.**

---

## Core Workflow (Never Bypassed)
1. Identify user target: full HCP1200 download, or subset (`rfMRI`, `tfMRI`, `t1t2`, `all`).
2. Generate a numbered plan with tools, outputs, runtime, storage, and risks.
3. Wait for explicit confirmation (`YES` / `execute` / `proceed`).
4. On confirmation, run download stage first.
5. After download success, delegate sequentially or in parallel to:
	 - `smri-skill` for structural MRI
	 - `fmri-skill` for functional MRI
	 - `dwi-skill` for diffusion MRI
6. Save outputs into an HCP-centered structure under `hcp_output/`.

---

## Download Stage (Mandatory First Step)

### Source Repository
Use scripts from:
- `https://github.com/CUHK-AIM-Group/NeuroSTORM/tree/main/scripts/dataset_download`

### Supported Download Entry Scripts
- `download_HCP_1200_all.py` (all modalities)
- `download_HCP_1200_rfMRI.py` (resting-state fMRI focused)
- `download_HCP_1200_tfMRI.py` (task fMRI focused)
- `download_HCP_1200_t1t2.py` (structural T1w/T2w focused)
- `all_pid.pkl` (subject list / metadata used by downloader)

### Delegation Rules for Download
- Environment/setup checks: `dependency-planner` + `conda-env-manager`
- Repository/script fetch and execution: `claw-shell`
- Optional raw-data organization to BIDS-style staging: `bids-organizer`

### Download Inputs to Confirm in Plan
- HCP credentials/token availability
- Target subset (`all`, `rfMRI`, `tfMRI`, `t1t2`)
- Subject list scope (full or custom IDs)
- Destination directory with sufficient disk space

---

## Multimodal Processing Delegation

After download completes, `hcp-skill` delegates by modality:

| Modality | Delegated skill | Typical tasks | Main outputs |
|---|---|---|---|
| sMRI (T1w/T2w) | `smri-skill` | brain extraction, tissue segmentation, cortical reconstruction, ROI morphometry | `smri_output/` derivatives and stats |
| fMRI (rfMRI/tfMRI) | `fmri-skill` | preprocessing, denoising, ROI time series, connectivity | `fmri_output/` derivatives, timeseries, connectivity |
| DTI/DWI | `dwi-skill` | diffusion preprocessing, tensor metrics, tractography/connectome | `dwi_output/` or `dti_output/` metrics and tract files |

### Delegation Strategy
- If user asks for full multimodal HCP analysis: run sMRI -> fMRI -> DTI in ordered phases.
- If user asks for one modality only: call only the corresponding modality skill.
- If compute resources are adequate and the user approves parallel runs: run modality pipelines in parallel after shared prerequisites are ready.

---

## Recommended Output Layout
All assets should be organized under `./hcp_output/`:
- `hcp_output/raw/` (downloaded original HCP files)
- `hcp_output/staging/` (optional normalized/BIDS-like staging)
- `hcp_output/smri/` (links or copies from `smri_output/`)
- `hcp_output/fmri/` (links or copies from `fmri_output/`)
- `hcp_output/dti/` (links or copies from `dti_output/` or `dwi_output/`)
- `hcp_output/logs/` (download + orchestration logs)

---

## Benchmark Adapter Guidance

For benchmark-style prompts, do not force the full `download -> staging -> multimodal processing` orchestration when the task is only asking for local HCP data staging or organization.

- If the task starts from raw HCP data already present on disk and only asks for BIDS-style staging / organization:
	- skip the mandatory download stage
	- do not automatically delegate to `smri-skill`, `fmri-skill`, or `dwi-skill`
	- default to the narrow path `local raw HCP discovery -> BIDS-style staging -> minimal metadata -> validation/report`
- In benchmark mode, do not require explicit confirmation before presenting the direct staging solution.
- Preserve the HCP-centered output contract under `hcp_output/staging/` when the task is specifically a staging benchmark.
- Only use the full multimodal orchestration and confirmation-heavy workflow when the prompt explicitly asks for download, end-to-end multimodal HCP processing, or post-staging structural / functional / diffusion analysis.

---

## Safety and Execution Policy
- No execution before explicit plan confirmation.
- All execution must be routed via `claw-shell`.
- Missing dependencies must be resolved by `dependency-planner` before running.
- If download fails for partial subjects, continue batch with clear failure report and retry list.

---

## Important Notes and Limitations
- HCP multimodal processing is resource intensive (CPU, RAM, and storage).
- Download stage may require stable network and credential renewal.
- HCP directory conventions can vary by subset; always validate expected files before starting modality pipelines.
- `hcp-skill` is orchestration-only; detailed preprocessing logic remains in `smri-skill`, `fmri-skill`, and `dwi-skill`.
- For highest-fidelity HCP-native preprocessing, optionally delegate to `hcppipeline-tool` as an alternative route.

---

## When to Call This Skill
- User asks for end-to-end HCP Young Adult workflow.
- User asks to download HCP1200 and then run sMRI/fMRI/DTI processing.
- User needs a single entry point for HCP multimodal orchestration.

---

## Complementary / Related Skills
- `smri-skill`
- `fmri-skill`
- `dwi-skill`
- `hcppipeline-tool`
- `bids-organizer`
- `dependency-planner`
- `conda-env-manager`
- `claw-shell`

---

## Reference
- NeuroSTORM HCP download scripts:
	- `https://github.com/CUHK-AIM-Group/NeuroSTORM/tree/main/scripts/dataset_download`

Created At: 2026-03-28 19:27 HKT
Last Updated At: 2026-03-28 19:27 HKT
Author: chengwang96