---
name: hcpa-skill
description: "Use this skill whenever the user wants an end-to-end workflow for the HCP Aging (HCP-A) dataset, including dataset download, BIDS organization, and multimodal processing of sMRI, fMRI, and dMRI. Triggers include: 'HCP Aging', 'HCP-A', 'process HCP Aging data', 'HCP Aging sMRI fMRI', or any request to run the HCP-A multimodal pipeline."
license: MIT License (NeuroClaw custom skill - freely modifiable within the project)
layer: subagent
skill_type: dataset
dependencies:
  - smri-skill
  - fmri-skill
  - dwi-skill
  - asl-skill
  - bids-organizer
  - claw-shell
complementary_skills:
  - hcppipeline-tool
---
# HCP-A Skill (Dataset-Orchestration Layer)

## Overview

`hcpa-skill` is the NeuroClaw orchestration skill for the **HCP Aging (HCP-A)** dataset.

It strictly follows the NeuroClaw hierarchical design principles:
- This skill **only describes WHAT needs to be done** and **which tool skill to delegate to**.
- It contains **no implementation code or concrete commands**.
- All concrete execution is delegated to existing base/tool skills via `claw-shell`.
- Companion scripts in `scripts/` provide reference implementations for data reorganization, phenotype extraction, and QC.

**Core workflow (never bypassed):**
1. Identify input HCP-A data and target modalities.
2. Generate a **numbered execution plan** clearly stating WHAT needs to be done and which tool skill will handle each step.
3. Present the full plan, estimated runtime, resource requirements, and risks to the user and wait for explicit confirmation ("YES" / "execute" / "proceed").
4. On confirmation, delegate every step to the appropriate skill via `claw-shell`.
5. After execution, save all outputs in a clean directory structure (`hcpa_output/`).

**Research use only.**

---

## Quick Reference

| Task | What needs to be done | Delegate to | Expected output |
|---|---|---|---|
| Data download | Select HCP-A/AABC packages in ConnectomeDB powered by BALSA | `claw-shell` | Raw or preprocessed imaging packages |
| BIDS staging | Reorganize HCP-A native layout to BIDS | `scripts/reorganize_hcpa.py` | BIDS-compliant dataset |
| sMRI processing | Brain extraction, tissue segmentation, cortical reconstruction | `smri-skill` | `smri_output/` derivatives |
| fMRI processing | Preprocessing, denoising, connectivity, task GLM | `fmri-skill` | `fmri_output/` derivatives |
| dMRI processing | Eddy correction, tensor metrics, tractography | `dwi-skill` | `dwi_output/` metrics |
| ASL processing | Perfusion preprocessing and CBF quantification | `asl-skill` | `asl_output/` derivatives |
| Phenotype extraction | Cognitive, health, demographic data | `scripts/extract_hcpa_phenotype.py` | Merged phenotype CSV |
| QC summary | Per-subject quality control | `scripts/hcpa_qc_summary.py` | QC summary + exclusion list |

---

## Download Stage (Mandatory First Step)

### Source
Current HCP-A/AABC data is distributed through **ConnectomeDB powered by BALSA**:
- Release page: https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases
- Current release: AABC Release 2 (2026-01-28)
- Register for BALSA and accept the AABC Data Use Terms. An academic, nonprofit, or government email address is required.
- Imaging packages transfer through IBM Aspera Connect. Select a modality package or subject subset and calculate storage before transfer.

### Dataset Characteristics
- **AABC Release 2**: 1,396 participants and 2,878 sessions; imaging is available for 1,390 participants across 2,789 sessions
- **Modalities**: T1w, T2w, high-resolution hippocampal T2, dMRI, rs-fMRI, task-fMRI, and ASL
- **Focus**: Normal aging, cognitive decline, brain structure-function changes across the lifespan
- **Unique feature**: Complements HCP-YA to cover the full adult lifespan (22-100 years)

### Download Inputs to Confirm in Plan
- BALSA/ConnectomeDB account and accepted AABC terms
- Release (`AABC Release 2` by default or legacy `HCP-A Lifespan 2.0` for reproduction)
- Target modalities (structural, functional, diffusion, ASL, or non-imaging)
- Subject list scope (full or custom subset)
- Destination directory with capacity calculated from selected package sizes

---

## HCP-A Task Paradigms

| Task | Description |
|---|---|
| VISMOTOR | Simultaneous visual and motor activation paradigm |
| CARIT | Conditioned Approach Response Inhibition Task |
| FACENAME | Face-name paired-associates memory task |
| REST | Resting-state functional MRI |

---

## BIDS Preparation

### Script: `scripts/reorganize_hcpa.py`

Converts HCP-A native directory structure to BIDS-compliant layout.

```bash
python skills/hcpa-skill/scripts/reorganize_hcpa.py \
  --input /path/to/HCPA/raw \
  --output /path/to/HCPA/bids \
  --participants /path/to/subject_list.txt
```

Features:
- Subject ID normalization: HCP format to BIDS `sub-` labels
- Session handling: multiple visits if applicable
- Modality routing: T1w, T2w, dMRI, rs-fMRI, task-fMRI
- Sidecar JSON generation from HCP metadata
- `dataset_description.json` and `participants.tsv` generation
- Dry-run mode: `--dry-run` to preview without copying

---

## Core Workflow (Never Bypassed)

1. Identify user target: full HCP-A processing, imaging subset, phenotype extraction, or BIDS staging only.
2. Generate a numbered plan with tools, outputs, runtime, storage, and risks.
3. Wait for explicit confirmation (`YES` / `execute` / `proceed`).
4. On confirmation, run download stage first (if needed).
5. After download success, run BIDS preparation using `scripts/reorganize_hcpa.py`.
6. Delegate to `smri-skill` for structural MRI processing.
7. Delegate to `fmri-skill` for functional MRI processing.
8. Delegate to `dwi-skill` for diffusion MRI processing.
9. Delegate to `asl-skill` when ASL data is selected.
10. If phenotype extraction is requested, run `scripts/extract_hcpa_phenotype.py`.
11. If QC summary is requested, run `scripts/hcpa_qc_summary.py`.
12. Save outputs into `hcpa_output/`.

---

## Modality Processing Delegation

| Modality | Delegated skill | Typical tasks | Main outputs |
|---|---|---|---|
| sMRI (T1w/T2w) | `smri-skill` | brain extraction, tissue segmentation, cortical reconstruction, ROI morphometry | `smri_output/` derivatives |
| fMRI (rs-fMRI/task-fMRI) | `fmri-skill` | preprocessing, denoising, ROI time series, connectivity, task GLM | `fmri_output/` derivatives |
| dMRI (DWI) | `dwi-skill` | eddy correction, tensor metrics, tractography, connectome | `dwi_output/` metrics |
| ASL | `asl-skill` | perfusion preprocessing and cerebral blood-flow quantification | `asl_output/` derivatives |

---

## Standard Output Layout

```
hcpa_output/
├── raw/                    # Downloaded original HCP-A files
├── bids/                   # BIDS-staged data
├── smri/                   # Structural MRI derivatives
├── fmri/                   # Functional MRI derivatives
├── dwi/                    # Diffusion MRI derivatives
├── asl/                    # ASL/perfusion derivatives
├── phenotype/              # Merged phenotype tables
├── qc/                     # QC summaries and exclusion lists
└── logs/                   # Download + orchestration logs
```

---

## Benchmark Adapter Guidance

For benchmark-style prompts, do not force the full orchestration when the task only asks for local HCP-A data staging.

- If the task starts from raw HCP-A data already present on disk and only asks for BIDS-style staging:
  - Skip the mandatory download stage
  - Default to the narrow path `local raw HCP-A discovery -> BIDS-style staging -> minimal metadata -> validation/report`
- In benchmark mode, do not require explicit confirmation before presenting the direct staging solution.

---

## Safety and Execution Policy
- No execution before explicit plan confirmation.
- All execution must be routed via `claw-shell`.
- Missing dependencies must be resolved by `dependency-planner` before running.

---

## Important Notes and Limitations
- HCP-A complements HCP-YA to cover the full adult lifespan (22-100 years).
- HCP-A processing is resource intensive; plan storage and compute accordingly.
- The HCP-A/AABC cohort is designed around typical aging; do not infer a clinical impairment cohort from age alone.
- Do not mix legacy Lifespan 2.0 packages with AABC Release 2 without documenting and harmonizing release-specific processing differences.
- For HCP-native preprocessing, optionally delegate to `hcppipeline-tool`.
- `hcpa-skill` is orchestration-only; detailed preprocessing logic remains in modality skills.

---

## When to Call This Skill
- User asks for end-to-end HCP Aging workflow.
- User asks to download HCP-A and run sMRI/fMRI/DTI processing.
- User needs BIDS staging for HCP-A data.
- User asks to extract HCP-A phenotype data (cognitive, health, demographic).

---

## Complementary / Related Skills
- `smri-skill` → structural MRI preprocessing
- `fmri-skill` → functional MRI preprocessing and analysis
- `dwi-skill` → diffusion MRI preprocessing and analysis
- `hcppipeline-tool` → HCP-native minimal preprocessing pipelines
- `bids-organizer` → BIDS validation and organization
- `brain-visualization` → visualization of derivatives
- `dependency-planner` → dependency resolution
- `conda-env-manager` → environment management
- `claw-shell` → command execution

---

## Reference
- HCP Aging: https://www.humanconnectome.org/study/hcp-lifespan-aging
- AABC Release 2: https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases
- HCP-A task protocols: https://www.humanconnectome.org/study/hcp-lifespan-aging/project-protocols
- Bookheimer et al. (2019): The Lifespan Human Connectome Project in Aging

Created At: 2026-05-06 13:02 HKT
Last Updated At: 2026-08-11 HKT
Author: chengwang96
