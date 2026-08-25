---
name: hcpya-skill
description: "Use this skill whenever the user wants an end-to-end workflow for the HCP Young Adult (HCP-YA / HCP1200) dataset, including dataset download, BIDS organization, and multimodal processing of sMRI, fMRI, and dMRI. Triggers include: 'HCP Young Adult', 'HCP-YA', 'HCP1200', 'process HCP data', 'HCP sMRI fMRI DTI', or any request to run the HCP-YA multimodal pipeline."
license: MIT License (NeuroClaw custom skill - freely modifiable within the project)
layer: subagent
skill_type: dataset
dependencies:
  - smri-skill
  - fmri-skill
  - dwi-skill
  - bids-organizer
  - claw-shell
complementary_skills:
  - hcppipeline-tool
---
# HCP-YA Skill (Dataset-Orchestration Layer)

## Overview

`hcpya-skill` is the NeuroClaw orchestration skill for the **HCP Young Adult (HCP-YA / HCP1200)** dataset.

It strictly follows the NeuroClaw hierarchical design principles:
- This skill **only describes WHAT needs to be done** and **which tool skill to delegate to**.
- It contains **no implementation code or concrete commands**.
- All concrete execution is delegated to existing base/tool skills via `claw-shell`.
- Companion scripts in `scripts/` provide reference implementations for data reorganization, phenotype extraction, and QC.

**Core workflow (never bypassed):**
1. Identify input HCP-YA data and target modalities.
2. Generate a **numbered execution plan** clearly stating WHAT needs to be done and which tool skill will handle each step.
3. Present the full plan, estimated runtime, resource requirements, and risks to the user and wait for explicit confirmation ("YES" / "execute" / "proceed").
4. On confirmation, delegate every step to the appropriate skill via `claw-shell`.
5. After execution, save all outputs in a clean directory structure (`hcpya_output/`).

**Research use only.**

---

## Quick Reference

| Task | What needs to be done | Delegate to | Expected output |
|---|---|---|---|
| Data download | Select HCP-YA packages in ConnectomeDB powered by BALSA after registration and terms acceptance | `claw-shell` | Raw or preprocessed HCP-YA packages |
| BIDS staging | Reorganize HCP-YA native layout to BIDS | `scripts/reorganize_hcpya.py` | BIDS-compliant dataset |
| sMRI processing | Brain extraction, tissue segmentation, cortical reconstruction | `smri-skill` | `smri_output/` derivatives |
| fMRI processing | Preprocessing, denoising, connectivity, task GLM | `fmri-skill` | `fmri_output/` derivatives |
| dMRI processing | Eddy correction, tensor metrics, tractography | `dwi-skill` | `dwi_output/` metrics |
| Phenotype extraction | Cognitive, behavioral, demographic data | `scripts/extract_hcpya_phenotype.py` | Merged phenotype CSV |
| QC summary | Per-subject quality control | `scripts/hcpya_qc_summary.py` | QC summary + exclusion list |

---

## Download Stage (Mandatory First Step)

### Source
The current HCP-YA release is distributed through **ConnectomeDB powered by BALSA**:
- Release: https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release
- Platform: https://balsa.wustl.edu/ (open the ConnectomeDB tab)
- Register and accept the applicable HCP data-use terms before downloading.
- Restricted non-imaging variables require the corresponding restricted-data approval.

### Release Selection
- Prefer the **HCP-YA 2025 Release** for new work. It contains updated processing and package organization.
- Do not mix 2025 processed packages with the legacy 2017 S1200 processed release in one analysis.
- Use the legacy controlled-access S3 route only when reproducing an S1200-era workflow and after satisfying HCP access terms.
- Download a modality-specific package or explicit subject subset instead of assuming the entire cohort is needed.

### Download Inputs to Confirm in Plan
- BALSA/ConnectomeDB account and accepted data-use terms
- Release (`2025` by default or legacy `S1200` for reproduction)
- Target package and modalities (structural, resting fMRI, task fMRI, diffusion, or non-imaging)
- Subject list scope and restricted-variable requirements
- Destination directory with capacity calculated from the selected packages

---

## HCP-YA Task Paradigms

| Task | Description | Duration |
|---|---|---|
| MOTOR | Finger tapping, toe movement, tongue movement | ~3 min |
| EMOTION | Faces and shapes matching | ~2 min |
| GAMBLING | Card guessing with reward/loss | ~3 min |
| LANGUAGE | Story comprehension and math | ~4 min |
| RELATIONAL | Relational reasoning matching | ~3 min |
| SOCIAL | Social cognition (mentalizing) movie clips | ~3 min |
| WM | Working memory (faces, places, tools, body parts) | ~5 min |
| REST | Resting-state (eyes open) | ~15 min × 4 runs |

---

## BIDS Preparation

### Script: `scripts/reorganize_hcpya.py`

Converts HCP-YA native directory structure to BIDS-compliant layout.

```bash
python skills/hcpya-skill/scripts/reorganize_hcpya.py \
  --input /path/to/HCPYA/raw \
  --output /path/to/HCPYA/bids \
  --participants /path/to/subject_list.txt
```

Features:
- Subject ID normalization: HCP format (`100307`) to BIDS (`sub-100307`)
- Modality routing: T1w, T2w, dMRI, rs-fMRI, task-fMRI (7 tasks)
- Sidecar JSON generation from HCP metadata
- `dataset_description.json` and `participants.tsv` generation
- Dry-run mode: `--dry-run` to preview without copying

---

## Core Workflow (Never Bypassed)

1. Identify user target: full HCP-YA processing, imaging subset, phenotype extraction, or BIDS staging only.
2. Generate a numbered plan with tools, outputs, runtime, storage, and risks.
3. Wait for explicit confirmation (`YES` / `execute` / `proceed`).
4. On confirmation, run download stage first (if needed).
5. After download success, run BIDS preparation using `scripts/reorganize_hcpya.py`.
6. Delegate to `smri-skill` for structural MRI processing.
7. Delegate to `fmri-skill` for functional MRI processing.
8. Delegate to `dwi-skill` for diffusion MRI processing.
9. If phenotype extraction is requested, run `scripts/extract_hcpya_phenotype.py`.
10. If QC summary is requested, run `scripts/hcpya_qc_summary.py`.
11. Save outputs into `hcpya_output/`.

---

## Modality Processing Delegation

| Modality | Delegated skill | Typical tasks | Main outputs |
|---|---|---|---|
| sMRI (T1w/T2w) | `smri-skill` | brain extraction, tissue segmentation, cortical reconstruction, ROI morphometry | `smri_output/` derivatives |
| fMRI (rs-fMRI/task-fMRI) | `fmri-skill` | preprocessing, denoising, ROI time series, connectivity, task GLM | `fmri_output/` derivatives |
| dMRI (DWI) | `dwi-skill` | eddy correction, tensor metrics, tractography, connectome | `dwi_output/` metrics |

---

## Standard Output Layout

```
hcpya_output/
├── raw/                    # Downloaded original HCP-YA files
├── bids/                   # BIDS-staged data
├── smri/                   # Structural MRI derivatives
├── fmri/                   # Functional MRI derivatives
├── dwi/                    # Diffusion MRI derivatives
├── phenotype/              # Merged phenotype tables
├── qc/                     # QC summaries and exclusion lists
└── logs/                   # Download + orchestration logs
```

---

## Benchmark Adapter Guidance

For benchmark-style prompts, do not force the full `download -> staging -> multimodal processing` orchestration when the task only asks for local HCP-YA data staging or organization.

- If the task starts from raw HCP-YA data already present on disk and only asks for BIDS-style staging:
  - Skip the mandatory download stage
  - Default to the narrow path `local raw HCP-YA discovery -> BIDS-style staging -> minimal metadata -> validation/report`
- In benchmark mode, do not require explicit confirmation before presenting the direct staging solution.
- Only use the full multimodal orchestration when the prompt explicitly asks for download or end-to-end processing.

---

## Safety and Execution Policy
- No execution before explicit plan confirmation.
- All execution must be routed via `claw-shell`.
- Missing dependencies must be resolved by `dependency-planner` before running.
- If download fails for partial subjects, continue batch with clear failure report and retry list.

---

## Important Notes and Limitations
- HCP-YA processing is resource intensive (CPU, RAM, and storage).
- HCP-YA releases span many large packages; estimate storage from the selected BALSA packages before transfer.
- The 2025 release reports processed data for 1,071 subjects and unprocessed imaging for 1,113 subjects; "HCP1200" is the cohort/release name, not a complete-case count.
- Age range: 22-35 years.
- For HCP-native preprocessing (minimal preprocessing pipelines), optionally delegate to `hcppipeline-tool`.
- `hcpya-skill` is orchestration-only; detailed preprocessing logic remains in modality skills.

---

## When to Call This Skill
- User asks for end-to-end HCP Young Adult workflow.
- User asks to download HCP1200 and run sMRI/fMRI/DTI processing.
- User needs BIDS staging for HCP-YA data.
- User asks to extract HCP-YA phenotype data (cognitive, behavioral, demographic).

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
- HCP-YA: https://www.humanconnectome.org/study/hcp-young-adult
- HCP-YA 2025 release: https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release
- HCP data-use terms: https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms
- Glasser et al. (2013): The Human Connectome Project minimally preprocessed pipelines

Created At: 2026-05-06 13:02 HKT
Last Updated At: 2026-08-11 HKT
Author: chengwang96
