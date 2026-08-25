---
name: scan-skill
description: "Use this skill for the Standardized Centralized Alzheimer's and Related Dementias Neuroimaging (SCAN) cohort, including NACC access planning, approved-export inventory, DICOM/NIfTI staging, BIDS organization, phenotype linkage, and MRI/PET processing. Trigger on 'SCAN dataset', 'NACC SCAN', 'ADRC imaging', 'SCAN MRI', 'SCAN PET', or requests to process a local SCAN export."
license: MIT License (NeuroClaw custom skill - freely modifiable within the project)
layer: subagent
skill_type: dataset
dependencies:
  - dcm2nii
  - bids-organizer
  - smri-skill
  - fmri-skill
  - dwi-skill
  - asl-skill
  - pet-skill
  - claw-shell
---
# SCAN Skill (Dataset-Orchestration Layer)

## Overview

Use this skill to turn an approved SCAN export into an auditable multimodal workflow. Keep access control, identifier handling, staging, phenotype linkage, and modality processing explicit.

SCAN means **Standardized Centralized Alzheimer's and Related Dementias Neuroimaging**. It combines standardized MRI and PET from Alzheimer's Disease Research Centers with longitudinal NACC clinical and cognitive data.

Research use only.

## Access gate

- Request SCAN data through the [NACC Quick Access File system](https://scan.naccdata.org/).
- Access is free but is not anonymous: NACC records a project proposal and approves the request.
- Defaced images, QC fields, imaging summaries, and analysis variables are available through the request system. Availability varies because images must complete defacing and QC before release.
- Do not automate account creation, approval, or authenticated scraping. Begin execution only after the user has placed an approved export on local storage.
- Never ask the user to paste credentials into commands, logs, chat, or repository files.

## Intake contract

Before processing, identify:

- approved request or release identifier
- export root and whether images are DICOM, NIfTI, or already BIDS-like
- requested modalities and subject/session subset
- NACC clinical/UDS tables supplied with the export
- available stable identifiers and visit/date fields
- local storage, compute, and DUA restrictions

Do not assume every participant has every optional sequence. Build modality availability from the files and QC tables, not from cohort-level descriptions.

## Quick reference

| Task | Delegate to | Output |
| --- | --- | --- |
| Inventory approved export | `claw-shell` | immutable file and checksum manifest |
| Convert approved DICOM | `dcm2nii` | NIfTI, JSON, bval, and bvec files |
| Stage or validate BIDS | `bids-organizer` | BIDS dataset plus validation report |
| T1w, T2w, or FLAIR processing | `smri-skill` | structural derivatives and QC |
| Resting/task fMRI processing | `fmri-skill` | functional derivatives and QC |
| dMRI processing | `dwi-skill` | diffusion derivatives and QC |
| ASL/perfusion processing | `asl-skill` | perfusion derivatives and QC |
| Amyloid, tau, or FDG PET processing | `pet-skill` | PET derivatives, SUVRs, and QC |
| Join NACC phenotype tables | `claw-shell` with structured tabular tools | keyed analysis table and join audit |

## Core workflow

1. Inspect the approved local export without modifying source files.
2. Create a manifest containing relative path, size, checksum, inferred modality, and source package identifier.
3. Identify the subject and visit keys in both imaging and NACC tables. Preserve the source keys in a restricted crosswalk and generate separate BIDS-safe labels.
4. Convert DICOM only when needed through `dcm2nii`; otherwise retain NIfTI and sidecars as received.
5. Stage and validate the requested subset through `bids-organizer`. Record missing sidecars or acquisition metadata instead of inventing them.
6. Join phenotype and imaging records using documented keys. Report unmatched, duplicated, and many-to-many records before analysis.
7. Delegate only modalities actually present and approved for the project.
8. Save QC, provenance, exclusions, and the exact cohort query beside the derivatives.

For a request that only asks whether SCAN is obtainable, stop after explaining the NACC access gate and do not claim that data were downloaded.

## Identifier and privacy rules

- Treat NACCID, PTID, visit dates, and crosswalks as controlled data even when images are defaced.
- Keep source identifiers out of public filenames, reports, model artifacts, and example data.
- Never attempt re-identification or facial reconstruction.
- Do not redistribute source images, subject-level tables, or restricted derivatives.
- Preserve the required SCAN/NACC acknowledgements and publication terms with the project provenance.

## Expected output layout

```text
scan_output/
|-- source_manifest/
|-- bids/
|-- phenotype/
|-- smri/
|-- fmri/
|-- dwi/
|-- perf/
|-- pet/
|-- qc/
`-- logs/
```

## Completion criteria

- The approved export and governing request are identified.
- Checksums and modality inventory are saved.
- Imaging-to-phenotype joins have explicit keys and an exception report.
- BIDS validation and modality QC results are retained.
- No restricted credentials, crosswalks, or source data are committed to Git.

## References

- SCAN: https://scan.naccdata.org/
- NACC data access: https://www.naccdata.org/about-nacc-data
- Repository access matrix: `docs/DATASET_ACCESS.md`

Created At: 2026-08-11 HKT
Last Updated At: 2026-08-11 HKT
