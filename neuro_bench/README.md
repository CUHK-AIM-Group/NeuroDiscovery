# NeuroBench

NeuroBench is the benchmark part of NeuroClaw for neuroscience workflow evaluation.
It focuses on whether an agent can complete real neuroimaging workflows end-to-end: organize raw data, run preprocessing pipelines, produce analysis outputs, and keep results reproducible.

Current status:
- 100 tasks with continuous IDs (T01-T100)
- Coverage from utility-level setup to multi-modal integrated pipelines
- Task design centered on practical deliverables (processed images, ROI tables, connectomes, QC artifacts)

## What's Included

The benchmark is grouped by workflow families so you can evaluate specific capabilities or run larger scenario bundles.

- **T01-T09**: Data organization, BIDS conversion, environment and utility tasks
- **T10-T14**: Basic DWI pipeline (load, mask, tensor fit, metrics, ROI)
- **T15-T20**: FreeSurfer-focused structural tasks
- **T21-T33**: Core FSL tasks (structural, functional, diffusion)
- **T34-T47**: Core HCPPipeline-style stages
- **T48-T54**: Nilearn ROI/connectivity/GLM tasks
- **T55-T61**: Extended DWI pipeline (QSIPrep, tractography, connectome)
- **T62-T72**: General multimodal workflows (BIDS, fMRIPrep, FEAT, CONN, EEG, WMH)
- **T73-T80**: Advanced fMRI workflows (XCP-D, FC/EC, first/group GLM)
- **T81-T89**: sMRI workflows (BIDS, FSL, FreeSurfer, fMRIPrep anat, ROI)
- **T90-T94**: ADNI workflows
- **T95-T100**: HCP dataset workflows (download, staging, sMRI/fMRI/DWI, full multimodal)

## Task Structure

Each task directory includes:
- `task.md`: objective, input, output, and key steps

In practice, `task.md` is the contract for evaluation. It defines:
- Required input assumptions (file type, folder organization, mandatory metadata)
- Processing objective and expected pipeline behavior
- Expected outputs and naming conventions
- Important checks to verify task completion quality

## Benchmark Usage

You can run NeuroClaw benchmark tasks in two ways:

### Web benchmark mode
```bash
python core/agent/main.py --web --benchmark
```

### CLI benchmark batch mode
```bash
python core/agent/main.py --benchmark
```

In CLI benchmark mode, NeuroClaw will ask for:
- the benchmark directory path
- the model name to evaluate

Then it will:
- recursively read every `task.md` under the selected benchmark directory
- sort tasks alphabetically by task folder name
- execute each task in sequence without asking for intermediate confirmation
- show progress in the terminal only
- save one report per task to `output/<case_id>_<model_name>.md`

Each report includes the solution thinking, the tools used, and the commands or code that were used or suggested.


Example:
```bash
cat T73_xcpd_denoising/task.md
```

## Scope

Modalities:
- sMRI
- fMRI
- DWI
- EEG

Coverage intent:
- Support both single-modality evaluation and cross-modality orchestration
- Include both preprocessing-oriented and analysis-oriented tasks
- Keep outputs suitable for downstream model training/inference experiments

Datasets/workflows covered:
- ADNI
- HCP
- BIDS-style generic workflows

Evaluation emphasis across scope:
- Executability: can the workflow be completed end-to-end
- Output validity: do generated artifacts match expected format/content
- Reproducibility readiness: are logs, parameters, and outputs auditable

## Next

Next additions will focus on **Model Executability and Reproducibility** tasks.

Planned direction includes:
- Model run tasks that verify environment setup, inference execution, and output integrity
- Reproducibility checks across repeated runs with stable parameters
- Stronger QC-oriented tasks for automated anomaly and failure detection
