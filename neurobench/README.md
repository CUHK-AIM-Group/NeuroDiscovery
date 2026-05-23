# NeuroBench

NeuroBench is the benchmark part of NeuroClaw for neuroscience workflow evaluation.
It focuses on whether an agent can complete real neuroimaging workflows end-to-end: organize raw data, run preprocessing pipelines, produce analysis outputs, and keep results reproducible.

Current status: 120 tasks (T01-T120) covering data orchestration, single-tool execution, multi-step pipelines, dev environment, research tooling, and **model training / evaluation**. These are *engineering* benchmarks - does the agent run the right tool / model with the right configuration? The full registry mapping every task to its category lives in [`task_atlas.json`](task_atlas.json).

## Operational benchmarks (T01-T120)

The 120 tasks are organised into seven categories:

| Category | Count | What it tests |
|---|---:|---|
| `data_orchestration` | 7 | BIDS organisation, dataset staging, format conversion (DICOM->NIfTI, downloads) |
| `tool_use` | 68 | Single-tool calls - DIPY metric, FSL extraction, FreeSurfer command, Nilearn function, etc. |
| `pipeline_execution` | 19 | End-to-end pipelines (fMRIPrep, HCP full, ADNI end-to-end, multi-modal full) |
| `dev_environment` | 4 | Conda envs, git workflows, dependency planning, Overleaf tooling |
| `research_tooling` | 2 | Literature search, multi-engine retrieval |
| `model_training` | 17 | Train + evaluate a brain model (FC / ROI time-series / voxel) on shared HCP-age + ABIDE-dx settings |
| `cross_model_evaluation` | 3 | Multi-atlas sweep, cross-dataset generalisation with harmonization, site-stratified vs leave-site-out |

Historical numbering (T01-T100) is preserved for backwards compatibility with prior leaderboard runs; the categorisation is overlaid via `task_atlas.json` rather than by moving directories. The model-training and cross-model-evaluation categories (T101-T120) were added 2026-05-24.

Original family-by-pipeline grouping:
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
- **T101-T117**: Model training and evaluation (BrainGNN, BNT, BrainNetCNN, IBGNN, LGGNN, ComBrainTF, Ridge baseline, STAGIN, BolT, FBNetGen, Brain Graphormer, Hierarchical GNN, SpaceNet, ROI-MLP, ROI-LSTM, NeuroSTORM, SwiFT)
- **T118-T120**: Cross-model / cross-dataset evaluation (multi-atlas sweep, cross-dataset generalisation with harmonization, site-stratified vs leave-site-out)

## Task Structure

Each task directory includes:
- `task.md`: objective, input, output, and key steps

In practice, `task.md` is the instruction file for evaluation. It defines:
- Required input assumptions (file type, folder organization, mandatory metadata)
- Processing objective and expected pipeline behavior
- Expected outputs and naming conventions
- Important checks to verify task completion quality

## Benchmark Usage

You can run NeuroClaw benchmark tasks in two ways:

NeuroBench accepts the following benchmark configurations:
- `with-skills`: the agent may use loaded skills from `skills/`
- `no-skills`: baseline run with skills disabled
- paired comparison: `--benchmark-compare-skills` runs both variants for the same task set

Benchmark scoring is handled separately with `--score-benchmark`. It reads reports in `output/`, applies a GPT-5.4 weighted rubric, and generates numeric scores for planning completeness, tool/skill reasonableness, and command/code correctness. For fairness, each task case is jointly scored across all comparable models in one batch to reduce scoring-standard drift. Skill-call counts are tracked separately for efficiency analysis.

To score existing benchmark reports:
```bash
python core/agent/main.py --score-benchmark
```

To speed up scoring on larger runs:
```bash
python core/agent/main.py --score-benchmark --score-workers 8
```

### Web benchmark mode
```bash
python core/agent/main.py --web --benchmark
```

### CLI benchmark batch mode
```bash
python core/agent/main.py --benchmark
```

Paired skill comparison in CLI mode:
```bash
python core/agent/main.py --benchmark --benchmark-compare-skills
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

Each report includes the solution thinking, the skills used, skill-call counts, and the commands or code that were used or suggested.


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
