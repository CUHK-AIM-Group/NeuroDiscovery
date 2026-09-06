# Neuroimaging execution tasks

This directory contains **500 neuroimaging execution tasks**, numbered **T01–T500**, for evaluating workflow planning, tool use, data processing and model execution. The NeuroDiscovery manuscript describes this collection as the 500 neuroimaging execution tasks. **NeuroBench** remains the name used by the existing NeuroClaw benchmark interface and directory.

The complete mapping of task directories to categories is in [`task_atlas.json`](task_atlas.json). Task definitions specify the inputs, outputs and checks needed to evaluate a workflow. Evaluation outputs are versioned separately from the task registry.

## Task registry (T01–T500)

All 500 tasks are assigned to seven categories:

| Category | Count | What it tests |
|---|---:|---|
| `data_orchestration` | 75 | Dataset staging, BIDS organisation, format conversion and metadata checks |
| `tool_use` | 80 | Single-tool execution with neuroimaging software and libraries |
| `pipeline_execution` | 75 | Multi-step preprocessing and analysis workflows |
| `dev_environment` | 70 | Environment setup, dependencies, containers and computing infrastructure |
| `research_tooling` | 70 | Literature retrieval, evidence handling and research reporting |
| `model_training` | 70 | Model training, validation and evaluation |
| `cross_model_evaluation` | 60 | Comparisons across models, atlases, sites and datasets |
| **Total** | **500** | |

Task identifiers remain stable as the registry expands. Category membership is defined by `task_atlas.json`, rather than by contiguous task-number ranges.

## Task definitions and published results

The 500 task definitions are included in the repository. The [benchmark results directory](../materials/benchmark_results/README.md) currently contains selected historical evaluation artifacts. Each artifact has its own task coverage and evaluation date; an older leaderboard does not become a 500-task result when the registry expands. The complete NeuroDiscovery manuscript evaluation bundle is being prepared for a separate versioned release.

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

These rubric scores describe the evaluated reports. Execution completion, artifact validity, numerical correctness and reproducibility must be established from the corresponding run logs, outputs and verification records. The scoring rubric alone does not establish those execution outcomes.

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
