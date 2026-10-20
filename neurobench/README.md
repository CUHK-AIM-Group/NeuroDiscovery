# Neuroimaging execution tasks

This directory contains **500 neuroimaging execution tasks**, numbered **T01–T500**, for evaluating workflow planning, tool use, data processing and model execution. The NeuroDiscovery manuscript uses this collection to evaluate **NeuroRuntime**, its execution platform. **NeuroBench** remains the name used by the existing NeuroDiscovery benchmark interface and directory.

The complete mapping of task directories to categories is in [`task_atlas.json`](task_atlas.json). Task definitions specify the inputs, outputs and checks needed to evaluate a workflow. Evaluation outputs are versioned separately from the task registry.

These execution tasks are distinct from the manuscript's nine scientific autoresearch tasks and its temporal evaluation of hypothesis generation (hindcasting).

## Directory layout

- `tasks/`: the 500 task definitions; task IDs and contents are unchanged.
- `runner/`: campaign drivers, scoring and offline execution qualification.
- `tests/`: harness regression tests, run with `python -m pytest --layer benchmark`.
- `task_atlas.json`: category mapping shared by the task registry.

Run the campaign CLI with `python -m neurobench.runner.campaign --help`.

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

```bash
python -m neurobench.runner.campaign --help
python -m neurobench.runner.codex500 --help
python -m neurobench.runner.scoring --output-root dev/outputs/selected-run --workers 8
python -m pytest --layer benchmark
```

`scoring.py` owns shared task/report parsing, the existing weighted scoring rubric and leaderboard generation. Report parsing requires no agent initialization. Actual model grading uses the configured backend; historical reports and frozen execution snapshots remain unchanged.

The Web benchmark response mode remains available with `python core/agent/main.py --web --benchmark`. Batch execution and scoring use the modules above.

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

Eight executable research-delivery development cases have been tested with GPT-6.1-Sol and DeepSeek V4.1 Flash under one NR OFF/ON comparison. They are separate from the 500-task proposal registry. The [twelve-case design](designs/execution_tasks.json) includes four remaining cases that are not execution-qualified. See [the current engineering guide, section 17.8](../docs/NEURORUNTIME_ENGINEERING.md#178-首批困难题的双模型开发测试) for inputs, actual budgets, original verdicts, supplemental public-protocol review, known evaluator defects and limitations. Local evidence is under `dev/runs/benchmark_hard_trial_20261020_r1/`; do not resume its frozen controller or resend failed/unknown inputs.

Next additions will focus on **Model Executability and Reproducibility** tasks.

Planned direction includes:
- Model run tasks that verify environment setup, inference execution, and output integrity
- Reproducibility checks across repeated runs with stable parameters
- Stronger QC-oriented tasks for automated anomaly and failure detection
