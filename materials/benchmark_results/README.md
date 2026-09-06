# Benchmark result artifacts

This directory contains selected historical NeuroClaw benchmark results. The [current task registry](../../neurobench/task_atlas.json) contains **500 task definitions**, but each result file covers the task set evaluated in its own run.

## Available evaluation

| Artifact | Scope |
| --- | --- |
| [benchmark_leaderboard_gpt-5.4_20260421_235200.md](benchmark_leaderboard_gpt-5.4_20260421_235200.md) | Historical leaderboard dated 2026-04-21; its metadata reports **100 scored cases** and identifies GPT-5.4 as the scorer. |
| [benchmark_scores_gpt-5.4_20260421_235200.json](benchmark_scores_gpt-5.4_20260421_235200.json) | Machine-readable score artifact accompanying that evaluation. |

The model name in these filenames identifies the scorer. The leaderboard includes multiple evaluated models and conditions. These files are retained as historical results and do not constitute the complete 500-task NeuroDiscovery manuscript evaluation.

## Interpreting scores

The report-scoring rubric evaluates planning completeness, tool/skill appropriateness and command/code correctness. A workflow score should be interpreted together with the corresponding execution logs, generated artifacts and verification records. It is not, by itself, a measure of completed raw-data analyses or scientific validity.

Compare runs only after checking their task IDs, model versions, execution conditions, scoring rubric and handling of failed or missing tasks. A change in the task registry does not change the coverage of an archived run.

## NeuroDiscovery manuscript results

The complete paper-specific evaluation bundle is being prepared for a versioned release. It will be linked from the [main repository README](../../README.md#neurodiscovery-and-research-materials) with its release identifier and artifact manifest. New results will be published with their own run metadata rather than replacing the historical files above.
