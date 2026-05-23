# T116_neurostorm_train_eval: NeuroSTORM (Voxel 4D) Training and Evaluation

## Task Description

Train and evaluate NeuroSTORM - a voxel-level 4D foundation/transformer model for fMRI - on HCP age regression and ABIDE classification. Voxel-based, included as one of two voxel benchmarks alongside SwiFT.

## Input Requirement

Required input(s):

- Preprocessed 4D BOLD NIfTI per subject (fMRIPrep / HCP pipeline output)
- Brain mask
- Subject list and labels CSV
- Pretrained NeuroSTORM weights (if using pretrain-then-finetune protocol)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `skills/neurostorm/` for model loading + fine-tuning recipe.
- 5-fold deterministic split shared with T101 (where subjects overlap).
- Mixed-precision training; gradient checkpointing required if memory < 40 GB.
- Save artefacts under `models/benchmark_results/T116_neurostorm/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- GPU memory + wall-time log
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age MAE should be competitive with BNT (T102) given enough pretraining.
- Document any deviation from default recipe; this is a heavy-compute benchmark.
