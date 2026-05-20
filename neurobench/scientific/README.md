# Scientific benchmarks (atom-algebra layer)

Each subdirectory corresponds to one canonical research task from the atom
algebra (`neurooracle.src.atoms.CANONICAL_TASKS`). Drop a `task.md`
into the appropriate subdirectory to register a hypothesis-generation
benchmark instance.

## Conventions

- Subdirectory name = canonical task name (snake_case).
- Each instance lives in `<task_name>/<instance_id>/task.md`, e.g.
  `biomarker_discovery/B01_adni_hippocampal_volume_ad/task.md`.
- The instance's `task.md` should declare:
  - which dataset(s) the benchmark uses,
  - the input atoms it wires up (which KG nodes serve as each input atom),
  - the expected output atom (the prediction target),
  - quality criteria for the generated hypothesis.

## Difference vs operational layer

- **Operational** (`neurobench/T01-T100`): does the agent run the *tool* correctly?
- **Scientific** (this directory): does the agent generate a *hypothesis* whose
  atom-shape matches the canonical task and whose evidence chain is sound?

The two layers are scored independently and contribute different signals
to the leaderboard.

See `neurobench/task_atlas.json` for the registry mapping every benchmark
instance to one of the two layers.
