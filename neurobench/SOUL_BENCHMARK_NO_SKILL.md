# SOUL_BENCHMARK_NO_SKILL.md - Benchmark Baseline Contract (No Skills)

Apply the following rules only for the current benchmark baseline response.

## Task Scope
- Complete the benchmark task directly and autonomously.
- Use the task description as the only source of instructions.
- Do not ask the user for planning, confirmation, or intermediate approval.

## Baseline Contract
- This run is a baseline without skills.
- Skill library access is strictly forbidden.
- Do not access any skill files, skill metadata, or loaded skill lists.
- Do not reference or infer from skill names, skill summaries, or skill dependency relations.
- Do not call tools, shells, Python execution, Docker, or external skill handlers.
- Do not rely on filesystem operations or runtime probes.
- Provide reasoning and command/code suggestions only.
- If a tool name appears in the task text, treat it as plain task context rather than as a skill invocation.

## Output Contract
- Final answers must contain exactly two top-level sections:
  - ## Solution Thinking
  - ## Commands Or Code
- Do not include sections named Input Requirement, Constraints, Evaluation, Task Description, or other scaffold headings.
- Do not echo the benchmark task markdown verbatim.

## Reporting Contract
- Keep results deterministic and reproducible.
- Ensure the answer is complete in one turn.
