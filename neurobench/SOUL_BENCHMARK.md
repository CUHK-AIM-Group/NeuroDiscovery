# SOUL_BENCHMARK.md - Benchmark Task Contract

Apply the following rules only for the current benchmark task response.

## Task Scope
- Complete the benchmark task directly and autonomously.
- Use the task description as the only source of instructions.
- Do not ask the user for planning, confirmation, or intermediate approval.
- Do not request the user to open files, provide extra explanations, or restate the benchmark contract.

## Output Contract
- Final answers must contain exactly two top-level sections:
  - ## Solution Thinking
  - ## Commands Or Code
- Do not include sections named Input Requirement, Constraints, Evaluation, Task Description, or any other task scaffold headings.
- Do not echo the benchmark task markdown verbatim.
- Prefer concise, actionable commands and code snippets over narrative.

## Execution Contract
- If the task is quick and does not require file input/output, execute it autonomously end-to-end.
- If the task requires input files such as imaging, genomics, or other datasets, do not ask the user to confirm anything.
- If the task requires output files, do not generate the actual data artifacts unless the task explicitly expects them to be written.
- For file I/O heavy tasks, provide the exact commands or code that should be used.
- For benchmark tasks, do not read files first unless the task description explicitly requires it and the benchmark mode policy allows it.

## Reporting Contract
- Save benchmark reports to the configured output folder only.
- Record the reasoning, tools used, and command/code snippets.
- Keep results deterministic and reproducible.
