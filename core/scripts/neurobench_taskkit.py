from __future__ import annotations

"""Shared helpers for NeuroBench task definition modules.

`body()` renders the standard task.md sections in the established style
(Task Description / Input Requirement / Constraints / Expected Output /
Evaluation). Category modules build TASKS lists of
(number, slug, category, title, body) tuples consumed by
generate_neurobench_tasks.py.
"""

from typing import List, Optional


def body(
    folder: str,
    desc: str,
    inputs: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    evaluation: Optional[List[str]] = None,
    save_to: Optional[str] = None,
) -> str:
    """Render a standard task.md body (without the H1 title line)."""
    parts: List[str] = []
    parts.append(f"## Task Description\n\n{desc.strip()}\n")

    parts.append("## Input Requirement\n")
    if inputs:
        parts.append("Required input(s):\n")
        parts.extend(f"- {i}\n" for i in inputs)
        parts.append(
            "\nIf any required input is missing, return:\n\n- Missing required input\n"
        )
    else:
        parts.append("\n- No interactive input.\n")

    cons = list(constraints or [])
    if save_to:
        cons.append(f"Save all generated artifacts to:\n  - {save_to}")
    if cons:
        parts.append("\n## Constraints\n")
        parts.extend(f"- {c}\n" for c in cons)

    if outputs:
        parts.append("\n## Expected Output\n\nExpected output artifact(s):\n")
        parts.extend(f"- {o}\n" for o in outputs)
        parts.append("\nRecommended metadata file:\n\n- result_YYYYMMDD_HHMMSS.json\n")

    parts.append("\n## Evaluation\n")
    if evaluation:
        parts.extend(f"- {e}\n" for e in evaluation)
    else:
        parts.append("- This test case is manually evaluated.\n")

    return "\n".join(parts)


def std_eval(folder: str) -> str:
    return f"benchmark_results/{folder}/"
