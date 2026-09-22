"""Run the blinded Case Study 1 execution task through Biomni's native A1 agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle-public", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api-key-env", default="BIOMNI_API_KEY")
    return parser.parse_args()


def prompt(task_dir: Path) -> str:
    return f"""Run a blinded native-execution reliability test for Case Study 1.

The only task inputs are in `{task_dir}`. Read `TASK.md` and
`task_manifest.json`, then use your native Python REPL to independently execute
the specified analysis on `tcp_subject_level_features.csv` for all registered
candidates. Do not search for or inspect hidden, reference, exhaustive, or
precomputed aggregate result files. Do not omit negative or null results.

Your final answer must be one `<solution>` tag containing only a valid JSON
object with outer key `results`, exactly one result per registered candidate,
and no prose inside the tag. Do not estimate values mentally: run the code.
"""


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable {args.api_key_env}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.out_dir / "workspace"
    task_dir = workspace / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TASK.md", "task_manifest.json", "tcp_subject_level_features.csv"):
        shutil.copy2(args.bundle_public / name, task_dir / name)

    sys.path.insert(0, str(args.repo.resolve()))
    from biomni.agent import A1
    from biomni.tool import support_tools

    support_tools._persistent_namespace.clear()
    os.chdir(workspace)
    preflight = support_tools.run_python_repl(
        "from pathlib import Path; "
        f"p=Path({str(task_dir.resolve())!r}); "
        "print((p/'TASK.md').read_text(encoding='utf-8').splitlines()[0]); "
        "print((p/'task_manifest.json').stat().st_size); "
        "print((p/'tcp_subject_level_features.csv').stat().st_size)"
    )
    (args.out_dir / "repl_preflight.txt").write_text(preflight, encoding="utf-8")

    started = time.perf_counter()
    summary: dict[str, object] = {
        "framework": "biomni_native",
        "model": args.model,
        "base_url": args.base_url,
        "workflow_returned": False,
        "human_repairs": 0,
    }
    try:
        agent = A1(
            path=str(workspace),
            llm=args.model,
            source="OpenAI",
            use_tool_retriever=False,
            base_url=args.base_url,
            api_key=api_key,
            expected_data_lake_files=[],
        )
        log, final = agent.go(prompt(task_dir.resolve()))
        (args.out_dir / "agent_log.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (args.out_dir / "final.txt").write_text(str(final), encoding="utf-8")
        summary["workflow_returned"] = True
        print(final)
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        (args.out_dir / "exception.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        summary["duration_seconds"] = time.perf_counter() - started
        (args.out_dir / "runner_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
