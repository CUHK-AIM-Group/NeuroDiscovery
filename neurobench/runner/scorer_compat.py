"""Audit NeuroBench task and report compatibility using the shared scorer."""
from __future__ import annotations

from neurobench.runner import scoring
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]




def audit(benchmark_root: Path, output_root: Path | None = None) -> dict:
    """Return a compatibility report over the campaign tree."""
    module = scoring
    contracts = module._discover_task_contracts(benchmark_root)
    report: dict = {
        "task_count": len(contracts),
        "task_ids": sorted(contracts.keys(), key=lambda value: (len(value), value)),
        "expected_500": len(contracts) == 500,
    }
    if output_root is not None and output_root.exists():
        index = module._discover_benchmark_reports(output_root)
        report["models"] = {
            model: {
                "cases": len(case_map),
                "sample_case_ids": sorted(case_map.keys())[:5],
                "missing_from_task_set": sorted(set(case_map) - set(contracts))[:10],
            }
            for model, case_map in sorted(index.items())
        }
        report["model_count"] = len(index)
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Scorer compatibility audit",
        "",
        f"- Benchmark tasks discovered by the scorer: {report['task_count']}",
        f"- All 500 ids present: {report['expected_500']}",
    ]
    if "models" in report:
        lines += ["", f"- Report models discovered: {report['model_count']}", ""]
        for model, info in report["models"].items():
            lines.append(f"  - {model}: cases={info['cases']}, sample={info['sample_case_ids']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check scorer compatibility of a campaign tree")
    parser.add_argument("--benchmark-root", default=str(REPO_ROOT / "neurobench" / "tasks"))
    parser.add_argument("--output-root", default="")
    args = parser.parse_args(argv)
    report = audit(
        Path(args.benchmark_root),
        Path(args.output_root) if args.output_root else None,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "task_ids"}, ensure_ascii=False, indent=2))
    if not report["expected_500"]:
        print("FAIL: scorer does not discover all 500 tasks")
        return 1
    print("OK: scorer discovers all 500 tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
