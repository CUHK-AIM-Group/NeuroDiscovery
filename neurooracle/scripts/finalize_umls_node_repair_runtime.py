"""Freeze a focused runtime correction without rewriting verified graph data.

The first acceptance disclosed CT -> ALFF through an interior substring. Keep
that record immutable, tighten the gate, and bind a superseding acceptance to
the same deeply verified graph plus the corrected implementation.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from build_umls_node_repair_candidate import (
    BASE_DIR, DEFAULT_OUTPUT, REPO, atomic_json, cheap, check_baselines,
    compact, evidence_scores, file_sha, read_json, utc_now,
)
from validate_umls_node_repair_candidate import endpoint_and_alias_regression, report


EXPECTED_CHANGED_IMPLEMENTATION = {
    "neurooracle/src/node_alias_safety.py",
    "neurooracle/src/hypothesis_engine.py",
    "neurooracle/scripts/validate_umls_node_repair_candidate.py",
    "neurooracle/tests/test_umls_node_repair.py",
}


def check_frozen_data(frozen: dict) -> None:
    if check_baselines() != frozen["baseline"]:
        raise ValueError("protected formal/base/review inputs changed")
    for item in frozen["artifacts"].values():
        actual = cheap(Path(item["path"]))
        if any(actual[key] != item[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError(f"frozen repair artifact changed: {item['path']}")


def finalize(candidate: Path, output: Path) -> dict:
    manifest_path = output / "RUNTIME_ACCEPTANCE.json"
    if manifest_path.exists() or (candidate / "CURRENT_ACCEPTANCE.json").exists():
        raise ValueError("a superseding acceptance already exists")
    source_path = candidate / "REPAIR_FREEZE.json"
    source = read_json(source_path)
    if source["status"] != "REPAIRED_CANDIDATE_VALIDATED_NOT_APPLIED":
        raise ValueError("repair data has no completed deep validation")
    source_fp = {**cheap(source_path), "sha256": file_sha(source_path)}
    check_frozen_data(source)
    implementation = {}
    changed = set()
    for name, previous in source["implementation"].items():
        path = REPO / name
        current = {**cheap(path), "sha256": file_sha(path)}
        implementation[name] = current
        if current["sha256"] != previous["sha256"]:
            changed.add(name)
    if changed != EXPECTED_CHANGED_IMPLEMENTATION:
        raise ValueError(f"unexpected implementation changes: {changed}")

    build = read_json(candidate / "BUILD_COMPLETE.json")
    endpoint = endpoint_and_alias_regression(candidate, build, artifact_output=output)
    old_endpoint = source["experiment_regression"]["endpoint_and_alias_regression"]
    if endpoint["endpoint_changes_artifact"]["sha256"] != old_endpoint["endpoint_changes_artifact"]["sha256"]:
        raise ValueError("raw claim semantic endpoint effects changed unexpectedly")
    if endpoint["reviewed_old_nodes_now_matching_mrsty"] != 1285:
        raise ValueError("semantic-type reconciliation gate failed")
    scores = evidence_scores(read_json(candidate / "EVIDENCE_FIXTURE_REPAIRED.json"))
    if scores != read_json(BASE_DIR / "EXPERIMENT_REGRESSION.json")["evidence_scores"]:
        raise ValueError("fixed evidence scores changed under corrected runtime")
    atomic_json(output / "FIXED_EVIDENCE_SCORES.json", scores)
    regression = deepcopy(source["experiment_regression"])
    regression["endpoint_and_alias_regression"] = endpoint
    regression["runtime_alias_safety_rechecked"] = True
    regression["fixed_claim_scores_recomputed"] = len(scores)
    regression["kge_validation_reused_from"] = source_fp
    regression["reuse_reason"] = (
        "Same logical turn; all graph/detail artifacts retain their frozen fingerprints. "
        "Only resolver safeguards, their tests and the validation gate changed. "
        "The KGE loader does not use resolve_name; its complete input/split evidence is unchanged."
    )
    atomic_json(output / "EXPERIMENT_REGRESSION.json", regression)

    suites = list(ET.parse(output / "TEST_RESULTS.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(suite.get(key, "0")) for suite in suites)
             for key in ("tests", "errors", "failures", "skipped")}
    if tests["tests"] != 65 or any(tests[key] for key in ("errors", "failures", "skipped")):
        raise ValueError(f"strengthened test gate failed: {tests}")
    text = report(build, source["validation"], regression, tests)
    text += f"""
## 最终运行时复核（本报告为当前验收入口）

父目录的首次验收明细暴露了CT被ALFF英文名称中的内部字母误命中的问题；该记录保持原样，不能单独作为当前代码的验收依据。本次限制无上下文CT的模糊子串回退，精确名称/别名仍可匹配独立实体，并将测试门槛改为不支持的缩写不得命中任何无关节点。

在同时包含ALFF和皮层厚度的旧/新记录夹具中，CT、CT imaging、fALFF、dynamic fALFF均为未解析，ALFF、CortThick和cortical thickness仍命中正确节点；另有测试确认独立Computed Tomography实体的精确CT别名可解析。

本次重新计算209条固定claim评分及22,017个相关端点，结果符合预期；3条fALFF端点不再使用ALFF的规范身份，原始claim不变。数据主图和附属库沿用{source['frozen_at']}的全量读取、哈希及KGE回归证据，并确认全部原冻结文件指纹不变，没有重写大文件。

本目录的RUNTIME_ACCEPTANCE.json绑定父目录REPAIR_FREEZE.json、修正后的实现文件与65项测试结果。父目录CURRENT_ACCEPTANCE.json指向本次验收。正式应用仍未执行。
"""
    (output / "CANDIDATE_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    check_frozen_data(source)
    for name, previous in implementation.items():
        if file_sha(REPO / name) != previous["sha256"]:
            raise ValueError("implementation changed during runtime validation")
    own_path = Path(__file__).resolve()
    implementation[str(own_path.relative_to(REPO)).replace("\\", "/")] = {
        **cheap(own_path), "sha256": file_sha(own_path),
    }
    artifacts = {}
    for name in ("CANDIDATE_REPORT.md", "EXPERIMENT_REGRESSION.json", "FIXED_EVIDENCE_SCORES.json",
                 "ALIAS_ENDPOINT_VIEW_CHANGES.jsonl", "TEST_RESULTS.xml"):
        path = output / name
        artifacts[name] = {**cheap(path), "sha256": file_sha(path)}
    result = {
        "schema_version": "umls.node_metadata_alias_repair.runtime.v2",
        "status": "REPAIRED_CANDIDATE_RUNTIME_VERIFIED_NOT_APPLIED",
        "frozen_at": utc_now(), "data_freeze": source_fp,
        "counts": source["counts"], "data_artifacts": source["artifacts"],
        "baseline": source["baseline"], "artifacts": artifacts,
        "implementation": implementation, "tests": tests, "experiment_regression": regression,
        "graph_or_details_rewritten": False, "formal_apply_performed": False,
        "supersedes_prior_runtime_acceptance": True,
    }
    atomic_json(manifest_path, result)
    atomic_json(candidate / "CURRENT_ACCEPTANCE.json", {
        "status": result["status"], "updated_at": result["frozen_at"],
        "acceptance": {**cheap(manifest_path), "sha256": file_sha(manifest_path)},
        "report_path": str(output / "CANDIDATE_REPORT.md"),
    })
    print(compact({"status": result["status"], "tests": tests,
                   "resolver": endpoint["actual_resolver_results"], "formal_applied": False}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    output = candidate / "runtime_alias_hardening_v2_20260906"
    output.mkdir(exist_ok=True)
    finalize(candidate, output)


if __name__ == "__main__":
    main()
