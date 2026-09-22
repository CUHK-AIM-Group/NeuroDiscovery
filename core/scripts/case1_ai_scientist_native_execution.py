"""Run one blinded native BFTS draft for the Case Study 1 execution task."""

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
    parser.add_argument("--max-steps", type=int, default=3)
    return parser.parse_args()


def idea_payload() -> dict[str, object]:
    return {
        "Name": "case1_blinded_native_execution",
        "Title": "Blinded TCP candidate-level statistical execution",
        "Abstract": (
            "Execute every registered Case Study 1 analysis exactly as defined "
            "in input/TASK.md and input/task_manifest.json using only "
            "input/tcp_subject_level_features.csv. Do not create synthetic "
            "data, train a predictive model, search externally, or inspect "
            "hidden/reference/exhaustive results. Write the exact required JSON "
            "object to working/native_results.json and print it, with one finite "
            "result for every candidate including null and negative findings."
        ),
        "Short Hypothesis": (
            "A correct self-contained program can reproduce all registered OLS "
            "and covariate-residualized Cohen d results from the public inputs; "
            "task completion is exact JSON generation rather than model training."
        ),
        "Experiments": [
            "Read only input/TASK.md, input/task_manifest.json, and "
            "input/tcp_subject_level_features.csv.",
            "Follow the analysis contract exactly for every registered candidate.",
            "Write the exact required JSON object to working/native_results.json "
            "and print the same JSON. Do not omit null or negative findings.",
            "Do not train a predictive model, create synthetic data, search for "
            "literature, or inspect any hidden/reference/exhaustive result file.",
        ],
        "Risk Factors and Limitations": [
            "Candidate identifiers and result field names must match exactly.",
            "The task is execution reliability, not model performance optimization.",
        ],
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TASK.md", "task_manifest.json", "tcp_subject_level_features.csv"):
        shutil.copy2(args.bundle_public / name, data_dir / name)

    idea_path = args.out_dir / "idea.json"
    idea_path.write_text(
        json.dumps(idea_payload(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sys.path.insert(0, str(args.repo.resolve()))
    from omegaconf import OmegaConf
    from ai_scientist.treesearch.agent_manager import AgentManager
    from ai_scientist.treesearch.utils.config import (
        load_cfg,
        load_task_desc,
        prep_agent_workspace,
        save_run,
    )

    raw_cfg = OmegaConf.load(args.repo / "bfts_config.yaml")
    raw_cfg.data_dir = str(data_dir.resolve())
    raw_cfg.desc_file = str(idea_path.resolve())
    raw_cfg.goal = None
    raw_cfg.eval = None
    raw_cfg.log_dir = str((args.out_dir / "logs").resolve())
    raw_cfg.workspace_dir = str((args.out_dir / "workspaces").resolve())
    raw_cfg.exp_name = "native-execution"
    raw_cfg.generate_report = False
    raw_cfg.agent.num_workers = 1
    raw_cfg.agent.steps = 1
    raw_cfg.agent.stages.stage1_max_iters = 1
    raw_cfg.agent.stages.stage2_max_iters = 1
    raw_cfg.agent.stages.stage3_max_iters = 1
    raw_cfg.agent.stages.stage4_max_iters = 1
    raw_cfg.agent.search.num_drafts = 1
    raw_cfg.agent.multi_seed_eval.num_seeds = 1
    raw_cfg.agent.code.model = args.model
    raw_cfg.agent.feedback.model = args.model
    raw_cfg.agent.vlm_feedback.model = args.model
    config_path = args.out_dir / "bfts_native_config.yaml"
    OmegaConf.save(raw_cfg, config_path)

    started = time.perf_counter()
    summary: dict[str, object] = {
        "framework": "ai_scientist_v2_native",
        "model": args.model,
        "workflow_returned": False,
        "human_repairs": 0,
    }
    try:
        cfg = load_cfg(config_path)
        task_desc = load_task_desc(cfg)
        prep_agent_workspace(cfg)
        manager = AgentManager(
            task_desc=task_desc,
            cfg=cfg,
            workspace_dir=Path(cfg.workspace_dir),
        )
        stage = manager.current_stage
        if stage is None:
            raise RuntimeError("AI Scientist-v2 created no initial BFTS stage")
        with manager._create_agent_for_stage(stage) as agent:
            agent.evaluation_metrics = (
                "completion_rate: 1.0 only when all registered candidates are "
                "computed and working/native_results.json is valid JSON; higher is better"
            )
            for _step in range(max(1, args.max_steps)):
                agent.step(lambda *unused_args, **unused_kwargs: None)
                result_files = sorted(
                    Path(cfg.workspace_dir).rglob("native_results.json")
                )
                if result_files and any(
                    node.is_buggy is False for node in manager.journals[stage.name].nodes
                ):
                    break

        journal = manager.journals[stage.name]
        save_run(cfg, journal, stage_name=stage.name)
        node_rows = []
        terminal_outputs = []
        for node in journal.nodes:
            terminal = node.term_out if node._term_out is not None else ""
            terminal_outputs.append(terminal)
            node_rows.append(
                {
                    "id": node.id,
                    "is_buggy": bool(node.is_buggy),
                    "exc_type": node.exc_type,
                    "exec_time": node.exec_time,
                }
            )
            (args.out_dir / f"node_{node.id}.py").write_text(
                node.code, encoding="utf-8"
            )
        (args.out_dir / "terminal_outputs.txt").write_text(
            "\n\n".join(terminal_outputs), encoding="utf-8"
        )
        result_files = sorted(Path(cfg.workspace_dir).rglob("native_results.json"))
        if result_files:
            shutil.copy2(result_files[-1], args.out_dir / "final.txt")
        else:
            (args.out_dir / "final.txt").write_text(
                "\n\n".join(terminal_outputs), encoding="utf-8"
            )
        summary.update(
            {
                "workflow_returned": True,
                "generated_nodes": len(node_rows),
                "nonbuggy_nodes": sum(not row["is_buggy"] for row in node_rows),
                "native_result_files": [str(path) for path in result_files],
                "nodes": node_rows,
            }
        )
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
