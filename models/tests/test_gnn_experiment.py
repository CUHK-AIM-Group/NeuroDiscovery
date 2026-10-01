"""Synthetic orchestration/contract checks; no experimental optimizer runs."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from models.kg_link_prediction import experiment as experiment
from models.kg_link_prediction.run_guard import BudgetExceeded, RunGuard
from models.kg_link_prediction.validation import ValidationError


def fixture_bundle():
    sources, items, edges, assignments = {}, [], [], {}

    def row(key, source, target, label, query):
        text = "Synthetic evidence fixture: " + key
        digest = hashlib.sha256(text.encode()).hexdigest()
        sources[digest] = dict(text=text, work_key=key, source_family=key, fold="corpus")
        return dict(item_id=key, query_id=query, source_id=source, target_id=target,
                    relation="rel", relation_family="rel", fact_family=key,
                    measurement="endpoint", role="primary_result", scope={"species": "synthetic"},
                    rationale="test only", mapping_status="reviewed_exact", scope_match=True,
                    previously_exposed=False, work_keys=[key], source_families=[key], label=label,
                    anchors=[dict(source_sha256=digest, start=0, end=len(text), quote=text)])

    for fold in ("train", "validation", "test"):
        for offset, label in enumerate(("supported", "unresolved")):
            key = fold + str(offset)
            items.append(row(key, fold + "source", key + "target", label, fold))
            assignments[key] = fold
    for entity in sorted({item[key] for item in items for key in ("source_id", "target_id")}):
        edge = row("edge-" + entity, entity, "hub", "supported", "unused")
        edge.update(feature_work_keys=[], feature_provenance_complete=True, provenance_verified=True)
        edges.append(edge)
    return dict(schema_version=1, task="source_supported_ranking", sources=sources, items=items,
                edges=edges, assignments=assignments, protected_works=["protected"], exposed_works=[],
                review=dict(reviewer="synthetic", reference_frozen_at="synthetic fixture",
                            exclusion_review="synthetic inventory", frozen_before_predictions=True))


def authorization(bundle):
    return dict(authorization_id="synthetic-only", max_runs=1, prior_runs=0,
                prior_training_records_acknowledged=24, max_total_seconds=30, max_run_seconds=30,
                max_cpu_bytes=2**30, max_gpu_bytes=0, device="cpu", training_authorized=True,
                user_request="synthetic test fixture, not real permission",
                ledger_path=str(experiment.LEDGER.resolve()),
                execution_sha256=experiment.execution_binding(bundle))


def test_ranking_contract_needs_no_invented_contradictions():
    report = experiment.check_bundle(fixture_bundle())
    assert report["coverage"]["test"] == dict(groups=1, complete_groups=1)
    assert report["labels"] == {"supported": 3, "unresolved": 3}
    assert report["training_authorized"] is False


@pytest.mark.parametrize("defect", ["duplicate", "query_fold", "scope", "review", "protected",
                                     "graph_label", "feature", "leak", "exposed", "no_coverage",
                                     "no_support", "source_hash", "negative"])
def test_contract_rejects_invalid_or_leaking_data(defect):
    bundle = fixture_bundle()
    if defect == "duplicate":
        bundle["items"].append(copy.deepcopy(bundle["items"][0]))
    elif defect == "query_fold":
        bundle["items"][-1]["query_id"] = "train"
    elif defect == "scope":
        bundle["items"][-1]["scope"] = {"species": "different"}
    elif defect == "review":
        bundle["review"]["frozen_before_predictions"] = False
    elif defect == "protected":
        bundle["protected_works"] = []
    elif defect == "graph_label":
        bundle["edges"][0]["label"] = "unresolved"
    elif defect == "feature":
        bundle["edges"][0]["feature_work_keys"] = ["external"]
    elif defect == "leak":
        bundle["edges"][0].update(source_id="testsource", target_id="test0target")
    elif defect == "exposed":
        bundle["exposed_works"] = ["test0"]
    elif defect == "no_coverage":
        bundle["items"][-1]["target_id"] = "unseen"
    elif defect == "no_support":
        bundle["items"][-2]["label"] = "unresolved"
    elif defect == "source_hash":
        next(iter(bundle["sources"].values()))["text"] += "tampered"
    else:
        bundle["items"][-1]["label"] = "contradicted"
    with pytest.raises(ValidationError):
        experiment.check_bundle(bundle)


def test_rank_ties_unknowns_and_incomplete_groups():
    groups = experiment.groups_for(fixture_bundle(), "test")
    result = experiment.ranking_summary(groups, {"test0": 1.0, "test1": 1.0})
    assert result["known_support_mrr"] == pytest.approx(2 / 3)
    assert result["strict_known_support_top1"] == 0
    assert result["scientific_accuracy"] is None
    missing = experiment.ranking_summary(groups, {"test0": 1.0})
    assert missing["known_support_mrr"] is None
    assert missing["measured_items"] == 1 and missing["total_items"] == 2
    assert missing["total_groups"] == 1 and missing["group_coverage"] == 0
    with pytest.raises(ValidationError):
        experiment.ranking_summary(groups, {"test0": float("nan")})


def test_structure_baseline_uses_only_graph_and_keeps_oov_missing():
    bundle = fixture_bundle()
    rows = copy.deepcopy(bundle["items"])
    scores = experiment.structure_scores(bundle["edges"], rows)
    assert set(scores.values()) == {1.0}
    rows[0]["target_id"] = "missing"
    assert experiment.structure_scores(bundle["edges"], rows)[rows[0]["item_id"]] is None


def test_check_cli_does_not_launch_or_create_budget(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bundle.json"
    experiment.write_json(path, fixture_bundle())
    monkeypatch.setattr(experiment, "RunGuard", lambda *args: pytest.fail("check must not reserve"))
    monkeypatch.setattr(experiment, "train_and_evaluate", lambda *args: pytest.fail("check must not train"))
    assert experiment.main(["check", "--bundle", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["training_authorized"] is False
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("field,value", [("training_authorized", False), ("execution_sha256", "old"),
                                         ("ledger_path", "new-ledger"), ("max_runs", 6),
                                         ("prior_training_records_acknowledged", 0), ("device", "auto")])
def test_new_allocation_required(field, value):
    bundle = fixture_bundle()
    permit = authorization(bundle)
    permit[field] = value
    with pytest.raises(BudgetExceeded):
        experiment.validate_authorization(permit, experiment.check_bundle(bundle))


def isolate_budget(monkeypatch, tmp_path):
    root = tmp_path / "runs"
    monkeypatch.setattr(experiment, "RUN_ROOT", root)
    monkeypatch.setattr(experiment, "LEDGER", root / "BUDGET.sqlite")
    return root


def test_worker_can_claim_only_once_across_outputs(tmp_path):
    bundle = fixture_bundle()
    permit = authorization(bundle)
    guard = RunGuard(tmp_path / "budget.sqlite", permit)
    attempt = guard.reserve(permit["execution_sha256"], 10)
    assert guard.claim_worker(attempt, permit["execution_sha256"]) == 10
    with pytest.raises(BudgetExceeded, match="already claimed"):
        guard.claim_worker(attempt, permit["execution_sha256"])


def test_launch_supervises_one_worker_and_prevents_second_run(tmp_path, monkeypatch):
    import psutil

    root = isolate_budget(monkeypatch, tmp_path)
    bundle = fixture_bundle()
    permit = authorization(bundle)
    calls = []

    def simulated_training(bundle, guard, seconds, device, output):
        calls.append(device)
        return {"status": "SYNTHETIC_ORCHESTRATION_ONLY"}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            assert command[:3] == [experiment.sys.executable, "-m", "models.kg_link_prediction.experiment"]
            experiment.worker(Path(command[command.index("--output") + 1]), int(command[-1]))

        def poll(self):
            return 0

    monkeypatch.setattr(experiment, "train_and_evaluate", simulated_training)
    monkeypatch.setattr(psutil, "Popen", Process)
    assert experiment.launch(bundle, permit, root / "first") == 0
    assert calls == ["cpu"]
    assert (root / "first" / "COMPLETE.json").exists()
    with pytest.raises(BudgetExceeded):
        experiment.launch(bundle, permit, root / "second")
    assert calls == ["cpu"]


def test_launch_failure_holds_reservation(tmp_path, monkeypatch):
    import psutil

    root = isolate_budget(monkeypatch, tmp_path)
    bundle = fixture_bundle()
    permit = authorization(bundle)

    def fail(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(psutil, "Popen", fail)
    with pytest.raises(OSError):
        experiment.launch(bundle, permit, root / "failed")
    guard = RunGuard(experiment.LEDGER, permit)
    with guard._connect() as connection:
        assert connection.execute("SELECT state FROM attempts").fetchone()[0] == "HELD"
    assert not (root / "failed" / "COMPLETE.json").exists()


def test_output_and_json_are_not_overwritten(tmp_path, monkeypatch):
    root = isolate_budget(monkeypatch, tmp_path)
    bundle = fixture_bundle()
    with pytest.raises(ValidationError):
        experiment.launch(bundle, authorization(bundle), tmp_path / "outside")
    assert not root.exists()
    path = tmp_path / "immutable.json"
    experiment.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        experiment.write_json(path, {})


def test_model_interfaces_without_backward_or_optimizer_updates(tmp_path, monkeypatch):
    import torch

    bundle = fixture_bundle()
    permit = authorization(bundle)
    permit["max_cpu_bytes"] = 8 * 2**30
    guard = RunGuard(tmp_path / "budget.sqlite", permit)
    monkeypatch.setattr(experiment, "CONFIG", dict(experiment.CONFIG, epochs=1, embedding_dim=4, layers=1))
    calls = []

    class NoUpdateOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def zero_grad(self):
            pass

        def step(self):
            calls.append("simulated_step_no_update")

    monkeypatch.setattr(torch.optim, "AdamW", NoUpdateOptimizer)
    monkeypatch.setattr(torch.Tensor, "backward", lambda *args, **kwargs: None)
    result = experiment.train_and_evaluate(bundle, guard, 30, "cpu", tmp_path)
    assert calls == ["simulated_step_no_update"]
    assert result["gnn"]["total_groups"] == result["structure"]["total_groups"] == 1
    assert result["scientific_acceptance"] is False
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["config"]["embedding_dim"] == 4


def test_late_success_is_held_not_reported_complete(tmp_path):
    bundle = fixture_bundle()
    guard = RunGuard(tmp_path / "budget.sqlite", authorization(bundle))
    attempt = guard.reserve("synthetic-late-worker", 1)

    class LateProcess:
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    with pytest.raises(BudgetExceeded, match="wall-time"):
        guard.supervise(LateProcess(), attempt, clock=lambda: 2, started=0)
    with guard._connect() as connection:
        assert connection.execute("SELECT state FROM attempts").fetchone()[0] == "HELD"
