"""Local completion-contract checks for the three non-idea AutoResearch modes."""

import hashlib
import json

from core.autoresearch_contracts import CONTRACT_VERSIONS, validate_mode_contract
from core.autoresearch_runtime import AutoResearchRun


def _write(workspace, name, content):
    path = workspace / name
    path.write_text(content, encoding="utf-8")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _split(workspace):
    train = _write(workspace, "train_ids.json", '["S1"]')
    validation = _write(workspace, "validation_ids.json", "[]")
    test = _write(workspace, "test_ids.json", "[]")
    return {"strategy": "grouped", "group_key": "subject_id", "seed": 17,
            "subject_ids": {"train": train, "validation": validation, "test": test},
            "train_subject_ids_sha256": train["sha256"],
            "validation_subject_ids_sha256": validation["sha256"],
            "test_subject_ids_sha256": test["sha256"]}


def _data_manifest(workspace):
    ids = _write(workspace, "subject_ids.json", '["S1"]')
    source = _write(workspace, "input.csv", "subject,value\nS1,1\n")
    output = _write(workspace, "data_output.csv", "subject,value\nS1,2\n")
    source.update(subject_ids=ids, subject_ids_sha256=ids["sha256"],
                  subject_count=1, subject_id_column="subject")
    output.update(subject_ids=ids, subject_ids_sha256=ids["sha256"],
                  subject_count=1, subject_id_column="subject",
                  shape=[1, 2], schema={"subject": "string", "value": "float"})
    train = _write(workspace, "train_ids.json", '["S1"]')
    validation = _write(workspace, "validation_ids.json", "[]")
    test = _write(workspace, "test_ids.json", "[]")
    split = {"strategy": "grouped", "group_key": "subject_id", "seed": 17,
             "subject_ids": {"train": train, "validation": validation, "test": test},
             **{f"{name}_subject_ids_sha256": record["sha256"]
                for name, record in (("train", train), ("validation", validation), ("test", test))}}
    qc_checks = [{"name": "finite_values", "status": "passed",
                  "observed": 0, "criterion": "nonfinite_count == 0"}]
    qc_report = _write(workspace, "qc.json", json.dumps({"status": "passed", "checks": qc_checks}))
    return {"contract": CONTRACT_VERSIONS["data"], "inputs": [source], "outputs": [output],
            "split": split,
            "transforms": [{"name": "scale_value", "parameters": {"factor": 2},
                            "fit_on": "none", "source": _write(workspace, "preprocess.py", "scale=2\n")}],
            "qc": {"status": "passed", "checks": qc_checks, "report": qc_report}}


def _model_manifest(workspace, model_input=None):
    if model_input is None:
        ids = _write(workspace, "model_subject_ids.json", '["S1"]')
        model_input = _write(workspace, "model_ready.npz", "model-ready")
        model_input.update(subject_ids=ids, subject_ids_sha256=ids["sha256"], subject_count=1)
    baseline = _write(workspace, "baseline.json", "{}")
    metric = _write(workspace, "metrics.json", "{}")
    checkpoint = _write(workspace, "checkpoint.pt", "weights")
    return {"contract": CONTRACT_VERSIONS["model"], "model_ready_input": model_input,
            "split": _split(workspace), "training_seed": 17,
            "leakage_check": {"status": "passed", "overlap_count": 0},
            "baseline": {"name": "mean", "metric": "mae", "source": baseline},
            "config": {"seed": 17, "architecture": "baseline"},
            "metrics": [{"name": "mae", "split": "test", "source": metric}],
            "checkpoint": checkpoint}


def test_data_contract_binds_hashes_split_shape_schema_and_qc(tmp_path):
    manifest = _data_manifest(tmp_path)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_mode_contract("data", tmp_path, ["data_manifest.json"])
    assert result["ok"]
    assert result["manifest_sha256"] == hashlib.sha256(
        (tmp_path / "data_manifest.json").read_bytes()).hexdigest()
    (tmp_path / "data_output.csv").write_text("changed", encoding="utf-8")
    assert not validate_mode_contract("data", tmp_path, ["data_manifest.json"])["ok"]


def test_data_contract_checks_subject_lists_and_tabular_shape(tmp_path):
    manifest = _data_manifest(tmp_path)
    manifest["outputs"][0]["subject_count"] = 2
    manifest["outputs"][0]["shape"] = [2, 2]
    manifest["outputs"][0]["schema"] = {"wrong_column": "string"}
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_mode_contract("data", tmp_path, ["data_manifest.json"])["errors"]
    assert any("subject_count does not match" in error for error in errors)
    assert any("shape does not match" in error for error in errors)
    assert any("schema columns do not match" in error for error in errors)


def test_data_contract_rejects_overlap_even_when_split_hashes_match_files(tmp_path):
    manifest = _data_manifest(tmp_path)
    validation = _write(tmp_path, "validation_ids.json", '["S1"]')
    manifest["split"]["subject_ids"]["validation"] = validation
    manifest["split"]["validation_subject_ids_sha256"] = validation["sha256"]
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_mode_contract("data", tmp_path, ["data_manifest.json"])["errors"]
    assert any("split subject IDs overlap" in error for error in errors)


def test_data_contract_rejects_family_group_leakage(tmp_path):
    manifest = _data_manifest(tmp_path)
    rows = "subject,value\nS1,1\nS2,2\n"
    ids = _write(tmp_path, "subject_ids.json", '["S1","S2"]')
    for key, name in (("inputs", "input.csv"), ("outputs", "data_output.csv")):
        record = manifest[key][0]
        record.update(_write(tmp_path, name, rows), subject_ids=ids,
                      subject_ids_sha256=ids["sha256"], subject_count=2)
    manifest["outputs"][0]["shape"] = [2, 2]
    validation = _write(tmp_path, "validation_ids.json", '["S2"]')
    manifest["split"]["subject_ids"]["validation"] = validation
    manifest["split"]["validation_subject_ids_sha256"] = validation["sha256"]
    manifest["split"]["group_key"] = "family_id"
    manifest["split"]["group_assignments"] = _write(
        tmp_path, "families.json", '{"S1":"F1","S2":"F1"}')
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_mode_contract("data", tmp_path, ["data_manifest.json"])["errors"]
    assert any("group assignments cross" in error for error in errors)
    manifest["split"]["group_assignments"] = _write(
        tmp_path, "families.json", '{"S1":"F1","S2":"F2"}')
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_mode_contract("data", tmp_path, ["data_manifest.json"])["ok"]


def test_data_contract_requires_bound_transform_and_matching_qc_report(tmp_path):
    manifest = _data_manifest(tmp_path)
    manifest["transforms"][0]["source"]["sha256"] = "0" * 64
    manifest["qc"]["checks"][0]["status"] = "failed"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_mode_contract("data", tmp_path, ["data_manifest.json"])["errors"]
    assert any("transforms[0].source sha256" in error for error in errors)
    assert any("qc.checks[0]" in error for error in errors)
    assert any("qc.report must contain" in error for error in errors)


def test_data_contract_binds_train_fitted_transform_to_train_subjects(tmp_path):
    manifest = _data_manifest(tmp_path)
    transform = manifest["transforms"][0]
    transform["fit_on"] = "train"
    transform["fit_subject_ids_sha256"] = "0" * 64
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_mode_contract("data", tmp_path, ["data_manifest.json"])["errors"]
    assert any("fit_subject_ids_sha256" in error for error in errors)
    transform["fit_subject_ids_sha256"] = manifest["split"]["train_subject_ids_sha256"]
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_mode_contract("data", tmp_path, ["data_manifest.json"])["ok"]


def test_model_contract_requires_leakage_free_checkpoint_and_metric_sources(tmp_path):
    manifest = _model_manifest(tmp_path)
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_mode_contract("model", tmp_path, ["model_manifest.json"])["ok"]
    manifest["leakage_check"]["overlap_count"] = 1
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_mode_contract("model", tmp_path, ["model_manifest.json"])
    assert not result["ok"] and any("overlap_count 0" in error for error in result["errors"])


def test_end_to_end_contract_requires_idea_chain_ranking_novelty_and_validation(tmp_path):
    chain = _write(tmp_path, "chain.json", '{"chains":[{"id":"C1"}]}')
    ranking = _write(tmp_path, "ranking.json", '{"ranked":[{"id":"C1"}]}')
    novelty_file = _write(tmp_path, "novelty.json", '{"status":"held","summary":"No novelty claim"}')
    data_value = _data_manifest(tmp_path)
    (tmp_path / "data_manifest.json").write_text(json.dumps(data_value), encoding="utf-8")
    data_manifest = _write(tmp_path, "data_manifest.json", (tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    model_value = _model_manifest(tmp_path, data_value["outputs"][0])
    model_value["split"] = data_value["split"]
    (tmp_path / "model_manifest.json").write_text(json.dumps(model_value), encoding="utf-8")
    model_manifest = _write(tmp_path, "model_manifest.json", (tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    validation = _write(tmp_path, "validation.json", '{"status":"inconclusive","summary":"Bounded check"}')
    manifest = {"contract": CONTRACT_VERSIONS["end-to-end"],
                "idea": {"chain": chain, "ranking": ranking,
                         "novelty": {**novelty_file, "status": "held"}},
                "data": data_manifest, "model": model_manifest, "validation": validation}
    (tmp_path / "end_to_end_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_mode_contract("end-to-end", tmp_path, ["end_to_end_manifest.json"])["ok"]
    assert validate_mode_contract("end-to-end", tmp_path,
                                  ["data_manifest.json", "model_manifest.json",
                                   "end_to_end_manifest.json"])["ok"]
    (tmp_path / "data_manifest.json").write_text("{}", encoding="utf-8")
    manifest["data"]["sha256"] = hashlib.sha256(
        (tmp_path / "data_manifest.json").read_bytes()).hexdigest()
    (tmp_path / "end_to_end_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_mode_contract("end-to-end", tmp_path, ["end_to_end_manifest.json"])
    assert not result["ok"] and any(error.startswith("data.inputs") for error in result["errors"])


def test_end_to_end_rejects_unrelated_model_input_and_changed_split(tmp_path):
    data = _data_manifest(tmp_path)
    model = _model_manifest(tmp_path)
    (tmp_path / "data_manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "model_manifest.json").write_text(json.dumps(model), encoding="utf-8")
    manifest = {"contract": CONTRACT_VERSIONS["end-to-end"],
                "idea": {"chain": _write(tmp_path, "chain.json", '{"chains":[{"id":"C1"}]}'),
                         "ranking": _write(tmp_path, "ranking.json", '{"ranked":[{"id":"C1"}]}'),
                         "novelty": _write(tmp_path, "novelty.json", '{"status":"held","summary":"No novelty claim"}')},
                "data": _write(tmp_path, "data_manifest.json", json.dumps(data)),
                "model": _write(tmp_path, "model_manifest.json", json.dumps(model)),
                "validation": _write(tmp_path, "validation.json", '{"status":"failed","summary":"Synthetic negative result"}')}
    manifest["idea"]["novelty"]["status"] = "held"
    (tmp_path / "end_to_end_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_mode_contract("end-to-end", tmp_path, ["end_to_end_manifest.json"])
    assert not result["ok"]
    assert any("model.model_ready_input" in error for error in result["errors"])

    model["model_ready_input"] = data["outputs"][0]
    model["split"]["seed"] += 1
    manifest["model"] = _write(tmp_path, "model_manifest.json", json.dumps(model))
    (tmp_path / "end_to_end_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_mode_contract("end-to-end", tmp_path, ["end_to_end_manifest.json"])
    assert not result["ok"]
    assert any("model.split must match" in error for error in result["errors"])


def test_finish_rejects_nonempty_report_without_mode_contract(tmp_path):
    report = tmp_path / "result.md"
    report.write_text("report", encoding="utf-8")
    run = AutoResearchRun(tmp_path, "data")
    run.observe("read_workspace_file", {"path": "result.md"}, {"success": True, "executed": True})
    result = run.finish({"status": "completed", "summary": "done", "validation": "checked",
                         "artifacts": ["result.md"], "evidence_ids": [1]})
    assert not result["success"]
    assert "Mode delivery contract failed" in result["error"]
    assert run.state["mode_contract"]["ok"] is False


def test_prompt_adds_contract_for_data_model_and_end_to_end_but_not_idea(tmp_path):
    assert "data_manifest.json" in AutoResearchRun(tmp_path, "data").prompt()
    assert "model_manifest.json" in AutoResearchRun(tmp_path, "model").prompt()
    assert "end_to_end_manifest.json" in AutoResearchRun(tmp_path, "end-to-end").prompt()
    assert "data_manifest.json" not in AutoResearchRun(tmp_path, "idea").prompt()


def test_end_to_end_finish_records_contract_result(tmp_path):
    for name, content in {
        "chain.json": '{"chains":[{"id":"C1"}]}',
        "ranking.json": '{"ranked":[{"id":"C1"}]}',
        "novelty.json": '{"status":"reviewed","summary":"Synthetic review"}',
        "validation.json": '{"status":"passed","summary":"Synthetic check"}',
    }.items():
        _write(tmp_path, name, content)
    data = _data_manifest(tmp_path)
    (tmp_path / "data_manifest.json").write_text(json.dumps(data), encoding="utf-8")
    model = _model_manifest(tmp_path, data["outputs"][0])
    model["split"] = data["split"]
    (tmp_path / "model_manifest.json").write_text(json.dumps(model), encoding="utf-8")
    refs = {name: {"path": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in ("chain.json", "ranking.json", "novelty.json", "data_manifest.json", "model_manifest.json", "validation.json")}
    manifest = {"contract": CONTRACT_VERSIONS["end-to-end"],
                "idea": {"chain": refs["chain.json"], "ranking": refs["ranking.json"],
                         "novelty": {**refs["novelty.json"], "status": "reviewed"}},
                "data": refs["data_manifest.json"], "model": refs["model_manifest.json"],
                "validation": refs["validation.json"]}
    (tmp_path / "end_to_end_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run = AutoResearchRun(tmp_path, "end-to-end")
    run.observe("read_workspace_file", {"path": "end_to_end_manifest.json"}, {"success": True, "executed": True})
    result = run.finish({"status": "completed", "summary": "done", "validation": "checked",
                         "artifacts": ["end_to_end_manifest.json"], "evidence_ids": [1]})
    assert result["success"]
    assert run.state["mode_contract"]["contract"] == CONTRACT_VERSIONS["end-to-end"]
    assert run.state["mode_contract"]["manifest_sha256"] == hashlib.sha256(
        (tmp_path / "end_to_end_manifest.json").read_bytes()).hexdigest()
