"""Small, explicit completion contracts for AutoResearch modes.

The contracts deliberately validate only bindings that can be checked locally:
paths, hashes, split identity, and the presence of the requested records.  They
do not assert scientific correctness or model quality.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CONTRACT_VERSIONS = {
    "data": "neuroclaw.data.v2",
    "model": "neuroclaw.model.v2",
    "end-to-end": "neuroclaw.end_to_end.v2",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.I)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _path(workspace: Path, raw: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{label} path is required")
        return None
    try:
        candidate = (workspace / raw).resolve()
        if not candidate.is_relative_to(workspace):
            raise ValueError
    except (OSError, ValueError, RuntimeError):
        errors.append(f"{label} path is outside the workspace")
        return None
    try:
        valid_file = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        valid_file = False
    if not valid_file:
        errors.append(f"{label} file is missing or empty: {raw}")
        return None
    return candidate


def _file_record(workspace: Path, record: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(record, dict):
        errors.append(f"{label} must contain path and sha256")
        return None
    path = _path(workspace, record.get("path"), label, errors)
    digest = record.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        errors.append(f"{label}.sha256 must be a 64-character hex digest")
    elif path is not None:
        try:
            actual = _digest(path)
        except OSError:
            errors.append(f"{label} file cannot be read")
            return None
        if actual.lower() != digest.lower():
            errors.append(f"{label} sha256 does not match the file")
    return path


def _manifest(workspace: Path, artifacts: list[str], names: tuple[str, ...],
              expected_contract: str, errors: list[str]) -> tuple[dict | None, Path | None]:
    candidates: list[Path] = []
    for raw in artifacts + list(names):
        try:
            path = (workspace / raw).resolve()
            if path.is_relative_to(workspace) and path.is_file() and path.suffix.lower() == ".json":
                candidates.append(path)
        except (OSError, ValueError, RuntimeError):
            continue
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(value, dict) and value.get("contract") == expected_contract:
            return value, path
    errors.append(f"missing readable mode manifest; expected one of {', '.join(names)}")
    return None, None


def _required_string(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} is required")


def _split(split: Any, label: str, errors: list[str]) -> None:
    if not isinstance(split, dict):
        errors.append(f"{label} must be an object")
        return
    for key in ("strategy", "group_key"):
        _required_string(split.get(key), f"{label}.{key}", errors)
    if type(split.get("seed")) is not int:
        errors.append(f"{label}.seed must be an integer")
    for key in ("train_subject_ids_sha256", "validation_subject_ids_sha256", "test_subject_ids_sha256"):
        value = split.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            errors.append(f"{label}.{key} must be a 64-character hex digest")


def _id_list(workspace: Path, record: Any, label: str, errors: list[str]) -> set[str] | None:
    path = _file_record(workspace, record, label, errors)
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        errors.append(f"{label} must be a readable JSON array of subject IDs")
        return None
    if (not isinstance(value, list) or
            any(not isinstance(item, str) or not item.strip() or item != item.strip()
                for item in value) or len(value) != len(set(value))):
        errors.append(f"{label} must contain unique, nonempty subject ID strings")
        return None
    return set(value)


def _subject_record(workspace: Path, record: Any, label: str,
                    errors: list[str]) -> set[str] | None:
    if not isinstance(record, dict):
        errors.append(f"{label} must be an object")
        return None
    value = record.get("subject_ids_sha256")
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        errors.append(f"{label}.subject_ids_sha256 must be a 64-character hex digest")
    if type(record.get("subject_count")) is not int or record["subject_count"] < 0:
        errors.append(f"{label}.subject_count must be a nonnegative integer")
    id_record = record.get("subject_ids")
    ids = _id_list(workspace, id_record, f"{label}.subject_ids", errors)
    if isinstance(id_record, dict) and id_record.get("sha256") != value:
        errors.append(f"{label}.subject_ids_sha256 must match the subject_ids file")
    if ids is not None and record.get("subject_count") != len(ids):
        errors.append(f"{label}.subject_count does not match the subject_ids file")
    return ids


def _tabular_record(path: Path | None, record: Any, ids: set[str] | None,
                    label: str, errors: list[str], *, output: bool) -> None:
    if path is None or path.suffix.lower() not in {".csv", ".tsv"} or not isinstance(record, dict):
        return
    id_column = record.get("subject_id_column")
    if not isinstance(id_column, str) or not id_column.strip():
        errors.append(f"{label}.subject_id_column is required for CSV/TSV")
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            header = next(reader)
            if not header or len(header) != len(set(header)) or any(not name for name in header):
                errors.append(f"{label} has an empty or duplicate table header")
                return
            if id_column not in header:
                errors.append(f"{label}.subject_id_column is absent from the table")
                return
            index = header.index(id_column)
            actual_ids: set[str] = set()
            rows = 0
            for row in reader:
                rows += 1
                if len(row) != len(header) or not row[index].strip():
                    errors.append(f"{label} has a malformed row or blank subject ID at row {rows}")
                    return
                actual_ids.add(row[index])
    except (OSError, UnicodeError, csv.Error, StopIteration):
        errors.append(f"{label} cannot be read as a CSV/TSV table")
        return
    if ids is not None and actual_ids != ids:
        errors.append(f"{label}.subject_ids do not match the table's {id_column} values")
    if output:
        if record.get("shape") != [rows, len(header)]:
            errors.append(f"{label}.shape does not match the table rows and columns")
        schema = record.get("schema")
        if isinstance(schema, dict) and set(schema) != set(header):
            errors.append(f"{label}.schema columns do not match the table header")


def _data_split(workspace: Path, split: Any, output_ids: set[str], errors: list[str]) -> None:
    _split(split, "split", errors)
    if not isinstance(split, dict):
        return
    records = split.get("subject_ids")
    if not isinstance(records, dict):
        errors.append("split.subject_ids must contain train, validation and test files")
        return
    partitions: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        record = records.get(name)
        ids = _id_list(workspace, record, f"split.subject_ids.{name}", errors)
        if isinstance(record, dict) and record.get("sha256") != split.get(f"{name}_subject_ids_sha256"):
            errors.append(f"split.{name}_subject_ids_sha256 must match its subject_ids file")
        if ids is not None:
            partitions[name] = ids
    if len(partitions) != 3:
        return
    if any(partitions[left] & partitions[right] for left, right in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("split subject IDs overlap across train, validation and test")
    membership = set().union(*partitions.values())
    if membership != output_ids:
        errors.append("split subject IDs must exactly cover the output subject IDs")
    if split.get("group_key") == "subject_id":
        assignments = {subject: subject for subject in membership}
    else:
        path = _file_record(workspace, split.get("group_assignments"),
                            "split.group_assignments", errors)
        try:
            assignments = json.loads(path.read_text(encoding="utf-8-sig")) if path else None
        except (OSError, ValueError, UnicodeError):
            assignments = None
        if (not isinstance(assignments, dict) or set(assignments) != membership or
                any(not isinstance(group, str) or not group.strip() for group in assignments.values())):
            errors.append("split.group_assignments must map every output subject to a nonempty group")
            return
    owners: dict[str, str] = {}
    for partition, ids in partitions.items():
        for subject in ids:
            group = assignments[subject]
            if group in owners and owners[group] != partition:
                errors.append("split group assignments cross train, validation or test")
                return
            owners[group] = partition


def _data(workspace: Path, manifest: dict, errors: list[str]) -> None:
    if manifest.get("contract") != CONTRACT_VERSIONS["data"]:
        errors.append(f"contract must be {CONTRACT_VERSIONS['data']}")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("inputs must be a nonempty list")
    else:
        input_ids: set[str] = set()
        for index, record in enumerate(inputs):
            label = f"inputs[{index}]"
            path = _file_record(workspace, record, label, errors)
            ids = _subject_record(workspace, record, label, errors)
            _tabular_record(path, record, ids, label, errors, output=False)
            if ids is not None:
                input_ids.update(ids)
    output_ids: set[str] = set()
    if not isinstance(outputs, list) or not outputs:
        errors.append("outputs must be a nonempty list")
    else:
        for index, record in enumerate(outputs):
            label = f"outputs[{index}]"
            path = _file_record(workspace, record, label, errors)
            ids = _subject_record(workspace, record, label, errors)
            _tabular_record(path, record, ids, label, errors, output=True)
            if ids is not None:
                output_ids.update(ids)
            if isinstance(record, dict):
                shape = record.get("shape")
                schema = record.get("schema")
                if (not isinstance(shape, list) or not shape or
                        any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
                    errors.append(f"{label}.shape must contain positive integer dimensions")
                if (not isinstance(schema, dict) or not schema or
                        any(not isinstance(name, str) or not name.strip() or
                            not isinstance(kind, str) or not kind.strip()
                            for name, kind in schema.items())):
                    errors.append(f"{label}.schema must map field names to nonempty type names")
    if isinstance(inputs, list) and inputs and isinstance(outputs, list) and outputs and not output_ids:
        errors.append("outputs must contain at least one subject ID")
    if isinstance(inputs, list) and inputs and not output_ids.issubset(input_ids):
        errors.append("output subject IDs must come from the input subject IDs")
    _data_split(workspace, manifest.get("split"), output_ids, errors)
    split = manifest.get("split")
    train_hash = split.get("train_subject_ids_sha256") if isinstance(split, dict) else None
    transforms = manifest.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        errors.append("transforms must be a nonempty list")
    else:
        for index, item in enumerate(transforms):
            label = f"transforms[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{label} must be an object")
                continue
            _required_string(item.get("name"), f"{label}.name", errors)
            if not isinstance(item.get("parameters"), dict):
                errors.append(f"{label}.parameters must be an object")
            if not isinstance(item.get("fit_on"), str) or item["fit_on"] not in {"train", "none"}:
                errors.append(f"{label}.fit_on must be train or none")
            elif item["fit_on"] == "train" and item.get("fit_subject_ids_sha256") != train_hash:
                errors.append(f"{label}.fit_subject_ids_sha256 must match the train split")
            _file_record(workspace, item.get("source"), f"{label}.source", errors)
    qc = manifest.get("qc")
    if not isinstance(qc, dict) or qc.get("status") != "passed" or not isinstance(qc.get("checks"), list) or not qc["checks"]:
        errors.append("qc.status must be passed with a nonempty checks list")
    else:
        for index, check in enumerate(qc["checks"]):
            if (not isinstance(check, dict) or not isinstance(check.get("name"), str) or
                    not check["name"].strip() or check.get("status") != "passed" or
                    "observed" not in check or check.get("observed") is None or
                    not isinstance(check.get("criterion"), str) or not check["criterion"].strip()):
                errors.append(f"qc.checks[{index}] needs a name, passed status, observed value and criterion")
        path = _file_record(workspace, qc.get("report"), "qc.report", errors)
        if path is not None:
            try:
                report = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, UnicodeError):
                report = None
            if not isinstance(report, dict) or report.get("status") != qc["status"] or report.get("checks") != qc["checks"]:
                errors.append("qc.report must contain the same status and checks as qc")


def _model(workspace: Path, manifest: dict, errors: list[str]) -> None:
    if manifest.get("contract") != CONTRACT_VERSIONS["model"]:
        errors.append(f"contract must be {CONTRACT_VERSIONS['model']}")
    model_input = manifest.get("model_ready_input")
    input_path = _file_record(workspace, model_input, "model_ready_input", errors)
    input_ids = _subject_record(workspace, model_input, "model_ready_input", errors)
    _tabular_record(input_path, model_input, input_ids, "model_ready_input", errors, output=False)
    _data_split(workspace, manifest.get("split"), input_ids or set(), errors)
    if type(manifest.get("training_seed")) is not int:
        errors.append("training_seed must be an integer")
    leakage = manifest.get("leakage_check")
    if not isinstance(leakage, dict) or leakage.get("status") != "passed" or leakage.get("overlap_count") != 0:
        errors.append("leakage_check must be passed with overlap_count 0")
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        errors.append("baseline is required")
    else:
        _required_string(baseline.get("name"), "baseline.name", errors)
        _required_string(baseline.get("metric"), "baseline.metric", errors)
        if not isinstance(baseline.get("source"), dict):
            errors.append("baseline.source is required")
        else:
            _file_record(workspace, baseline["source"], "baseline.source", errors)
    config = manifest.get("config")
    if isinstance(config, dict) and "path" in config:
        _file_record(workspace, config, "config", errors)
    elif not isinstance(config, dict) or not config:
        errors.append("config must be a nonempty inline object or a path/sha256 record")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a nonempty list")
    else:
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"metrics[{index}] must be an object")
                continue
            _required_string(metric.get("name"), f"metrics[{index}].name", errors)
            _required_string(metric.get("split"), f"metrics[{index}].split", errors)
            if not isinstance(metric.get("source"), dict):
                errors.append(f"metrics[{index}].source is required")
            else:
                _file_record(workspace, metric["source"], f"metrics[{index}].source", errors)
    _file_record(workspace, manifest.get("checkpoint"), "checkpoint", errors)


def _end_to_end(workspace: Path, manifest: dict, errors: list[str]) -> None:
    if manifest.get("contract") != CONTRACT_VERSIONS["end-to-end"]:
        errors.append(f"contract must be {CONTRACT_VERSIONS['end-to-end']}")
    idea = manifest.get("idea")
    if not isinstance(idea, dict):
        errors.append("idea section is required")
    else:
        for key in ("chain", "ranking"):
            path = _file_record(workspace, idea.get(key), f"idea.{key}", errors)
            if path is not None and path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError, UnicodeError):
                    payload = None
                if not isinstance(payload, (dict, list)) or not payload:
                    errors.append(f"idea.{key} JSON must contain a nonempty chain or ranking record")
        novelty = idea.get("novelty")
        if (not isinstance(novelty, dict) or not isinstance(novelty.get("status"), str) or
                novelty["status"] not in {"reviewed", "held", "rejected"}):
            errors.append("idea.novelty.status must be reviewed, held, or rejected")
        else:
            path = _file_record(workspace, novelty, "idea.novelty", errors)
            try:
                report = json.loads(path.read_text(encoding="utf-8-sig")) if path and path.suffix.lower() == ".json" else None
            except (OSError, ValueError, UnicodeError):
                report = None
            if (not isinstance(report, dict) or report.get("status") != novelty["status"] or
                    not isinstance(report.get("summary"), str) or not report["summary"].strip()):
                errors.append("idea.novelty must be JSON with matching status and a summary")
    stages: dict[str, dict] = {}
    for key in ("data", "model", "validation"):
        record = manifest.get(key)
        if key in {"data", "model"} and isinstance(record, dict) and isinstance(record.get("manifest"), dict):
            record = record["manifest"]
        path = _file_record(workspace, record, key, errors)
        if key == "validation":
            try:
                report = json.loads(path.read_text(encoding="utf-8-sig")) if path and path.suffix.lower() == ".json" else None
            except (OSError, ValueError, UnicodeError):
                report = None
            if (not isinstance(report, dict) or not isinstance(report.get("status"), str) or
                    report["status"] not in {"passed", "failed", "inconclusive"} or
                    not isinstance(report.get("summary"), str) or not report["summary"].strip()):
                errors.append("validation must be JSON with passed, failed or inconclusive status and a summary")
        if path is not None and key in {"data", "model"}:
            try:
                referenced = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, UnicodeError):
                errors.append(f"{key} manifest is not readable JSON")
                continue
            if not isinstance(referenced, dict):
                errors.append(f"{key} manifest must be a JSON object")
                continue
            stages[key] = referenced
            nested_errors: list[str] = []
            (_data if key == "data" else _model)(workspace, referenced, nested_errors)
            errors.extend(f"{key}.{item}" for item in nested_errors)
    if "data" in stages and "model" in stages:
        data, model = stages["data"], stages["model"]
        model_input = model.get("model_ready_input")
        outputs = data.get("outputs")
        if isinstance(model_input, dict) and isinstance(outputs, list):
            input_path = model_input.get("path")
            input_hash = model_input.get("sha256")
            if not any(isinstance(output, dict) and output.get("path") == input_path
                       and output.get("sha256") == input_hash
                       and output.get("subject_ids_sha256") == model_input.get("subject_ids_sha256")
                       and output.get("subject_count") == model_input.get("subject_count")
                       for output in outputs):
                errors.append("model.model_ready_input must match a data.outputs path, sha256 and subjects")
        data_split, model_split = data.get("split"), model.get("split")
        if isinstance(data_split, dict) and isinstance(model_split, dict):
            keys = ("strategy", "group_key", "seed", "train_subject_ids_sha256",
                    "validation_subject_ids_sha256", "test_subject_ids_sha256")
            if any(data_split.get(key) != model_split.get(key) for key in keys):
                errors.append("model.split must match the data.split strategy, group, seed and subject IDs")


def validate_mode_contract(mode: str, workspace: Path, artifacts: list[str]) -> dict[str, Any]:
    """Validate the mode manifest and all locally bound records."""
    errors: list[str] = []
    names = {
        "data": ("data_manifest.json", "DATA_MANIFEST.json"),
        "model": ("model_manifest.json", "MODEL_MANIFEST.json"),
        "end-to-end": ("end_to_end_manifest.json", "END_TO_END_MANIFEST.json"),
    }.get(mode, ())
    manifest, path = _manifest(workspace.resolve(), artifacts, names,
                               CONTRACT_VERSIONS.get(mode, ""), errors)
    if manifest is not None:
        if mode == "data":
            _data(workspace.resolve(), manifest, errors)
        elif mode == "model":
            _model(workspace.resolve(), manifest, errors)
        elif mode == "end-to-end":
            _end_to_end(workspace.resolve(), manifest, errors)
    manifest_digest = None
    if path is not None:
        try:
            manifest_digest = _digest(path)
        except OSError:
            errors.append("mode manifest cannot be read after validation")
    return {"ok": not errors, "errors": errors, "manifest": str(path) if path else None,
            "manifest_sha256": manifest_digest,
            "contract": CONTRACT_VERSIONS.get(mode)}


def mode_contract_prompt(mode: str) -> str:
    prompts = {
        "data": "Write data_manifest.json before finish_autoresearch. Use contract neuroclaw.data.v2. Each input/output needs path+sha256, subject_count, subject_ids_sha256 and subject_ids:{path,sha256} pointing to a JSON array of unique IDs; CSV/TSV also needs subject_id_column. Each output needs positive-integer shape and schema (field:type). Split needs strategy, group_key, seed, train/validation/test_subject_ids_sha256 and subject_ids:{train,validation,test} file records; those JSON lists must be disjoint and exactly cover output IDs. For group_key other than subject_id, provide group_assignments:{path,sha256} as a JSON subject-to-group map. Each transform needs name, parameters object, fit_on:train|none and source:{path,sha256}; fit_on:train also needs fit_subject_ids_sha256 matching the train split. QC needs status:passed, named passed checks with observed value and criterion, and report:{path,sha256} containing the same status/checks. All paths must be workspace files; use pseudonymous IDs where appropriate.",
        "model": "Write model_manifest.json before finish_autoresearch. Use contract neuroclaw.model.v2. The model_ready_input needs path+sha256, subject_count, subject_ids_sha256 and subject_ids:{path,sha256}; CSV/TSV also needs subject_id_column. Reuse the Data split format: three hashed subject-ID JSON lists, group_key, strategy and split seed; lists must cover model input IDs without subject or group overlap. Record a separate integer training_seed, leakage_check passed with overlap_count=0, a named baseline and hashed source, a nonempty config, metrics with hashed source files, and a checkpoint path+sha256.",
        "end-to-end": "Write end_to_end_manifest.json before finish_autoresearch. Use contract neuroclaw.end_to_end.v2 and bind data and model manifests, nonempty Idea chain and ranking files, a novelty JSON file with status reviewed/held/rejected and summary, and a validation JSON file with status passed/failed/inconclusive and summary; each file needs path+sha256. The Model input and split must match the Data output and split, including subject hashes/counts. Use generate_idea_hypotheses and rank_idea_hypotheses when generating a new Idea; preserve conditional chains and never claim scientific novelty from the contract.",
    }
    return prompts.get(mode, "")
