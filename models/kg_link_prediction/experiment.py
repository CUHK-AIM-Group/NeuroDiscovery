"""Single entrypoint for frozen, source-supported relation ranking experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .validation import ValidationError, ValidationStopper, check_item, fingerprint, preflight
from .run_guard import BudgetExceeded, RunGuard


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "tmp" / "gnn_ranking"
LEDGER = RUN_ROOT / "BUDGET.sqlite"
CONFIG = dict(model="rgcn", embedding_dim=64, layers=2, dropout=0.1,
              decoder="directional", reverse_relations=True,
              epochs=50, patience=5, seed=123, learning_rate=0.001)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def execution_binding(bundle):
    files = ("experiment.py", "validation.py", "run_guard.py", "gnn.py", "__init__.py")
    return fingerprint({"bundle": bundle, "code": {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in files
    }})


def groups_for(bundle, fold):
    groups = defaultdict(list)
    for row in bundle["items"]:
        if bundle["assignments"][row["item_id"]] == fold:
            groups[row["query_id"]].append(row)
    return groups


def vocabulary(edges):
    entities = {row[endpoint] for row in edges for endpoint in ("source_id", "target_id")}
    return entities, {row["relation"] for row in edges}


def measurable(row, entities, relations):
    return row["source_id"] in entities and row["target_id"] in entities and row["relation"] in relations


def check_bundle(bundle):
    if bundle.get("schema_version") != 1 or bundle.get("task") != "source_supported_ranking":
        raise ValidationError("source_supported_ranking schema_version 1 required")
    allowed = {"schema_version", "task", "items", "sources", "assignments", "edges",
               "protected_works", "exposed_works", "review"}
    if set(bundle) != allowed:
        raise ValidationError("exact bundle fields required; no external features/config overrides")
    review = bundle["review"]
    if not isinstance(review, dict) or any(
        not isinstance(review.get(key), str) or not review[key].strip()
        for key in ("reviewer", "reference_frozen_at", "exclusion_review")
    ) or review.get("frozen_before_predictions") is not True:
        raise ValidationError("source/exclusion review and pre-prediction freeze required")
    for key in ("protected_works", "exposed_works"):
        values = bundle[key]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValidationError(f"invalid {key}")
    if not bundle["protected_works"]:
        raise ValidationError("protected work inventory required; completeness needs review")
    protected = set(bundle["protected_works"])
    report = preflight(bundle["items"], bundle["sources"], bundle["assignments"],
                       bundle["edges"], protected, set(bundle["exposed_works"]),
                       require_contradictions=False)
    for edge in bundle["edges"]:
        check_item(edge, bundle["sources"], protected)
        if edge["label"] != "supported" or edge["feature_work_keys"]:
            raise ValidationError("training graph requires supported sources and no external features")
    seen = {}
    entities, relations = vocabulary(bundle["edges"])
    coverage = {}
    for fold in ("train", "validation", "test"):
        for row in bundle["items"]:
            if not isinstance(row.get("query_id"), str) or not row["query_id"].strip():
                raise ValidationError("query_id required")
        groups = groups_for(bundle, fold)
        complete = 0
        for query, rows in groups.items():
            if query in seen:
                raise ValidationError("query crosses folds")
            seen[query] = fold
            contexts = {fingerprint({key: row[key] for key in
                        ("source_id", "relation", "measurement", "scope", "role")}) for row in rows}
            if len(contexts) != 1 or len({row["target_id"] for row in rows}) != len(rows):
                raise ValidationError("query needs identical context and distinct targets")
            if len(rows) < 2 or not any(row["label"] == "supported" for row in rows):
                raise ValidationError("query needs candidates and known support")
            complete += all(measurable(row, entities, relations) for row in rows)
        coverage[fold] = {"groups": len(groups), "complete_groups": complete}
        if fold != "train" and not complete:
            raise ValidationError(f"{fold} has no completely measurable query")
    report.update(bundle_sha256=fingerprint(bundle), execution_sha256=execution_binding(bundle),
                  coverage=coverage, labels=dict(Counter(row["label"] for row in bundle["items"])),
                  task=bundle["task"], config=CONFIG, review_is_not_independent_validation=True)
    return report


def ranking_summary(groups, scores):
    reciprocals, top_hits, rows_out = [], [], []
    measured_items = complete = 0
    for query, rows in sorted(groups.items()):
        values = [scores.get(row["item_id"]) for row in rows]
        for value in values:
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValidationError("invalid ranking score")
        measured_items += sum(value is not None for value in values)
        is_complete = all(value is not None for value in values)
        supported_ranks = []
        for row, value in zip(rows, values):
            rank = None
            if is_complete:
                ties = sum(math.isclose(other, value, rel_tol=1e-4, abs_tol=1e-6) for other in values)
                better = sum(other > value and not math.isclose(other, value, rel_tol=1e-4, abs_tol=1e-6)
                             for other in values)
                rank = 1 + better + (ties - 1) / 2
                if row["label"] == "supported":
                    supported_ranks.append(rank)
            rows_out.append(dict(item_id=row["item_id"], query_id=query, label=row["label"],
                                 score=value, rank=rank, complete_query=is_complete,
                                 unmeasured_reason="graph_vocabulary_missing" if value is None else None))
        if is_complete:
            if not supported_ranks:
                raise ValidationError("query has no known support")
            complete += 1
            reciprocals.append(sum(1 / rank for rank in supported_ranks) / len(supported_ranks))
            top_hits.append(float(min(supported_ranks) == 1))
    total_items = sum(len(rows) for rows in groups.values())
    return dict(known_support_mrr=sum(reciprocals) / complete if complete else None,
                strict_known_support_top1=sum(top_hits) / complete if complete else None,
                complete_groups=complete, total_groups=len(groups), measured_items=measured_items,
                total_items=total_items, group_coverage=complete / len(groups) if groups else None,
                scientific_accuracy=None, rows=rows_out)


def structure_scores(edges, rows):
    neighbours = defaultdict(set)
    entities, relations = vocabulary(edges)
    for edge in edges:
        neighbours[edge["source_id"]].add(edge["target_id"])
        neighbours[edge["target_id"]].add(edge["source_id"])
    return {row["item_id"]: float(len(neighbours[row["source_id"]] & neighbours[row["target_id"]]))
            if measurable(row, entities, relations) else None for row in rows}


def validate_authorization(authorization, report):
    if (authorization.get("training_authorized") is not True
            or authorization.get("execution_sha256") != report["execution_sha256"]
            or authorization.get("ledger_path") != str(LEDGER.resolve())
            or not isinstance(authorization.get("user_request"), str)
            or not authorization["user_request"].strip()
            or authorization.get("prior_training_records_acknowledged", 0) < 24):
        raise BudgetExceeded("new explicit, source/code-bound allocation required; old budget is exhausted")
    if authorization.get("device") not in {"cpu", "cuda:0"}:
        raise BudgetExceeded("explicit cpu or cuda:0 device required")
    if authorization.get("max_runs") != 1 or authorization.get("prior_runs") != 0:
        raise BudgetExceeded("one new fixed experiment only; historical runs acknowledged separately")


def train_and_evaluate(bundle, guard, seconds, device, output):
    import numpy as np
    import psutil
    import torch
    from .gnn import GNNLinkPredictor, TripleIndex, directed_message_graph, sample_negatives
    from neurooracle.src.kge.triple_loader import Triple

    started = time.monotonic()
    peak_cpu = peak_gpu = 0
    process = psutil.Process()
    if device.startswith("cuda"):
        guard.configure_cuda(torch, 0)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    triples = [Triple(*values) for values in sorted({
        (edge["source_id"], edge["relation"], edge["target_id"]) for edge in bundle["edges"]})]
    index = TripleIndex.from_triples(triples)
    train_ids = index.encode(triples)
    graph_index, graph_type = directed_message_graph(train_ids, len(index.relation_to_id))
    graph_index, graph_type = graph_index.to(device), graph_type.to(device)
    model = GNNLinkPredictor(len(index.entity_to_id), len(index.relation_to_id),
                             **{key: CONFIG[key] for key in ("model", "embedding_dim", "layers", "dropout",
                                                            "decoder", "reverse_relations")}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    stopper = ValidationStopper(CONFIG["patience"])
    known = {tuple(row) for row in train_ids.tolist()}
    entities, relations = vocabulary(bundle["edges"])

    def check_resources():
        nonlocal peak_cpu, peak_gpu
        peak_cpu = max(peak_cpu, process.memory_info().rss)
        if device.startswith("cuda"):
            peak_gpu = max(peak_gpu, torch.cuda.max_memory_reserved(0))
        guard.check_usage(elapsed=time.monotonic() - started, gpu_bytes=peak_gpu,
                          cpu_bytes=peak_cpu, reserved_seconds=seconds)

    def evaluate(fold):
        groups = groups_for(bundle, fold)
        rows = [row for group in groups.values() for row in group]
        available = [row for row in rows if measurable(row, entities, relations)]
        scores = {row["item_id"]: None for row in rows}
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(graph_index, graph_type)
            for start in range(0, len(available), 256):
                chunk = available[start:start + 256]
                encoded = index.encode([Triple(row["source_id"], row["relation"], row["target_id"]) for row in chunk])
                values = model.score(embeddings, encoded.to(device)).cpu().tolist()
                scores.update({row["item_id"]: value for row, value in zip(chunk, values)})
                check_resources()
        return ranking_summary(groups, scores)

    history = []
    check_resources()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        negative = sample_negatives(train_ids, len(entities), known, 1, rng)
        if negative.numel() == 0:
            raise ValidationError("no training contrast samples available")
        embeddings = model.encode(graph_index, graph_type)
        positive_logits = model.score(embeddings, train_ids.to(device))
        negative_logits = model.score(embeddings, negative.to(device))
        loss = (torch.nn.functional.softplus(-positive_logits).mean()
                + torch.nn.functional.softplus(negative_logits).mean())
        if not torch.isfinite(loss):
            raise ValidationError("nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        check_resources()
        validation = evaluate("validation")
        checkpoint = io.BytesIO()
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
        history.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                            validation_mrr=validation["known_support_mrr"]))
        if stopper.observe(validation["known_support_mrr"], checkpoint.getvalue(), split="validation"):
            break
    model.load_state_dict(torch.load(io.BytesIO(stopper.checkpoint), weights_only=True, map_location=device))
    test = evaluate("test")
    test_groups = groups_for(bundle, "test")
    baseline = ranking_summary(test_groups, structure_scores(bundle["edges"],
                               [row for group in test_groups.values() for row in group]))
    check_resources()
    with (output / "checkpoint.pt").open("xb") as stream:
        torch.save(dict(state_dict=model.cpu().state_dict(), config=CONFIG,
                        entity_to_id=index.entity_to_id, relation_to_id=index.relation_to_id,
                        execution_sha256=execution_binding(bundle)), stream)
    return dict(status="LIMITED_RANKING_COMPARISON", config=CONFIG, history=history,
                gnn=test, structure=baseline,
                mrr_delta=test["known_support_mrr"] - baseline["known_support_mrr"],
                peak_observed_rss_bytes=peak_cpu, peak_cuda_reserved_bytes=peak_gpu,
                worker_seconds=time.monotonic() - started, scientific_acceptance=False,
                production_authorized=False, independent_scientific_validation=False)


def worker(output, attempt_id):
    bundle = read_json(output / "bundle.json")
    authorization = read_json(output / "authorization.json")
    report = check_bundle(bundle)
    validate_authorization(authorization, report)
    guard = RunGuard(LEDGER, authorization)
    seconds = guard.claim_worker(attempt_id, report["execution_sha256"])
    write_json(output / "WORKER_STARTED.json", {"attempt_id": attempt_id})
    result = train_and_evaluate(bundle, guard, seconds, authorization["device"], output)
    result["execution_sha256"] = report["execution_sha256"]
    write_json(output / "RESULT.json", result)


def launch(bundle, authorization, output):
    import psutil

    report = check_bundle(bundle)
    validate_authorization(authorization, report)
    output = Path(output).resolve()
    if output.parent != RUN_ROOT.resolve():
        raise ValidationError("output must be a new direct child of tmp/gnn_ranking")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    guard = RunGuard(LEDGER, authorization)
    output.mkdir(exist_ok=False)
    seconds = authorization["max_run_seconds"]
    attempt_id = guard.reserve(report["execution_sha256"], seconds)
    started = time.monotonic()
    try:
        write_json(output / "bundle.json", bundle)
        write_json(output / "authorization.json", authorization)
        write_json(output / "CHECK.json", report)
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            process = psutil.Popen([sys.executable, "-m", "models.kg_link_prediction.experiment",
                                    "_worker", "--output", str(output), "--attempt-id", str(attempt_id)],
                                   cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except BaseException:
        guard.finish(attempt_id, time.monotonic() - started, success=False)
        raise
    returncode = guard.supervise(process, attempt_id, started=started)
    if returncode == 0:
        result = read_json(output / "RESULT.json")
        if result.get("execution_sha256") != report["execution_sha256"]:
            raise ValidationError("result binding mismatch")
        write_json(output / "COMPLETE.json", {"attempt_id": attempt_id,
                   "execution_sha256": report["execution_sha256"], "scientific_acceptance": False})
    return returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="offline contract only; never trains")
    check.add_argument("--bundle", type=Path, required=True)
    run = commands.add_parser("run", help="requires a new explicit frozen allocation")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    child = commands.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--attempt-id", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(check_bundle(read_json(args.bundle)), ensure_ascii=False, allow_nan=False))
        return 0
    if args.command == "_worker":
        worker(args.output, args.attempt_id)
        return 0
    return launch(read_json(args.bundle), read_json(args.authorization), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
