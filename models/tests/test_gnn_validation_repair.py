import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.kg_link_prediction import validation as contract
from models.kg_link_prediction.audit_validation import audit, main as audit_main
from models.kg_link_prediction.run_guard import BudgetExceeded, RunGuard


def bound_item(label="supported"):
    text = "In adult mice, compound X decreased the measured endpoint Y."
    digest = hashlib.sha256(text.encode()).hexdigest()
    row = {
        "item_id": "case", "source_id": "compound:X", "relation": "decreases",
        "target_id": "endpoint:Y", "fact_family": "fact-1", "relation_family": "decreases",
        "measurement": "endpoint Y", "role": "primary_result", "scope": {"species": "mouse", "age": "adult"},
        "rationale": "Synthetic contract fixture, not a scientific label", "mapping_status": "reviewed_exact",
        "work_keys": ["work-1"], "source_families": ["family-1"], "label": label,
        "scope_match": True, "previously_exposed": False,
        "anchors": [{"source_sha256": digest, "start": 0, "end": len(text), "quote": text}],
    }
    sources = {digest: {"text": text, "work_key": "work-1", "source_family": "family-1", "fold": "corpus"}}
    return row, sources


def budget(**overrides):
    return dict(authorization_id="synthetic-test-only", max_runs=2, prior_runs=0,
                max_total_seconds=20, max_run_seconds=10, max_gpu_bytes=600,
                max_cpu_bytes=1200, **overrides)


def test_matching_does_not_find_action_inside_interaction_or_logic_inside_psychological():
    result = contract.literal_mentions("interaction and psychological assessment", {"a": ["action"], "b": ["logic"]})
    assert result["matches"] == []


def test_matching_retains_ambiguity_and_nested_names_without_resolving_them():
    result = contract.literal_mentions("GABA&#160;receptor in Brain", {
        "gaba": ["GABA"], "receptor": ["GABA receptor"], "brain-1": ["brain"], "brain-2": ["Brain"]})
    assert len(result["matches"]) == 3
    assert next(row for row in result["matches"] if row["alias"] == "brain")["identity_status"] == "ambiguous"
    assert result["automatic_entity_resolution"] is False


def test_source_binding_accepts_literal_raw_text():
    row, sources = bound_item()
    contract.check_item(row, sources, set())


@pytest.mark.parametrize("basis", [None, "reverse_not_mentioned", "null", "different_topic"])
def test_unasserted_reverse_and_null_never_become_negatives(basis):
    row, sources = bound_item("contradicted")
    row["negative_basis"] = basis
    with pytest.raises(contract.ValidationError, match="negative"):
        contract.check_item(row, sources, set())


@pytest.mark.parametrize("change", ["quote", "hash", "scope", "mapping", "heldout", "family"])
def test_source_identity_scope_and_mapping_fail_closed(change):
    row, sources = bound_item()
    source = next(iter(sources.values()))
    if change == "quote":
        row["anchors"][0]["quote"] = "wrong quote"
    elif change == "hash":
        source["text"] += " changed"
    elif change == "scope":
        row["scope_match"] = False
    elif change == "mapping":
        row["mapping_status"] = "substring"
    elif change == "heldout":
        source["fold"] = "heldout"
    else:
        row["source_families"] = ["other"]
    with pytest.raises(contract.ValidationError):
        contract.check_item(row, sources, set())


def test_protected_work_is_rejected_even_if_fold_claims_corpus():
    row, sources = bound_item()
    with pytest.raises(contract.ValidationError, match="heldout"):
        contract.check_item(row, sources, {"work-1"})


def test_family_split_unifies_inverse_synonym_and_multisource_bridge():
    first, _ = bound_item()
    second = dict(first, item_id="second", source_id=first["target_id"], target_id=first["source_id"],
                  relation="inverse", fact_family="other", work_keys=["work-2"], source_families=["family-2"])
    third = dict(first, item_id="third", source_id="third", target_id="fourth",
                 fact_family="fact-3", work_keys=["work-2", "work-3"], source_families=["family-2", "family-3"])
    rows = [first, second, third]
    assignments = contract.family_split(rows, "fixed-seed")
    assert len(set(assignments.values())) == 1
    assert contract.family_split(list(reversed(rows)), "fixed-seed") == assignments


def preflight_fixture():
    rows, sources, assignments = [], {}, {}
    for fold in ("train", "validation", "test"):
        for label in ("supported", "contradicted", "unresolved"):
            row, _ = bound_item(label)
            key = fold + "-" + label
            text = key + ": synthetic source fixture"
            digest = hashlib.sha256(text.encode()).hexdigest()
            row.update(item_id=key, source_id=key + "-source", target_id=key + "-target",
                       fact_family=key, work_keys=[key], source_families=[key])
            row["anchors"] = [{"source_sha256": digest, "start": 0, "end": len(text),
                               "quote": text, "stance": "contradicts_exact_scoped_claim"}]
            row["negative_basis"] = "explicit_source_contradiction"
            sources[digest] = {"text": text, "work_key": key, "source_family": key, "fold": "corpus"}
            rows.append(row)
            assignments[key] = fold
    edge = dict(rows[0], feature_work_keys=[], feature_provenance_complete=True, provenance_verified=True)
    return rows, sources, assignments, [edge]


def test_preflight_binds_manifest_but_never_grants_scientific_acceptance():
    rows, sources, assignments, edges = preflight_fixture()
    report = contract.preflight(rows, sources, assignments, edges, set(), set())
    assert len(report["binding_sha256"]) == 64
    assert report["training_authorized"] is False
    assert report["scientific_acceptance"] is False


@pytest.mark.parametrize("defect", ["exposure", "family", "message", "missing_class", "missing_fold"])
def test_preflight_blocks_unqualified_manifests(defect):
    rows, sources, assignments, edges = preflight_fixture()
    if defect == "exposure":
        rows[-1]["previously_exposed"] = True
    elif defect == "family":
        rows[-1]["source_families"] = rows[0]["source_families"]
    elif defect == "message":
        edges.append(dict(edges[0], source_id=rows[-1]["target_id"], target_id=rows[-1]["source_id"]))
    elif defect == "missing_class":
        rows[-1]["label"] = "supported"
    else:
        assignments = {key: "train" for key in assignments}
    with pytest.raises(contract.ValidationError):
        contract.preflight(rows, sources, assignments, edges, set(), set())


def test_graph_filter_blocks_inverse_family_feature_and_unknown_provenance():
    held, _ = bound_item()
    safe = {"source_id": "A", "target_id": "B", "relation": "rel", "fact_family": "clean",
            "work_keys": ["safe-work"], "source_families": ["safe-family"],
            "feature_work_keys": [], "feature_provenance_complete": True, "provenance_verified": True}
    bad = [dict(safe, source_id=held["target_id"], target_id=held["source_id"], relation="synonym"),
           dict(safe, source_families=["family-1"]), dict(safe, feature_work_keys=["work-1"]),
           dict(safe, provenance_verified=False), dict(safe, work_keys=["protected"])]
    result = contract.filter_training_graph([safe, *bad], [held], {"protected"})
    assert result["edges"] == [safe]
    assert len(result["excluded"]) == 5


def test_paths_reject_cycles_and_disconnected_reversals():
    first = {"source_id": "A", "relation": "assessed_by", "target_id": "B"}
    for second in ({"source_id": "B", "relation": "measures", "target_id": "A"},
                   {"source_id": "C", "relation": "rel", "target_id": "D"}):
        with pytest.raises(contract.ValidationError):
            contract.validate_path([first, second])
    contract.validate_path([first, {"source_id": "B", "relation": "rel", "target_id": "C"}])


def test_empty_metrics_are_null_and_ties_not_wins():
    assert contract.direction_summary([], 2)["strict_win_rate"] is None
    result = contract.direction_summary([(0.5, 0.50000001)], 2)
    assert result["wins"] == 0 and result["ties"] == 1 and result["unmeasured"] == 1
    assert result["scientific_accuracy"] is None


def test_stopper_uses_validation_and_preserves_best_checkpoint():
    stopper = contract.ValidationStopper(patience=2)
    assert not stopper.observe(0.6, b"best", split="validation")
    assert not stopper.observe(0.5, b"worse", split="validation")
    assert stopper.observe(0.5, b"worse-again", split="validation")
    assert stopper.checkpoint == b"best"
    with pytest.raises(contract.ValidationError):
        stopper.observe(0.9, b"test-leak", split="test")


def test_budget_survives_restart_no_resend_or_reset(tmp_path):
    path = tmp_path / "ledger.sqlite"
    guard = RunGuard(path, budget())
    attempt = guard.reserve("seed-1", 10)
    restarted = RunGuard(path, budget())
    with pytest.raises(BudgetExceeded):
        restarted.reserve("new-tag", 10)
    guard.finish(attempt, 1, success=True)
    with pytest.raises(BudgetExceeded):
        restarted.reserve("seed-1", 10)
    second = restarted.reserve("seed-2", 10)
    restarted.finish(second, 1, success=True)
    with pytest.raises(BudgetExceeded):
        restarted.reserve("seed-3", 1)
    changed = budget()
    changed["max_runs"] = 100
    with pytest.raises(BudgetExceeded):
        RunGuard(path, changed)


def test_prior_run_overrun_blocks_any_new_training(tmp_path):
    authorization = budget()
    authorization.update(prior_runs=24, max_runs=6)
    guard = RunGuard(tmp_path / "ledger.sqlite", authorization)
    with pytest.raises(BudgetExceeded, match="exhausted"):
        guard.reserve("retry", 1)


def test_concurrent_reservation_cannot_double_spend(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "ledger.sqlite"
    RunGuard(path, budget())

    def reserve(key):
        try:
            return RunGuard(path, budget()).reserve(key, 10)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(reserve, ["first", "second"]))
    assert sum(attempt is not None for attempt in attempts) == 1


def test_failed_attempt_stays_held_across_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    guard = RunGuard(path, budget())
    attempt = guard.reserve("failed", 10)
    guard.finish(attempt, 2, success=False)
    with pytest.raises(BudgetExceeded, match="review"):
        RunGuard(path, budget()).reserve("replacement", 10)


@pytest.mark.parametrize("reading", [{"elapsed": 10}, {"gpu_bytes": 601}, {"cpu_bytes": 1201}])
def test_resource_caps(reading, tmp_path):
    guard = RunGuard(tmp_path / "ledger.sqlite", budget())
    values = dict(elapsed=0, gpu_bytes=0, cpu_bytes=0, reserved_seconds=10)
    values.update(reading)
    with pytest.raises(BudgetExceeded):
        guard.check_usage(**values)


def test_cuda_allocator_cap_before_model_allocation(tmp_path):
    fractions = []
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(total_memory=1000),
                           set_per_process_memory_fraction=lambda fraction, device: fractions.append(fraction))
    RunGuard(tmp_path / "ledger.sqlite", budget()).configure_cuda(SimpleNamespace(cuda=cuda))
    assert fractions == [0.6]


def test_supervisor_kills_owned_worker_on_rss_overrun_and_holds(tmp_path):
    class Worker:
        returncode = None

        def poll(self):
            return self.returncode

        def memory_info(self):
            return SimpleNamespace(rss=1201)

        def kill(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    guard = RunGuard(tmp_path / "ledger.sqlite", budget())
    attempt = guard.reserve("owned-worker", 10)
    worker = Worker()
    with pytest.raises(BudgetExceeded):
        guard.supervise(worker, attempt)
    assert worker.returncode == -1
    with pytest.raises(BudgetExceeded):
        guard.reserve("retry", 1)


def test_existing_reference_is_withdrawn_without_new_labels_or_overwrite(tmp_path):
    directory = Path("tmp/gnn_validation_20260927")
    if not directory.exists():
        pytest.skip("local diagnostic artifacts unavailable")
    report = audit(directory)
    assert report["status"] == "HELD_NO_TRAINING"
    assert all(row["new_scientific_label"] is None for row in report["corrected_items"])
    assert report["training_slots_remaining"] == 0
    output = tmp_path / "correction.json"
    audit_main(["--input-dir", str(directory), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        audit_main(["--input-dir", str(directory), "--output", str(output)])
