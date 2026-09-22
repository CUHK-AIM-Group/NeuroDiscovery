"""Offline tests for published defaults and explicitly pinned graph inputs."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from neurooracle.src import graph_paths
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.storage import load_graph, save_graph


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def fingerprint(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROCLAW_KG_CAMPAIGN", raising=False)
    monkeypatch.setattr(graph_paths, "REPO_ROOT", tmp_path)
    graph = write_json(tmp_path / "accepted/graph.json", {"concepts": {}, "edges": []})
    receipt = write_json(tmp_path / "accepted/CURRENT_ACCEPTANCE.json", dict(
        graph=fingerprint(graph), checks={"independent_full_structure_scan": True}))
    campaign = write_json(tmp_path / "accepted/CAMPAIGN.json", dict(
        status="COMPLETED", active_process=None, current_graph=fingerprint(graph),
        current_acceptance=fingerprint(receipt)))
    config = write_json(tmp_path / "neurooracle/configs/graph_explorer.json", dict(
        version=1, campaign="accepted/CAMPAIGN.json"))
    return dict(root=tmp_path, graph=graph, receipt=receipt, campaign=campaign, config=config)


def add_formal(release):
    path = write_json(release["root"] / "neurooracle/data/full_v2/knowledge_graph.json",
                      {"concepts": {}, "edges": []})
    write_json(path.parent / "CURRENT_STATE.json", dict(status="canonical_current",
        canonical_files={"knowledge_graph": {"path": str(path), "bytes": path.stat().st_size}}))
    return path


def update_receipt(release, receipt):
    write_json(release["receipt"], receipt)
    campaign = json.loads(release["campaign"].read_text())
    campaign["current_acceptance"] = fingerprint(release["receipt"])
    write_json(release["campaign"], campaign)


def test_published_campaign_wins_over_base_and_newer_work_files(release):
    add_formal(release)
    write_json(release["root"] / "unreviewed_latest/knowledge_graph.json", {"not": "accepted"})
    assert graph_paths.current_graph_path() == release["graph"]


def test_formal_used_only_without_a_configured_campaign(release):
    formal = add_formal(release)
    release["config"].unlink()
    assert graph_paths.current_graph_path() == formal


def test_missing_configured_campaign_never_falls_back(release):
    add_formal(release)
    release["campaign"].unlink()
    with pytest.raises(FileNotFoundError):
        graph_paths.current_graph_path()


@pytest.mark.parametrize("status,active", [("RUNNING", None), ("STOPPED", None), ("COMPLETED", 123)])
def test_unready_publication_is_rejected(release, status, active):
    payload = json.loads(release["campaign"].read_text())
    payload.update(status=status, active_process=active)
    write_json(release["campaign"], payload)
    with pytest.raises(ValueError, match="not ready"):
        graph_paths.current_graph_path()


@pytest.mark.parametrize("change", ["graph", "proof"])
def test_matching_acceptance_is_required(release, change):
    receipt = json.loads(release["receipt"].read_text())
    if change == "graph":
        receipt["graph"]["sha256"] = "0" * 64
    else:
        receipt["checks"]["independent_full_structure_scan"] = False
    update_receipt(release, receipt)
    with pytest.raises(ValueError, match="validated acceptance"):
        graph_paths.current_graph_path()


def test_graph_fingerprint_change_is_rejected(release):
    release["graph"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint changed"):
        graph_paths.current_graph_path()


def test_receipt_content_is_hashed(release):
    content = release["receipt"].read_bytes()
    tampered = content.replace(b"true", b"null")
    assert len(tampered) == len(content)
    old = release["receipt"].stat()
    release["receipt"].write_bytes(tampered)
    import os
    os.utime(release["receipt"], ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="integrity check"):
        graph_paths.current_graph_path()


def test_resolution_tracks_next_accepted_revision(release):
    assert graph_paths.current_graph_path() == release["graph"]
    newer = write_json(release["root"] / "accepted_v2/graph.json", {"concepts": {}, "edges": []})
    receipt = json.loads(release["receipt"].read_text())
    receipt["graph"] = fingerprint(newer)
    update_receipt(release, receipt)
    campaign = json.loads(release["campaign"].read_text())
    campaign["current_graph"] = fingerprint(newer)
    write_json(release["campaign"], campaign)
    assert graph_paths.current_graph_path() == newer


def test_environment_pointer_uses_the_same_acceptance_checks(release, monkeypatch):
    release["config"].unlink()
    monkeypatch.setenv("NEUROCLAW_KG_CAMPAIGN", str(release["campaign"]))
    assert graph_paths.current_graph_path() == release["graph"]
    monkeypatch.setenv("NEUROCLAW_KG_CAMPAIGN", "")
    with pytest.raises(ValueError, match="Empty"):
        graph_paths.current_graph_path()


def test_explicit_frozen_input_does_not_consult_live_campaign(release):
    frozen = write_json(release["root"] / "frozen/graph.json", {"concepts": {}, "edges": []})
    release["campaign"].unlink()
    assert graph_paths.resolve_graph_path(frozen) == frozen
    assert len(load_graph(frozen)) == 0


def test_explicit_compressed_graph(release):
    path = release["root"] / "frozen.json"
    with gzip.open(str(path) + ".gz", "wt", encoding="utf-8") as handle:
        json.dump({"concepts": {}, "edges": []}, handle)
    assert graph_paths.resolve_graph_path(path) == Path(str(path) + ".gz")
    assert len(load_graph(path)) == 0


def test_missing_graph_is_not_silently_empty(release):
    missing = release["root"] / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_graph(missing)
    assert len(load_graph(missing, allow_missing=True)) == 0
    release["campaign"].unlink()
    with pytest.raises(FileNotFoundError):
        load_graph(allow_missing=True)


def test_default_load_and_write_are_separate(release):
    before = release["graph"].read_bytes()
    assert len(load_graph()) == 0
    with pytest.raises(ValueError, match="explicit output"):
        save_graph(KnowledgeGraph())
    assert release["graph"].read_bytes() == before
    output = release["root"] / "new_build.json"
    save_graph(KnowledgeGraph(), output)
    assert output.is_file()


def test_separate_output_rejects_same_file(release):
    with pytest.raises(ValueError, match="separate --output"):
        graph_paths.separate_graph_output(release["graph"], None)
    with pytest.raises(ValueError, match="must differ"):
        graph_paths.separate_graph_output(release["graph"], release["graph"])
    out = release["root"] / "draft.json"
    assert graph_paths.separate_graph_output(release["graph"], out) == out


def test_formal_state_validation(release):
    formal = add_formal(release)
    release["config"].unlink()
    formal.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        graph_paths.current_graph_path()


def test_cli_passes_published_path_to_new_host_run(release, monkeypatch):
    import sys
    from neurooracle.src import hypothesis_cli, host_agent_autoresearch
    captured = {}
    def init(**kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(host_agent_autoresearch, "init_run", init)
    monkeypatch.setattr(sys, "argv", ["cli", "host-agent-init", "case1_transdiagnostic",
                                     "--output-dir", str(release["root"] / "run")])
    hypothesis_cli.main()
    assert captured["graph_path"] == str(release["graph"])
    assert captured["kge_path"] is None


def test_cli_requires_explicit_checkpoint_before_any_experiment(release, monkeypatch):
    import sys
    from neurooracle.src import hypothesis_cli
    monkeypatch.setattr(sys, "argv", ["cli", "case-study", "case1_transdiagnostic",
                                     "--output-dir", str(release["root"] / "run")])
    with pytest.raises(SystemExit) as exc:
        hypothesis_cli.main()
    assert exc.value.code == 2
    assert not (release["root"] / "run").exists()


def test_no_fixed_snapshot_defaults_in_reading_entrypoints():
    root = Path(__file__).resolve().parents[2]
    for name in ("neurooracle/src/storage.py", "neurooracle/src/hypothesis_cli.py",
                 "neurooracle/run_cycle.sh", "neurooracle/run_case_study.sh",
                 "skills/knowledge-graph-builder/scripts/graph_query.py",
                 "skills/knowledge-graph-builder/scripts/new_data_source_template.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "full_snapshot_v1" not in source, name
        assert "full_snapshot_v2" not in source, name
