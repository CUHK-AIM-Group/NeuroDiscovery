from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts.case1_autoresearch_controllers import (
    BrainPilotController,
    _biomni_tool_audit,
)
from core.scripts.case1_autoresearch_protocol import (
    HIDDEN_OUTCOME_FIELDS,
    Case1AutoresearchProtocol,
    blinded_public_registry,
    classify_outcome,
    evaluate_prefix,
    menu_indices,
    reveal_feedback,
    validate_selected_ids,
    write_selection_commitment,
)
from core.scripts.biomni_case1_closed_loop_worker import _execution_audit
from core.scripts.run_case1_full_autoresearch_comparison import (
    NeuroDiscoveryController,
    OutcomeVault,
    load_completed_seed_artifacts,
    pairwise_selection_overlap,
)


def _candidates(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index}" for index in range(n)],
            "disease": ["MDD"] * n,
            "modality": ["fmri"] * n,
            "source": ["atlas"] * n,
            "roi_index": list(range(n)),
            "anatomy_full": [f"region-{index}" for index in range(n)],
            "hemisphere": ["left"] * n,
            "network": ["default"] * n,
            "structure_class": ["cortical"] * n,
            "feature": ["alff"] * n,
            "feature_family": ["amplitude"] * n,
            "adjusted_residual_d": [0.2] * n,
            "p_value": [0.001] * n,
            "execution_succeeded": [True] * n,
            "is_gt_top": [False] * n,
            "score_neurodiscovery": [0.5] * n,
            "kg_pair_support": [2.0] * n,
        }
    )


def _write_completed_seed(
    root: Path,
    *,
    method: str,
    seed: int,
    protocol: Case1AutoresearchProtocol,
) -> Path:
    seed_dir = root / method / f"seed_{seed:02d}"
    slots = pd.DataFrame(
        {
            "method": [method] * protocol.experiment_slots,
            "seed": [seed] * protocol.experiment_slots,
            "round": [
                index // protocol.batch_size
                for index in range(protocol.experiment_slots)
            ],
            "slot": [
                index % protocol.batch_size + 1
                for index in range(protocol.experiment_slots)
            ],
            "valid": [True] * protocol.experiment_slots,
            "execution_succeeded": [True] * protocol.experiment_slots,
            "is_gt_top": [False] * protocol.experiment_slots,
        }
    )
    seed_dir.mkdir(parents=True)
    slots.to_csv(seed_dir / "slots.csv", index=False)
    pd.DataFrame(
        {
            "method": [method] * protocol.rounds,
            "seed": [seed] * protocol.rounds,
            "budget": list(
                range(
                    protocol.batch_size,
                    protocol.experiment_slots + 1,
                    protocol.batch_size,
                )
            ),
        }
    ).to_csv(seed_dir / "direct_curve.csv", index=False)
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "case1-autoresearch-seed-run.v1",
                "method": method,
                "seed": seed,
                "protocol": protocol.to_dict(),
                "requested_slots": protocol.experiment_slots,
            }
        ),
        encoding="utf-8",
    )
    for round_index in range(protocol.rounds):
        round_dir = seed_dir / f"round_{round_index + 1:02d}"
        round_dir.mkdir()
        commitment = {
            "method": method,
            "seed": seed,
            "round": round_index,
            "requested_slots": protocol.batch_size,
            "outcomes_read_before_commit": False,
            "menu_sha256": f"menu-{round_index}",
            "commit_sha256": f"commit-{round_index}",
        }
        (round_dir / "selection_commitment.json").write_text(
            json.dumps(commitment), encoding="utf-8"
        )
        (round_dir / "round_audit.json").write_text(
            json.dumps(
                {
                    "selection_commitment_sha256": f"commit-{round_index}",
                    "menu_sha256": f"menu-{round_index}",
                    "outcomes_revealed_after_commitment": True,
                }
            ),
            encoding="utf-8",
        )
        (round_dir / "experimental_feedback.json").write_text(
            "[]", encoding="utf-8"
        )
    return seed_dir


def test_resume_reuses_only_fully_committed_seed_artifacts(tmp_path: Path) -> None:
    protocol = Case1AutoresearchProtocol(rounds=2, batch_size=2, menu_multiplier=2)
    seed_dir = _write_completed_seed(
        tmp_path, method="neurodiscovery", seed=3, protocol=protocol
    )

    reused = load_completed_seed_artifacts(
        output_dir=tmp_path,
        method="neurodiscovery",
        seed=3,
        protocol=protocol,
    )
    assert reused is not None
    assert len(reused[0]) == protocol.experiment_slots

    (seed_dir / "round_02" / "round_audit.json").unlink()
    with pytest.raises(RuntimeError, match="round artifacts"):
        load_completed_seed_artifacts(
            output_dir=tmp_path,
            method="neurodiscovery",
            seed=3,
            protocol=protocol,
        )


def test_resume_treats_seed_without_manifest_as_incomplete(tmp_path: Path) -> None:
    partial = tmp_path / "sciagents" / "seed_00" / "round_01"
    partial.mkdir(parents=True)
    assert (
        load_completed_seed_artifacts(
            output_dir=tmp_path,
            method="sciagents",
            seed=0,
            protocol=Case1AutoresearchProtocol(),
        )
        is None
    )


def test_neurodiscovery_records_selection_change_against_static_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOverlay:
        def __init__(self) -> None:
            self.changes: list[bool] = []

        def score_candidates(self, **_: object) -> tuple[object, dict[str, object]]:
            return pd.Series([0.0, 2.0]).to_numpy(), {"overlay_read": True}

        def note_selection_change(self, changed: bool) -> None:
            self.changes.append(changed)

    def select_max(
        indices: object,
        score: object,
        *_: object,
        **__: object,
    ) -> object:
        values = pd.Series(score).to_numpy()
        candidates = pd.Series(indices).to_numpy(dtype=int)
        return candidates[[int(values[candidates].argmax())]]

    monkeypatch.setattr(
        "core.scripts.run_case1_full_autoresearch_comparison._select_diverse_batch",
        select_max,
    )
    controller = NeuroDiscoveryController.__new__(NeuroDiscoveryController)
    controller.public = pd.DataFrame(
        {
            "candidate_id": ["static", "feedback"],
            "disease": ["MDD", "MDD"],
            "feature_family": ["amplitude", "amplitude"],
            "source": ["atlas", "atlas"],
        }
    )
    controller.static_score = pd.Series([1.0, 0.0]).to_numpy()
    controller.static_audit = {}
    controller.rng = np.random.default_rng(7)
    controller.overlay = FakeOverlay()

    selection = controller.select(
        menu_candidate_indices=np.array([0, 1]),
        start_rank=1,
        requested_slots=1,
        round_index=1,
    )

    assert selection.items[0]["candidate_id"] == "feedback"
    assert selection.metadata["selection_changed_from_static"] is True
    assert selection.metadata["static_counterfactual_overlap"] == 0
    assert controller.overlay.changes == [True]


def test_blinded_registry_removes_every_hidden_field() -> None:
    public = blinded_public_registry(_candidates())
    assert not (set(public) & HIDDEN_OUTCOME_FIELDS)
    assert not any(column.startswith(("score_", "kg_")) for column in public)


def test_round_menus_are_deterministic_shared_and_non_repeating() -> None:
    protocol = Case1AutoresearchProtocol(rounds=2, batch_size=2, menu_multiplier=3)
    first = menu_indices(20, seed=7, protocol=protocol)
    second = menu_indices(20, seed=7, protocol=protocol)
    assert [values.tolist() for values in first] == [values.tolist() for values in second]
    assert set(first[0]).isdisjoint(set(first[1]))


def test_invalid_or_missing_output_consumes_slots_without_repair() -> None:
    selected = [
        {
            "rank": 1,
            "candidate_id": "candidate-1",
            "rationale": "testable",
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "candidate_id": "off-menu",
            "rationale": "invalid",
            "confidence": 0.7,
        },
    ]
    accepted, audit = validate_selected_ids(
        selected,
        menu_candidate_ids={"candidate-1", "candidate-2"},
        previously_executed=set(),
        start_rank=1,
        requested_slots=3,
    )
    assert accepted == ["candidate-1"]
    assert [row["status"] for row in audit] == [
        "valid",
        "off_menu_candidate_id",
        "missing_rank;invalid_confidence;off_menu_candidate_id;missing_rationale",
    ]


def test_brainpilot_resume_does_not_reuse_a_stale_service_session(tmp_path) -> None:
    round_dir = tmp_path / "round_01"
    round_dir.mkdir()
    (round_dir / "final.txt").write_text(
        json.dumps(
            {
                "method": "brainpilot_native",
                "hypotheses": [
                    {
                        "rank": 1,
                        "candidate_id": "candidate-1",
                        "rationale": "testable",
                        "confidence": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "client_meta.json").write_text(
        json.dumps({"session_id": "stale-session"}), encoding="utf-8"
    )
    controller = BrainPilotController.__new__(BrainPilotController)
    controller.method = "brainpilot_native"
    controller.secret = "secret"
    controller.model = "model"
    controller.reasoning_effort = "high"
    controller.service_url = "http://unused"
    controller.timeout_seconds = 1
    controller.repo = tmp_path
    controller.commit = "test"
    controller.session_id = None

    result = controller.select(
        task={}, round_dir=round_dir, prompt="unused", start_rank=1
    )

    assert len(result.items) == 1
    assert controller.session_id is None


def test_brainpilot_client_ignores_replayed_prior_round_events(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select the next hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
export class BrainPilotClient {
  async createSession() { return "session-1"; }
  async sendMessage() {
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    yield { type: "RUN_STARTED", run_id: "old-run" };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "old-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "candidate-old",
          rationale: "stale result",
          confidence: 0.5,
        }],
      }),
    };
    yield { type: "RUN_FINISHED", run_id: "old-run" };
    yield {
      type: "CUSTOM",
      name: "session_state",
      _ts: "2026-08-22T00:00:01Z",
      value: { runState: { active: false, runId: null } },
    };
    yield { type: "RUN_FINISHED", run_id: "old-trace", agent_name: "trace" };
    yield {
      type: "CUSTOM",
      name: "session_state",
      _ts: "2026-08-22T00:00:02Z",
      value: { runState: { active: false, runId: null } },
    };
    yield {
      type: "RUN_STARTED",
      run_id: "new-principal-run",
      agent_name: "principal",
    };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "new-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "candidate-new",
          rationale: "fresh result",
          confidence: 0.8,
        }],
      }),
    };
    yield {
      type: "RUN_FINISHED",
      run_id: "new-principal-run",
      agent_name: "principal",
    };
    yield {
      type: "CUSTOM",
      name: "session_state",
      _ts: "2026-08-22T00:00:03Z",
      value: { runState: { active: false, runId: null } },
    };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
            "--session-id",
            "session-1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    result = json.loads((out_dir / "final.txt").read_text(encoding="utf-8"))
    metadata = json.loads(
        (out_dir / "client_meta.json").read_text(encoding="utf-8")
    )
    assert result["hypotheses"][0]["candidate_id"] == "candidate-new"
    assert metadata["transport_run_id"] == "transport-run"
    assert metadata["principal_run_id"] == "new-principal-run"
    assert metadata["idle_boundary_observed"] is True
    assert metadata["replayed_events_drained"] == 4
    assert metadata["post_send_events_skipped"] == 2


def test_brainpilot_client_sends_before_streaming_a_new_session(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
let messageSent = false;
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    messageSent = true;
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    if (!messageSent) throw new Error("stream was consumed before sendMessage");
    yield {
      type: "RUN_STARTED",
      run_id: "principal-run",
      agent_name: "principal",
    };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "delegation-message",
      delta: "Delegating evidence review to specialist agents.",
    };
    yield {
      type: "RUN_FINISHED",
      run_id: "principal-run",
      agent_name: "principal",
    };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "librarian",
      message_id: "child-message",
      delta: "Evidence review complete.",
    };
    yield {
      type: "RUN_STARTED",
      run_id: "principal-summary-run",
      agent_name: "principal",
    };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "summary-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "candidate-new",
          rationale: "fresh result",
          confidence: 0.8,
        }],
      }),
    };
    yield {
      type: "RUN_FINISHED",
      run_id: "principal-summary-run",
      agent_name: "principal",
    };
    yield {
      type: "CUSTOM",
      name: "session_state",
      value: { runState: { active: false, runId: null } },
    };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    metadata = json.loads(
        (out_dir / "client_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["session_reused"] is False
    assert metadata["principal_run_id"] == "principal-run"
    assert metadata["principal_run_ids"] == [
        "principal-run",
        "principal-summary-run",
    ]
    assert metadata["idle_boundary_observed"] is False
    assert metadata["replayed_events_drained"] == 0


def test_brainpilot_client_preserves_identical_consecutive_stream_chunks(
    tmp_path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate--new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
const payload = JSON.stringify({
  method: "brainpilot_native",
  hypotheses: [{
    rank: 1,
    candidate_id: "candidate--new",
    rationale: "verbatim candidate survives character streaming",
    confidence: 0.88,
  }],
});
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    yield { type: "RUN_STARTED", run_id: "principal-run", agent_name: "principal" };
    for (const delta of payload) {
      yield {
        type: "TEXT_MESSAGE_CONTENT",
        agent_name: "principal",
        message_id: "candidate-message",
        delta,
      };
    }
    yield {
      type: "CUSTOM",
      name: "session_state",
      value: { runState: { active: false, runId: null } },
    };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    result = json.loads((out_dir / "final.txt").read_text(encoding="utf-8"))
    assert result["hypotheses"][0]["candidate_id"] == "candidate--new"


def test_brainpilot_client_rejects_placeholder_candidate_outside_menu(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    yield { type: "RUN_STARTED", run_id: "principal-run", agent_name: "principal" };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "template-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "<verbatim menu ID>",
          rationale: "<one sentence>",
          confidence: 0.75,
        }],
      }),
    };
    yield {
      type: "CUSTOM",
      name: "session_state",
      value: { runState: { active: false, runId: null } },
    };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    result = subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "no parseable native candidate artifact" in result.stderr
    assert not (out_dir / "final.txt").exists()


def test_brainpilot_client_refuses_partial_output_at_active_event_limit(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    yield { type: "RUN_STARTED", run_id: "principal-run", agent_name: "principal" };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "candidate-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "candidate-new",
          rationale: "valid but not final",
          confidence: 0.8,
        }],
      }),
    };
    while (true) yield { type: "CUSTOM", name: "heartbeat" };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    result = subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
            "--max-events",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "remained active at event limit" in result.stderr
    assert len(json.loads((out_dir / "events.json").read_text(encoding="utf-8"))) == 2
    assert not (out_dir / "final.txt").exists()


def test_brainpilot_client_reconnects_without_resending_round_message(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("select hypotheses", encoding="utf-8")
    menu_path = tmp_path / "menu.csv"
    menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_client = tmp_path / "mock-client.mjs"
    mock_client.write_text(
        """
let streamCalls = 0;
let sendCalls = 0;
const started = {
  type: "RUN_STARTED",
  run_id: "principal-run",
  agent_name: "principal",
};
const candidate = {
  type: "TEXT_MESSAGE_CONTENT",
  agent_name: "principal",
  message_id: "candidate-message",
  delta: JSON.stringify({
    method: "brainpilot_native",
    hypotheses: [{
      rank: 1,
      candidate_id: "candidate-new",
      rationale: "survives a transport reconnect",
      confidence: 0.8,
    }],
  }),
};
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    sendCalls += 1;
    if (sendCalls > 1) throw new Error("round message was resent");
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    streamCalls += 1;
    yield started;
    yield candidate;
    if (streamCalls === 1) throw new Error("simulated socket termination");
    yield {
      type: "CUSTOM",
      name: "session_state",
      value: { runState: { active: false, runId: null } },
    };
  }
}
""",
        encoding="utf-8",
    )
    client_script = (
        Path(__file__).resolve().parents[1]
        / "brainpilot_case_study_batch_client.mjs"
    )

    subprocess.run(
        [
            node,
            str(client_script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(out_dir),
            "--client-dist",
            str(mock_client),
            "--menu",
            str(menu_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    metadata = json.loads(
        (out_dir / "client_meta.json").read_text(encoding="utf-8")
    )
    events = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    assert metadata["stream_reconnects"] == 1
    assert len(events) == 3
    assert json.loads((out_dir / "final.txt").read_text(encoding="utf-8"))[
        "hypotheses"
    ][0]["candidate_id"] == "candidate-new"


def test_brainpilot_client_uses_authoritative_idle_state_when_sse_stalls(
    tmp_path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the BrainPilot client integration test")

    class StateHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"runState": {"active": False, "runId": None}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StateHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("select hypotheses", encoding="utf-8")
        menu_path = tmp_path / "menu.csv"
        menu_path.write_text("candidate_id\ncandidate-new\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        mock_client = tmp_path / "mock-client.mjs"
        mock_client.write_text(
            """
export class BrainPilotClient {
  async createSession() { return "new-session"; }
  async sendMessage() {
    return { accepted: true, runId: "transport-run", queued: false };
  }
  async *streamEvents() {
    yield { type: "RUN_STARTED", run_id: "principal-run", agent_name: "principal" };
    yield {
      type: "TEXT_MESSAGE_CONTENT",
      agent_name: "principal",
      message_id: "candidate-message",
      delta: JSON.stringify({
        method: "brainpilot_native",
        hypotheses: [{
          rank: 1,
          candidate_id: "candidate-new",
          rationale: "complete before the SSE terminal event is lost",
          confidence: 0.8,
        }],
      }),
    };
    await new Promise(() => {});
  }
}
""",
            encoding="utf-8",
        )
        client_script = (
            Path(__file__).resolve().parents[1]
            / "brainpilot_case_study_batch_client.mjs"
        )

        subprocess.run(
            [
                node,
                str(client_script),
                "--prompt",
                str(prompt_path),
                "--out",
                str(out_dir),
                "--client-dist",
                str(mock_client),
                "--menu",
                str(menu_path),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/api",
                "--state-poll-ms",
                "20",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    events = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    assert events[-1]["_brainpilot_state_poll"] is True
    assert json.loads((out_dir / "final.txt").read_text(encoding="utf-8"))[
        "hypotheses"
    ][0]["candidate_id"] == "candidate-new"


def test_biomni_tool_audit_records_real_execution_and_retrieval(tmp_path) -> None:
    path = tmp_path / "biomni_execution_audit.json"
    path.write_text(
        json.dumps(
            {
                "audit_source": "biomni_a1_execution_results",
                "executed_blocks": 1,
                "pubmed_query_calls": 1,
                "pubmed_result_titles": 1,
                "execution_errors": 0,
            }
        ),
        encoding="utf-8",
    )

    audit = _biomni_tool_audit(path)

    assert audit["executed_blocks"] == 1
    assert audit["pubmed_query_calls"] == 1
    assert audit["pubmed_result_titles"] == 1
    assert audit["execution_errors"] == 0


def test_biomni_execution_audit_ignores_unexecuted_drafted_blocks() -> None:
    log = [
        "<execute>query_pubmed(query='drafted')</execute>",
        "<observation>Title: Executed result</observation>",
    ]
    execution_entries = [
        {"triggering_message": "<execute>print('plan')</execute>"}
    ]

    audit = _execution_audit(log, execution_entries)

    assert audit["executed_blocks"] == 1
    assert audit["pubmed_query_calls"] == 0
    assert audit["pubmed_result_titles"] == 0


def test_feedback_does_not_reveal_gt_and_commit_precedes_reveal(tmp_path) -> None:
    protocol = Case1AutoresearchProtocol(rounds=1, batch_size=1, menu_multiplier=1)
    commit_path = tmp_path / "selection_commitment.json"
    commit = write_selection_commitment(
        commit_path,
        method="test",
        seed=0,
        round_index=0,
        requested_slots=1,
        selected_candidate_ids=["candidate-1"],
        validation_rows=[{"rank": 1, "valid": True}],
        menu_sha256="abc",
    )
    assert commit_path.is_file()
    assert json.loads(commit_path.read_text())["outcomes_read_before_commit"] is False

    outcomes = {
        "candidate-1": {
            "adjusted_residual_d": -0.4,
            "p_value": 0.001,
            "execution_succeeded": True,
            "is_gt_top": True,
        }
    }
    feedback = reveal_feedback(
        ["candidate-1"], outcomes, round_index=0, protocol=protocol
    )
    assert feedback[0]["status"] == "supported"
    assert feedback[0]["observed_direction"] == "case_lower"
    assert "is_gt_top" not in feedback[0]
    assert commit["selected_candidate_ids"] == ["candidate-1"]


def test_support_rule_and_slot_level_metrics() -> None:
    protocol = Case1AutoresearchProtocol()
    assert (
        classify_outcome(
            {
                "adjusted_residual_d": 0.2,
                "p_value": 0.001,
                "execution_succeeded": True,
            },
            protocol,
        )
        == "supported"
    )
    slots = pd.DataFrame(
        {
            "slot": [1, 2, 3],
            "valid": [True, False, True],
            "execution_succeeded": [True, False, False],
            "is_gt_top": [True, False, False],
            "supported": [True, False, False],
        }
    )
    result = evaluate_prefix(slots, requested_budgets=[3]).iloc[0]
    assert result["format_success_rate"] == 2 / 3
    assert result["execution_success_rate_per_slot"] == 1 / 3
    assert result["gt_precision_per_slot"] == 1 / 3


def test_pairwise_overlap_distinguishes_equal_counts_from_equal_selections() -> None:
    slots = pd.DataFrame(
        {
            "seed": [0] * 6,
            "method": ["a", "a", "a", "b", "b", "b"],
            "candidate_id": ["x", "y", "z", "x", "u", "v"],
            "valid": [True] * 6,
        }
    )

    result = pairwise_selection_overlap(slots).iloc[0]

    assert result["intersection"] == 1
    assert result["union"] == 5
    assert result["jaccard"] == pytest.approx(0.2)
    assert result["overlap_fraction_smaller"] == pytest.approx(1 / 3)


def test_outcome_vault_requires_exact_prior_commitment(tmp_path) -> None:
    candidates = _candidates(3)
    candidates.loc[0, "is_gt_top"] = True
    vault = OutcomeVault(candidates)
    assert vault.gt_total == 1
    commitment_path = tmp_path / "selection_commitment.json"

    with pytest.raises(RuntimeError, match="before selection commitment"):
        vault.reveal(["candidate-1"], commitment_path=commitment_path)

    write_selection_commitment(
        commitment_path,
        method="test",
        seed=0,
        round_index=0,
        requested_slots=1,
        selected_candidate_ids=["candidate-1"],
        validation_rows=[{"rank": 1, "valid": True}],
        menu_sha256="abc",
    )
    with pytest.raises(RuntimeError, match="does not match committed candidates"):
        vault.reveal(["candidate-2"], commitment_path=commitment_path)

    revealed = vault.reveal(["candidate-1"], commitment_path=commitment_path)
    assert revealed["candidate-1"]["adjusted_residual_d"] == 0.2
