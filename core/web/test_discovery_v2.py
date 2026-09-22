"""Synthetic respondent tests and read-only presentation-to-source checks."""
import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from core.web.build_discovery_pilot_v2 import build, select_additions, V1_HASH
from core.web.discovery_scoring import score_session, aggregate_exports
from core.web.discovery_study import DiscoveryStudy, StudyError, create_preview_app

DEFAULT_PACK = Path(__file__).resolve().parent / "study_materials" / "cs1_discovery_pilot_v2.json"
V1 = DEFAULT_PACK.with_name("cs1_discovery_pilot_v1.json")
SOURCE = Path("C:/Users/45846/Downloads/markdown/CaseStudy1_UserStudy_Handoff_20260915")
PROFILE = {"code": "TEST-V2", "experience": "3-5", "consent": True}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.service = DiscoveryStudy(DEFAULT_PACK, data_dir=self.root, record_kind="test")
        self.view = self.service.create(copy.deepcopy(PROFILE))
        self.id, self.token = self.view["session"]["id"], self.view["session_token"]
        self.counter = 0

    def change(self, action, **payload):
        self.counter += 1
        self.view = self.service.mutate(self.id, self.token, action, {
            "request_id": f"v2-test-{self.counter:03}", "revision": self.view["session"]["revision"], **payload})
        return self.view

    def fill(self, stage, value="3"):
        for card in self.view["cards"]:
            self.change("save", card_id=card["id"], answers={q["id"]: value for q in self.view["questions"] if q["stage"] == stage})

    def complete(self):
        self.fill("A"); self.change("reveal"); self.fill("B"); self.change("submit")
        return self.service.export(self.id, self.token)

    def fake_export(self, code="SYNTH-A", value="3"):
        state = copy.deepcopy(self.view["session"])
        state.update({"id": "SYNTHETIC-" + code, "stage": "complete", "record_kind": "pilot"})
        state["profile"]["code"] = code
        for cid in state["order"]:
            state["answers"][cid] = {q["id"]: value for q in self.view["questions"]}
        return {"session": state, "questions": self.view["questions"], "scoring": self.view["scoring"]}

    def test_ten_real_distinct_sources_and_six_dimensions(self):
        p = self.service.pack
        self.assertEqual(len(p["cards"]), 10)
        self.assertEqual(len(p["questions"]), 6)
        self.assertEqual([q["stage"] for q in p["questions"]], ["A"] * 3 + ["B"] * 3)
        cs = [x["source_evidence"]["candidate"] for x in p["organizer"]["mapping"]]
        self.assertEqual(len({c["canonical_hypothesis_id"] for c in cs}), 10)
        self.assertEqual(len({c["endpoint"] for c in cs}), 10)
        self.assertEqual(len({c["pre_external_metric_group"]["group_anchor"] for c in cs}), 10)
        self.assertTrue(all(c["external_status"] == "EXTERNAL_COMPLETE" for c in cs))
        self.assertEqual(sum(len(c["post"]["external_rows"]) for c in p["cards"]), 82)

    def test_every_displayed_numeric_result_bound_to_source(self):
        for card, private in zip(self.service.pack["cards"], self.service.pack["organizer"]["mapping"]):
            source = private["source_evidence"]
            candidate = source["candidate"]
            self.assertEqual(card["pre"]["hypothesis"], candidate["canonical_record"]["hypothesis_zh"])
            self.assertEqual(card["post"]["tcp_joint"], candidate["canonical_record"]["joint_direction_stability"])
            self.assertEqual(card["post"]["ucla_joint"], candidate["original_external_assessment"]["joint_direction_frequency"])
            self.assertEqual(len(card["post"]["external_rows"]), len(candidate["external_component_rows"]))
            for displayed, raw in zip(card["post"]["external_rows"], candidate["external_component_rows"]):
                for key, value in displayed.items():
                    if key != "row_id": self.assertEqual(value, raw["record"][key])
            for internal in card["post"]["internal_rows"]:
                for key, value in candidate["canonical_record"]["components"][internal["domain"]].items():
                    self.assertEqual(internal[key], value)
            parent = source["feedback_lineage"][0]["parent_prior_same_seed_TCP_records"][0]
            self.assertEqual(card["post"]["feedback"]["parent_components"], parent["feedback"]["components"])
            self.assertEqual(card["post"]["feedback"]["parent_joint"], parent["feedback"]["joint_direction_stability"])
            self.assertTrue(source["first_proposal"]["selection_locked_before_TCP"])
            self.assertEqual(len(source["first_proposal"]["six_model_reviews"]), 6)
            self.assertTrue(source["feedback_lineage"][0]["same_feedback_verified"])
            self.assertFalse(re.search(r"(?<![A-Za-z0-9])E\d{3}", card["pre"]["rationale"] + card["post"]["feedback"]["action_text"]))

    def test_exact_rebuild_and_parent_preserved(self):
        self.assertEqual(hashlib.sha256(V1.read_bytes()).hexdigest(), V1_HASH)
        self.assertEqual(build(SOURCE, V1), self.service.pack)

    def test_selection_does_not_depend_on_external_success(self):
        p = self.service.pack
        candidates = json.loads((SOURCE / "candidate_index.json").read_text(encoding="utf-8"))["records"]
        proposals = {x["proposal_event_id"]: x for x in json.loads((SOURCE / "proposal_slots.json").read_text(encoding="utf-8"))["records"]}
        lineages = {x["proposal_event_id"]: x for x in json.loads((SOURCE / "feedback_lineage.json").read_text(encoding="utf-8"))["records"]}
        existing = [x["source_evidence"]["candidate"] for x in p["organizer"]["mapping"][:3]]
        before, _ = select_additions(candidates, proposals, lineages, existing)
        for c in candidates:
            c["original_external_assessment"] = {"fake_changed_outcome": 999}
            c["external_component_rows"] = []
        after, _ = select_additions(candidates, proposals, lineages, existing)
        self.assertEqual([c["canonical_hypothesis_id"] for c in before], [c["canonical_hypothesis_id"] for c in after])

    def test_profile_removed_fields_not_defaulted_or_accepted(self):
        self.assertEqual(set(self.view["session"]["profile"]), {"code", "experience", "consent"})
        self.assertEqual(self.view["session"]["profile_schema_version"], "anonymous-experience-v4")
        self.assertEqual(self.view["session"]["record_kind"], "test")
        for field in ["specialty", "old_study", "case_exposure", "mode"]:
            with self.assertRaises(StudyError): self.service.create({**PROFILE, field: "no"})
        no_consent = self.service.create({"code": "TEST-NOCONSENT", "experience": "0-2"})
        self.assertEqual(set(no_consent["session"]["profile"]), {"code", "experience"})

    def test_exact_scores_and_absence_of_default_answers(self):
        for q in self.view["questions"]:
            self.assertEqual([o["score"] for o in q["options"]], [1, 2, 3, 4, 5, None, None])
            self.assertNotIn("default", q)
        self.assertTrue(all(not a for a in self.view["session"]["answers"].values()))

    def test_all_six_mean_and_dimension_scores(self):
        exp = self.fake_export()
        for cid in exp["session"]["order"]:
            exp["session"]["answers"][cid] = dict(zip(exp["scoring"]["item_ids"], ["1", "2", "3", "4", "5", "3"]))
        scores = score_session(exp["session"], exp)
        self.assertEqual(scores["composite_mean"], 3)
        self.assertEqual(scores["complete_case_count"], 10)
        self.assertEqual(scores["dimensions"]["grounding"]["mean"], 1)
        self.assertEqual(scores["dimensions"]["value"]["mean"], 5)

    def test_unable_is_null_not_zero_or_midpoint(self):
        exp = self.fake_export(value="5")
        cid = exp["session"]["order"][0]
        exp["session"]["answers"][cid]["novelty"] = "outside"
        scores = score_session(exp["session"], exp)
        self.assertIsNone(scores["cards"][cid]["composite"])
        self.assertIsNone(scores["cards"][cid]["items"]["novelty"])
        self.assertEqual(scores["dimensions"]["novelty"]["mean"], 5)
        self.assertEqual(scores["dimensions"]["novelty"]["n"], 9)
        self.assertEqual(scores["dimensions"]["novelty"]["unavailable_counts"], {"outside": 1})
        self.assertEqual(scores["composite_mean"], 5)
        self.assertEqual(scores["complete_case_count"], 9)

    def test_all_unavailable_has_no_composite(self):
        exp = self.fake_export(value="insufficient")
        result = aggregate_exports([exp])
        self.assertIsNone(result["composite_mean"])
        self.assertEqual(result["composite_cases_covered"], 0)
        self.assertIsNone(result["dimensions"]["novelty"]["mean"])

    def test_aggregation_balances_cases_and_reports_coverage(self):
        a, b = self.fake_export("A", "1"), self.fake_export("B", "5")
        # Case one has one valid expert; other cases have two. Case balancing
        # gives (1 + 9*3)/10 = 2.8, not the pooled-rating mean 55/19.
        for q in b["questions"]:
            b["session"]["answers"][b["session"]["order"][0]][q["id"]] = "outside"
        result = aggregate_exports([a, b])
        self.assertAlmostEqual(result["composite_mean"], 2.8)
        self.assertAlmostEqual(result["dimensions"]["grounding"]["mean"], 2.8)
        self.assertEqual(result["expert_count"], 2)
        self.assertEqual(result["dimensions"]["grounding"]["n_ratings"], 19)

    def test_tests_duplicates_and_incomplete_excluded(self):
        a, b, c = self.fake_export("A"), self.fake_export("B"), self.fake_export("C")
        b["session"]["record_kind"] = "test"
        c["session"]["stage"] = "A"
        result = aggregate_exports([a, a, b, c])
        self.assertEqual(result["expert_count"], 1)
        self.assertEqual(result["excluded"], {"duplicate_export": 1, "test": 1, "incomplete_session": 1})

    def test_versions_and_duplicate_experts_not_pooled(self):
        a, b = self.fake_export("A"), self.fake_export("B")
        b["session"]["pack_hash"] = "different"
        with self.assertRaises(ValueError): aggregate_exports([a, b])
        b = self.fake_export("B")
        b["session"]["profile"]["code"] = "A"
        with self.assertRaises(ValueError): aggregate_exports([a, b])

    def test_complete_api_locks_exports_numeric_summary(self):
        exported = self.complete()
        self.assertEqual(exported["score_summary"]["composite_mean"], 3)
        self.assertEqual(exported["score_summary"]["complete_case_count"], 10)
        self.assertNotIn(self.token, json.dumps(exported))
        self.assertEqual(exported["export_schema"], 2)
        with self.assertRaises(StudyError): self.change("save", card_id=self.view["cards"][0]["id"])
        self.assertEqual(aggregate_exports([exported])["expert_count"], 0)

    def test_full_ten_case_phase_gate_and_no_early_summary(self):
        self.assertNotIn("score_summary", self.view)
        for card in self.view["cards"][:-1]:
            self.change("save", card_id=card["id"], answers={q["id"]: "3" for q in self.view["questions"] if q["stage"] == "A"})
        with self.assertRaises(StudyError): self.change("reveal")
        self.assertEqual(self.view["session"]["stage"], "A")
        self.assertNotIn("post", self.view["cards"][0])
        self.fill("A"); self.change("reveal")
        self.assertNotIn("score_summary", self.view)
        with self.assertRaises(StudyError): self.change("save", card_id=self.view["cards"][0]["id"], answers={"novelty": "5"})

    def test_no_curator_verdict_or_private_data_returned(self):
        self.fill("A"); self.change("reveal")
        serialized = json.dumps(self.view)
        for key in ["organizer", "source_run", "original_material_id", "curator_summary", "pragmatic_triage_pass", "source_evidence"]:
            self.assertNotIn('"' + key + '"', serialized)

    def test_v1_resume_unchanged_in_v2_server(self):
        legacy = DiscoveryStudy(V1, self.root)
        old = legacy.create({**PROFILE, "specialty": "神经影像", "old_study": "yes", "case_exposure": "unsure", "mode": "test"})
        resumed = self.service.get(old["session"]["id"], old["session_token"])
        self.assertEqual(resumed["session"], old["session"])
        self.assertEqual(len(resumed["cards"]), 3)
        self.assertEqual(len(resumed["questions"]), 12)
        self.assertIsNone(resumed["scoring"])

    def test_previous_v2_profile_preserved_on_resume_and_save(self):
        # Construct a historical profile only in this isolated synthetic database.
        previous = copy.deepcopy(self.view["session"])
        previous.pop("profile_schema_version")
        previous["profile"]["specialty"] = "其他交叉学科"
        with self.service.connection() as db:
            db.execute("UPDATE sessions SET state=? WHERE id=?", (json.dumps(previous), self.id))
        self.view = self.service.get(self.id, self.token)
        self.assertEqual(self.view["session"], previous)
        self.change("save", card_id=self.view["cards"][0]["id"], answers={"grounding": "3"})
        self.assertEqual(self.view["session"]["profile"], previous["profile"])
        self.assertNotIn("profile_schema_version", self.view["session"])

    def test_interface_removed_fields_and_static_protection(self):
        with TestClient(create_preview_app(DEFAULT_PACK, self.root / "api", "test")) as client:
            html = client.get("/discovery-study").text
            for text in ['name="specialty"', 'name="old_study"', 'name="case_exposure"', 'name="mode"', "主要专业领域", "本次用途", "无法判断不是低分"]:
                self.assertNotIn(text, html)
            config = client.get("/api/studies/discovery/config").json()
            self.assertNotIn("specialties", config)
            self.assertEqual(config["meta"]["card_count"], 10)
            self.assertEqual(len(config["questions"]), 6)
            self.assertEqual(client.get("/static/cs1_discovery_pilot_v2.json").status_code, 404)
            created = client.post("/api/studies/discovery/sessions", json=PROFILE)
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["session"]["record_kind"], "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
