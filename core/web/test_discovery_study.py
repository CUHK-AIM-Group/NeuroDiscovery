"""Synthetic tests only. No production study data or experimental API access."""
import copy
import json
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient
from fastapi import FastAPI
from core.web.discovery_study import DiscoveryStudy, StudyError, create_preview_app, register_discovery_routes

DEFAULT_PACK = Path(__file__).resolve().parent / "study_materials" / "cs1_discovery_pilot_v1.json"

PROFILE = {"code": "TEST-QA", "specialty": "神经影像", "experience": "3-5", "old_study": "no", "case_exposure": "no", "mode": "test", "consent": True}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = DiscoveryStudy(DEFAULT_PACK, data_dir=self.root)
        self.created = self.service.create(copy.deepcopy(PROFILE))
        self.id, self.token = self.created["session"]["id"], self.created["session_token"]
        self.counter = 0

    def get(self):
        return self.service.get(self.id, self.token)

    def change(self, action, **values):
        self.counter += 1
        return self.service.mutate(self.id, self.token, action, {"request_id": f"test-request-{self.counter}", "revision": self.get()["session"]["revision"], **values})

    def fill(self, stage, value="insufficient"):
        view = self.get()
        for card in view["cards"]:
            answers = {q["id"]: value for q in view["questions"] if q["stage"] == stage}
            self.change("save", card_id=card["id"], answers=answers, note="synthetic test", issues=[])

    def reveal(self):
        self.fill("A")
        return self.change("reveal")

    def test_source_values_and_all_rows(self):
        pack = self.service.pack
        checked = 0
        for card, original in zip(pack["cards"], pack["organizer"]["mapping"]):
            candidate = original["source_evidence"]["candidate"]
            self.assertEqual(card["post"]["tcp_joint"], candidate["canonical_record"]["joint_direction_stability"])
            self.assertEqual(len(card["post"]["external_rows"]), 8)
            for shown, raw in zip(card["post"]["external_rows"], candidate["external_component_rows"]):
                for key, value in shown.items():
                    if key != "row_id":
                        self.assertEqual(value, raw["record"][key]); checked += 1
            for internal in card["post"]["internal_rows"]:
                for key in ["beta", "standardized_beta", "residual_RMS"]:
                    self.assertEqual(internal[key], candidate["canonical_record"]["components"][internal["domain"]][key]); checked += 1
            self.assertFalse(re.search(r"E\d{3}", card["pre"]["rationale"]))
            self.assertFalse(card["post"]["system_narrative_available"])
        self.assertEqual(checked, 306)

    def test_no_hidden_results_in_stage_a_or_export(self):
        for view in [self.service.config(), self.get(), self.service.export(self.id, self.token)]:
            serialized = json.dumps(view)
            for private in ["organizer", "original_material_id", "hypothesis_id", "source_run", "source_dir", "external_rows", "ucla_joint", "curator_summary", "secret_hash", "session_token"]:
                self.assertNotIn(f'"{private}"', serialized)
        for card in self.get()["cards"]:
            self.assertNotIn("post", card)

    def test_pre_result_lock_requires_all_cards(self):
        with self.assertRaises(StudyError): self.change("reveal")
        card = self.get()["cards"][0]
        self.change("save", card_id=card["id"], answers={q["id"]: "outside" for q in self.service.pack["questions"] if q["stage"] == "A"})
        with self.assertRaises(StudyError): self.change("reveal")
        self.assertEqual(self.get()["session"]["stage"], "A")

    def test_stage_a_frozen_and_post_requires_reveal(self):
        card = self.get()["cards"][0]["id"]
        with self.assertRaises(StudyError): self.change("save", card_id=card, answers={"evidence": "3"})
        revealed = self.reveal()
        self.assertIn("post", revealed["cards"][0])
        with self.assertRaises(StudyError): self.change("save", card_id=card, answers={"novelty": "5"})
        self.assertEqual(self.get()["session"]["answers"][card]["novelty"], "insufficient")

    def test_unknown_options_and_notes(self):
        card = self.get()["cards"][0]["id"]
        for values in [{"answers":{"novelty":"6"}}, {"answers":{"secret":"1"}}, {"note":"x"*4001}, {"issues":["not-an-issue"]}, {"active_seconds":28801}]:
            with self.assertRaises(StudyError): self.change("save", card_id=card, **values)
        self.assertEqual(self.get()["session"]["revision"], 0)

    def test_token_required_no_cross_session_access(self):
        other = self.service.create({**PROFILE, "code":"TEST-OTHER"})
        for token in ["", "wrong", other["session_token"]]:
            with self.assertRaises(StudyError): self.service.get(self.id, token)

    def test_stale_revision_rejected(self):
        card = self.get()["cards"][0]["id"]
        self.change("save", card_id=card, answers={"novelty":"3"})
        with self.assertRaises(StudyError):
            self.service.mutate(self.id, self.token, "save", {"request_id":"stale-request", "revision":0, "card_id":card, "answers":{"novelty":"5"}})
        self.assertEqual(self.get()["session"]["answers"][card]["novelty"], "3")

    def test_idempotent_retry_and_collision(self):
        card = self.get()["cards"][0]["id"]
        payload = {"request_id":"same-request", "revision":0, "card_id":card, "answers":{"novelty":"3"}}
        first = self.service.mutate(self.id, self.token, "save", payload)
        second = self.service.mutate(self.id, self.token, "save", payload)
        self.assertEqual(first["session"]["revision"],second["session"]["revision"])
        with self.assertRaises(StudyError): self.service.mutate(self.id, self.token, "save", {**payload, "answers":{"novelty":"5"}})

    def test_resume_and_complete_immutable(self):
        self.reveal(); self.fill("B"); completed = self.change("submit")
        self.assertEqual(completed["session"]["stage"], "complete")
        restarted = DiscoveryStudy(DEFAULT_PACK, data_dir=self.root)
        self.assertEqual(restarted.get(self.id, self.token)["session"], completed["session"])
        with self.assertRaises(StudyError): self.change("save", card_id=self.get()["cards"][0]["id"])
        export = self.service.export(self.id, self.token)
        self.assertNotIn(self.token, json.dumps(export))
        self.assertEqual(export["session"]["profile"]["mode"], "test")

    def test_discard_removes_only_unfinished_sessions_with_token(self):
        with self.assertRaises(StudyError): self.service.discard(self.id, "wrong-token")
        self.assertEqual(self.get()["session"]["id"], self.id)
        result = self.service.discard(self.id, self.token)
        self.assertTrue(result["deleted"])
        with self.assertRaises(StudyError): self.get()

        again = self.service.create(copy.deepcopy(PROFILE))
        self.id, self.token = again["session"]["id"], again["session_token"]
        self.reveal(); self.fill("B"); self.change("submit")
        with self.assertRaises(StudyError) as ctx: self.service.discard(self.id, self.token)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(self.get()["session"]["stage"], "complete")

    def test_discard_route_uses_session_token(self):
        app = create_preview_app(DEFAULT_PACK, self.root / "api-discard")
        with TestClient(app) as client:
            created = client.post("/api/studies/discovery/sessions", json=PROFILE).json()
            url = "/api/studies/discovery/sessions/" + created["session"]["id"]
            self.assertEqual(client.delete(url).status_code, 403)
            headers = {"X-Discovery-Session-Token": created["session_token"]}
            self.assertEqual(client.delete(url, headers=headers).json()["deleted"], True)
            self.assertEqual(client.get(url, headers=headers).status_code, 403)

    def test_failed_execution_not_negative_science(self):
        self.reveal(); self.fill("B")
        card = self.get()["cards"][0]["id"]
        self.change("save", card_id=card, answers={"validity":"1", "outcome":"1"})
        with self.assertRaises(StudyError): self.change("submit")
        self.change("save", card_id=card, answers={"outcome":"not_assessable", "evidence":"not_assessable", "coverage":"not_assessable"})
        self.assertEqual(self.change("submit")["session"]["stage"], "complete")

    def test_version_snapshot_and_identity_guard(self):
        original = self.get()
        changed = copy.deepcopy(self.service.pack)
        changed["public_meta"]["title"] = "new draft"
        path = self.root / "new_pack.json"
        path.write_text(json.dumps(changed),encoding="utf-8")
        with self.assertRaises(ValueError): DiscoveryStudy(path,self.root)
        changed["pack_id"] = "future-pilot-v2"
        path.write_text(json.dumps(changed),encoding="utf-8")
        future = DiscoveryStudy(path,self.root)
        self.assertEqual(future.get(self.id,self.token)["meta"],original["meta"])
        new = future.create({**PROFILE, "code":"TEST-NEW"})
        self.assertNotEqual(new["session"]["pack_hash"],original["session"]["pack_hash"])

    def test_required_profile_and_no_defaults(self):
        for invalid in [{}, {**PROFILE,"consent":False}, {**PROFILE,"code":"x"}, {**PROFILE,"mode":"formal"}]:
            with self.assertRaises(StudyError): self.service.create(invalid)
        self.assertTrue(all(not x for x in self.get()["session"]["answers"].values()))
        self.assertTrue(all("default" not in q for q in self.service.pack["questions"]))

    def test_durable_export_not_overwriting_or_leaking(self):
        first = self.service.save_export(self.id, self.token)
        second = self.service.save_export(self.id, self.token)
        self.assertNotEqual(first["path"], second["path"])
        exported = Path(first["path"]).read_text(encoding="utf-8")
        self.assertNotIn(self.token, exported)
        self.assertNotIn('"post"', exported)
        self.assertEqual(json.loads(exported)["session"]["stage"], "A")

    def test_api_security_and_phase_gate(self):
        app = create_preview_app(DEFAULT_PACK,self.root / "api")
        with TestClient(app) as client:
            self.assertEqual(client.get("/discovery-study").status_code,200)
            self.assertEqual(client.get("/static/discovery-study.js").status_code,200)
            for url in ["/static/cs1_discovery_pilot_v1.json", "/study_materials/cs1_discovery_pilot_v1.json", "/static/server.py"]:
                self.assertEqual(client.get(url).status_code,404)
            created = client.post("/api/studies/discovery/sessions",json=PROFILE).json()
            url = "/api/studies/discovery/sessions/"+created["session"]["id"]
            self.assertEqual(client.get(url).status_code,403)
            headers = {"X-Discovery-Session-Token":created["session_token"]}
            self.assertNotIn("post",client.get(url,headers=headers).json()["cards"][0])
            self.assertEqual(client.post(url+"/reveal",headers=headers,json={"revision":0,"request_id":"api-request-1"}).status_code,400)
            export=client.get(url+"/export",headers=headers)
            self.assertEqual(export.status_code,200)
            self.assertIn("attachment",export.headers["content-disposition"])
            self.assertEqual(export.headers["cache-control"],"no-store")

    def test_missing_optional_pack_does_not_break_main_app(self):
        with patch("core.web.discovery_study.DiscoveryStudy", side_effect=FileNotFoundError):
            app = FastAPI()
            register_discovery_routes(app)
            with TestClient(app) as client:
                self.assertEqual(client.get("/discovery-study").status_code, 200)
                self.assertEqual(client.get("/api/studies/discovery/config").status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
