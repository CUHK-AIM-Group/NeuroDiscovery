import tempfile
import unittest

from fastapi.testclient import TestClient

from core.web.evaluation_app import ASSETS, BANK, STUDY_ID, create_app


class EvaluationAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.client = TestClient(create_app(self.directory.name), base_url="http://127.0.0.1")
        self.addCleanup(self.client.close)

    def ranking_payload(self):
        return {"study_id": STUDY_ID, "participant_id": "TEST-EVALUATION", "assignment_id": "P01",
                "candidate_path": str(BANK), "condition": "manual", "experience": "3-5"}

    def test_allowed_pages_assets_and_isolated_config(self):
        for route in ["/", "/study", "/discovery-study"] + [f"/static/{name}" for name in ASSETS]:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertEqual(response.headers["x-frame-options"], "SAMEORIGIN")
        self.assertEqual(self.client.get("/api/health").json()["mode"], "human-evaluation-only")
        config = self.client.get("/api/studies/config").json()
        self.assertTrue(config["evaluation_only"])
        self.assertEqual(config["graph_path"], "")

    def test_research_and_organizer_routes_are_absent(self):
        for route in ["/api/chat", "/api/graph/status", "/api/models", "/api/studies/results",
                      "/static/index.html", "/static/../evaluation_app.py", "/openapi.json"]:
            self.assertEqual(self.client.get(route).status_code, 404, route)
        self.assertEqual(self.client.post("/api/studies/execution-results/import", json={}).status_code, 404)

    def test_local_browser_boundary(self):
        for headers in [{"Origin": "https://attacker.example"}, {"Host": "attacker.example"},
                        {"Sec-Fetch-Site": "cross-site"}]:
            self.assertEqual(self.client.get("/api/health", headers=headers).status_code, 403)
        self.assertEqual(self.client.post("/api/studies/auth", content="{}").status_code, 415)
        self.assertEqual(self.client.post("/api/studies/auth", json={}).status_code, 200)

    def test_fixed_materials_and_assignments(self):
        for patch in [{"candidate_path": "C:/private.json"}, {"graph_path": "C:/private.json"},
                      {"assignment_id": "ALL"}, {"assignment_id": []}, {"condition": []}, {"study_id": "other"}]:
            self.assertEqual(self.client.post("/api/studies/sessions", json=self.ranking_payload() | patch).status_code, 400)

    def test_both_evaluations_persist_and_export_after_restart(self):
        response = self.client.post("/api/studies/discovery/sessions", json={
            "code": "TEST-EVALUATION", "experience": "3-5", "assignment_id": "P01"})
        self.assertEqual(response.status_code, 200, response.text)
        discovery = response.json()
        self.assertTrue(all(len(card["pre"]["references"]) >= 5 for card in discovery["cards"]))
        discovery_endpoint = f"/api/studies/discovery/sessions/{discovery['session']['id']}"
        discovery_headers = {"X-Discovery-Session-Token": discovery["session_token"]}
        for index, card in enumerate(discovery["cards"]):
            response = self.client.post(discovery_endpoint + "/save", headers=discovery_headers, json={
                "request_id": f"evaluation-save-{index}", "revision": discovery["session"]["revision"],
                "card_id": card["id"], "answers": {question["id"]: "4" for question in discovery["questions"]}})
            self.assertEqual(response.status_code, 200, response.text)
            discovery.update(response.json())
        response = self.client.post(discovery_endpoint + "/submit", headers=discovery_headers, json={
            "request_id": "evaluation-submit", "revision": discovery["session"]["revision"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["score_summary"]["composite_mean"], 4)
        response = self.client.post("/api/studies/sessions", json=self.ranking_payload())
        self.assertEqual(response.status_code, 200, response.text)
        ranking = response.json()
        session_id = ranking["session_id"]
        pair = ranking["pairs"][0]
        event = {"type": "pairwise_choice", "elapsed_ms": 1500, "payload": {
            "pair_id": pair["pair_id"], "left_id": pair["left_id"], "right_id": pair["right_id"],
            "choice": "left", "winner_id": pair["left_id"], "decision_ms": 1500}}
        self.assertEqual(self.client.post(f"/api/studies/sessions/{session_id}/events", json={"events": [event]}).json()["accepted"], 1)
        with TestClient(create_app(self.directory.name), base_url="http://127.0.0.1") as restarted:
            restored = restarted.get(f"/api/studies/sessions/{session_id}?include_events=true").json()
            self.assertTrue(restored["events"])
            discovery_id = discovery["session"]["id"]
            response = restarted.get(f"/api/studies/discovery/sessions/{discovery_id}",
                                     headers={"X-Discovery-Session-Token": discovery["session_token"]})
            self.assertEqual(response.status_code, 200)
            response = restarted.post(f"/api/studies/sessions/{session_id}/submit", json={
                "ranking": [candidate["id"] for candidate in ranking["candidates"]],
                "active_seconds": 600, "wall_seconds": 610})
            self.assertEqual(response.status_code, 200, response.text)
            response = restarted.post("/api/studies/evaluations/export", json={
                "participant_code": "TEST-EVALUATION", "discovery_sessions": [
                    {"id": discovery_id, "token": discovery["session_token"]}],
                "ranking_session_ids": [session_id]})
            self.assertEqual(response.status_code, 200, response.text)
            bundle = response.json()
            self.assertEqual(bundle["human_evaluation_2"]["completed_sessions"], 1)
            self.assertEqual(bundle["human_evaluation_2"]["sessions"][0]["questions"][0]["answer"], "left")
            self.assertEqual(bundle["completion_status"], "partial")


if __name__ == "__main__":
    unittest.main()
