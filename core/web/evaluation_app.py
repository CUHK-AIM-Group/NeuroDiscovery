"""Offline human-evaluation server with no research or model runtime imports."""
import argparse
import getpass
import os
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse

from core.web.discovery_study import DEFAULT_PACK, DiscoveryStudy, register_discovery_routes
from neurooracle.src.user_study import PROTOCOL_VERSION, UserStudyService, load_pair_assignments

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("static")
BANK = ROOT / "neurooracle/data/user_study/case1_tcp_expert_study_v3.json"
STUDY_ID = "case1-tcp-external-validation-v1"
ASSETS = frozenset({
    "evaluation-home.html", "evaluation-home.js", "evaluation-home.css",
    "study.html", "discovery-study.html", "discovery-study.css", "discovery-study.js",
    "discovery-study-i18n.js", "study-workspace.css", "study-workspace.js",
    "workspace-tokens.css", "evaluation-export.js",
})


def create_app(data_root, pack_path=DEFAULT_PACK, bank_path=BANK):
    data_root, bank_path = Path(data_root).resolve(), Path(bank_path).resolve()
    discovery = DiscoveryStudy(pack_path, data_root / "discovery")
    ranking = UserStudyService(data_root / "ranking")
    _, manifest_hash, source, _ = ranking.load_candidates(bank_path)
    assignments = load_pair_assignments(source, manifest_hash)
    if not assignments:
        raise ValueError("The evaluation package requires the frozen HE2 assignments")
    app = FastAPI(title="Human Evaluation", docs_url=None, redoc_url=None, openapi_url=None)
    register_discovery_routes(app, discovery, ranking)

    @app.middleware("http")
    async def local_requests_only(request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"error": "Local access only"}, status_code=403)
        if origin and origin != f"http://{request.headers.get('host')}":
            return JSONResponse({"error": "Cross-origin requests are not allowed"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"error": "Cross-site requests are not allowed"}, status_code=403)
        if request.method == "POST" and request.headers.get("content-type", "").split(";")[0] != "application/json":
            return JSONResponse({"error": "JSON requests required"}, status_code=415)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'self'; form-action 'self'"
        return response

    def call(method, *args, **kwargs):
        try:
            return getattr(ranking, method)(*args, **kwargs)
        except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    @app.get("/")
    async def home():
        return FileResponse(STATIC / "evaluation-home.html")

    @app.get("/study")
    async def study():
        return FileResponse(STATIC / "study.html")

    @app.get("/static/{name}")
    async def asset(name: str):
        if name not in ASSETS:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(STATIC / name)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "mode": "human-evaluation-only", "pack_id": discovery.pack["pack_id"],
                "instance_id": os.environ.get("NEURODISCOVERY_EVALUATION_INSTANCE", "")}

    @app.post("/api/studies/auth")
    async def authenticate():
        return {"token": "", "lifetime": "local_only"}

    @app.get("/api/studies/config")
    async def config():
        return {
            "evaluation_only": True,
            "protocol_version": PROTOCOL_VERSION, "study_id": STUDY_ID,
            "case_study": "case1_tcp_external_validation", "conditions": ["manual", "assisted"],
            "case_name": {"zh": "Human Evaluation 2 · 假设排序", "en": "Human Evaluation 2 · Hypothesis ranking"},
            "suggested_candidate_sources": [str(bank_path)], "graph_path": "",
            "default_participant_id": getpass.getuser(), "study_root": str(ranking.root),
            "session_protocol": ranking.load_study_protocol(bank_path),
            "pair_assignments": {"version": assignments["version"], "rule_zh": assignments["rule_zh"],
                                 "options": [{"id": expert, "pair_count": len(assigned)}
                                             for expert, assigned in assignments["experts"].items()]},
        }

    @app.post("/api/studies/sessions")
    async def create_session(payload: dict = Body(...)):
        if payload.get("graph_path") or payload.get("candidate_path") not in (None, "", str(bank_path)):
            return JSONResponse({"error": "Only packaged evaluation materials are allowed"}, status_code=400)
        if (payload.get("study_id") != STUDY_ID or not isinstance(payload.get("assignment_id"), str)
                or payload["assignment_id"] == "ALL" or payload["assignment_id"] not in assignments["experts"]):
            return JSONResponse({"error": "Choose a packaged participant assignment"}, status_code=400)
        if payload.get("condition", "manual") not in ("manual", "assisted"):
            return JSONResponse({"error": "Invalid evaluation condition"}, status_code=400)
        return call("create_session", study_id=STUDY_ID, participant_id=str(payload.get("participant_id") or ""),
                    condition=payload.get("condition", "manual"), candidate_path=bank_path,
                    case_study="case1_tcp_external_validation", experience=str(payload.get("experience") or ""),
                    assignment_id=payload["assignment_id"], random_seed=0)

    @app.get("/api/studies/sessions")
    async def sessions(study_id: str):
        return {"study_id": study_id, "sessions": ranking.list_sessions(study_id)}

    @app.get("/api/studies/sessions/{session_id}")
    async def get_session(session_id: str, include_events: bool = False):
        return call("get_session", session_id, include_events=include_events)

    @app.delete("/api/studies/sessions/{session_id}")
    async def delete_session(session_id: str):
        return call("delete_active_session", session_id)

    @app.post("/api/studies/sessions/{session_id}/events")
    async def events(session_id: str, payload: dict = Body(...)):
        if not isinstance(payload.get("events", []), list):
            return JSONResponse({"error": "Events must be a list"}, status_code=400)
        result = call("append_events", session_id, payload.get("events", []))
        return result if isinstance(result, JSONResponse) else {"accepted": result}

    @app.post("/api/studies/sessions/{session_id}/submit")
    async def submit(session_id: str, payload: dict = Body(...)):
        return call("submit_session", session_id, ranking=payload.get("ranking", []),
                    active_seconds=payload.get("active_seconds", 0), wall_seconds=payload.get("wall_seconds", 0),
                    buckets=payload.get("buckets", {}), completion_reason=payload.get("completion_reason", "submitted"))

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
