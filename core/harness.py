"""NeuroDiscovery capabilities shared by terminal and graphical clients."""

from pathlib import Path

from core.autoresearch import AUTORESEARCH_MODE_OFF, SUPPORTED_AUTORESEARCH_MODES


def manifest(*, evaluation_enabled: bool = True) -> dict:
    capabilities = [
        {"id": "chat", "label": "Research chat", "view": "chat", "kind": "view"},
        {"id": "graph", "label": "Knowledge graph", "view": "neurooracle", "kind": "view"},
        {"id": "autoresearch", "label": "AutoResearch", "kind": "configure"},
        {"id": "evaluation", "label": "Human Evaluation", "view": "expert-study", "kind": "view"},
        {"id": "ranking", "label": "Hypothesis ranking", "view": "hypothesis-ranking", "kind": "view"},
        {"id": "results", "label": "Evaluation results", "view": "study-results", "kind": "view"},
    ]
    if not evaluation_enabled:
        capabilities = [item for item in capabilities if item["id"] not in {"evaluation", "ranking", "results"}]
    return {
        "schema_version": 1,
        "name": "NeuroDiscovery",
        "engine": "NeuroRuntime",
        "runtime_contract": "autoresearch-v1",
        "runtime_features": {"server_queue": True, "tool_approval": True, "safe_boundary_recovery": True,
                             "semantic_compaction_opt_in": True, "bounded_review_repair": True},
        "capabilities": capabilities,
        "autoresearch_modes": [AUTORESEARCH_MODE_OFF, *SUPPORTED_AUTORESEARCH_MODES],
        "execution_policy": {
            "navigation_starts_research": False,
            "mode_selection_starts_research": False,
            "scientific_acceptance": "existing source, evidence and human-review gates",
        },
    }


def register_routes(app, static_dir: Path, *, evaluation_enabled: bool = True) -> None:
    from fastapi.responses import FileResponse

    @app.get("/api/harness")
    async def capabilities():
        return manifest(evaluation_enabled=evaluation_enabled)

    @app.get("/harness")
    async def workspace():
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})
