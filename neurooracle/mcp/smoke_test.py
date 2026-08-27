"""Small in-process MCP handshake test; run with the academic MCP venv."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from mcp.client import Client

from . import academic_server
from .academic_service import AcademicService


async def _run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        formal_root = Path(temporary) / "neurooracle" / "data" / "full_v2"
        formal_root.mkdir(parents=True)
        (formal_root / "CURRENT_STATE.json").write_text(
            json.dumps(
                {
                    "taxonomy_version": "case_study_membership.v2",
                    "formal_kg_statistics": {
                        "general": {"papers": 0, "claims": 0},
                        "case_studies": {"brain_age": {"papers": 0, "claims": 0}},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        previous_service = academic_server.SERVICE
        academic_server.SERVICE = AcademicService(Path(temporary))
        try:
            async with Client(academic_server.mcp) as client:
                listed = await client.list_tools()
                names = {tool.name for tool in listed.tools}
                assert len(names) == 9
                assert "neuroclaw_academic_search_literature" in names
                assert "neuroclaw_academic_start_sparse_collection" in names
                assert "inject" not in " ".join(sorted(names))
                assert (client.instructions or "") == academic_server.SERVER_INSTRUCTIONS
                result = await client.call_tool(
                    "neuroclaw_academic_get_kg_coverage",
                    {"case_study_id": "brain_age"},
                )
                assert not result.is_error
                assert result.structured_content["case_studies"]["brain_age"]["papers"] >= 0
        finally:
            academic_server.SERVICE = previous_service
    print("academic MCP in-process handshake: PASS")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
