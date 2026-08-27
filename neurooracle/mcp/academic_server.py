"""STDIO MCP server for NeuroClaw academic literature workflows."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from .academic_service import SERVICE


SERVER_INSTRUCTIONS = (
    "Use this server for NeuroClaw academic literature discovery and staging. "
    "All searches are fixed to 1980-2026. Search provenance is not Case Study "
    "membership; labels are non-exclusive and determined only during claim "
    "extraction using the frozen policy. Always deduplicate against the formal "
    "KG and active staging before extraction. Use complete abstracts. The formal "
    "KG is read-only; no injection tool is exposed. A valid collection seal is "
    "required before extraction."
)

mcp = MCPServer(
    name="neuroclaw-academic",
    title="NeuroClaw Academic Literature",
    description=(
        "Search, deduplicate, retrieve complete abstracts, inspect staging batches, "
        "and validate extraction seals without modifying the formal NeuroOracle KG."
    ),
    instructions=SERVER_INSTRUCTIONS,
    version="0.1.0",
    log_level="WARNING",
)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
READ_ONLY_NETWORK = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)
STAGING_WRITE_NETWORK = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


@mcp.tool(
    name="neuroclaw_academic_list_case_studies",
    title="List NeuroClaw Case Studies",
    description=(
        "List the 17 peer Case Study IDs, display numbers, names, current formal-KG "
        "coverage, and frozen non-exclusive membership policy."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def list_case_studies() -> dict[str, Any]:
    """Return the canonical Case Study catalog and current coverage."""

    return await asyncio.to_thread(SERVICE.list_case_studies)


@mcp.tool(
    name="neuroclaw_academic_get_kg_coverage",
    title="Get Formal KG Coverage",
    description=(
        "Read canonical formal-KG paper and claim counts for general and either all "
        "Case Studies or one specified Case Study."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def get_kg_coverage(case_study_id: str = "") -> dict[str, Any]:
    """Use an empty case_study_id to return all Case Studies."""

    return await asyncio.to_thread(SERVICE.get_kg_coverage, case_study_id)


@mcp.tool(
    name="neuroclaw_academic_check_paper_identity",
    title="Check Paper Identity",
    description=(
        "Check whether a paper is already in the formal KG or any active staging "
        "corpus using PMID, DOI, PMCID, arXiv, OpenAlex, or title plus year."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def check_paper_identity(
    paper_id: str = "",
    pmid: str = "",
    doi: str = "",
    pmcid: str = "",
    arxiv_id: str = "",
    openalex_id: str = "",
    title: str = "",
    year: int | None = None,
) -> dict[str, Any]:
    """Provide at least one stable identifier, or title and publication year."""

    return await asyncio.to_thread(
        SERVICE.check_paper_identity,
        paper_id=paper_id,
        pmid=pmid,
        doi=doi,
        pmcid=pmcid,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        title=title,
        year=year,
    )


@mcp.tool(
    name="neuroclaw_academic_search_literature",
    title="Search Deduplicated Literature",
    description=(
        "Search Europe PMC in the fixed 1980-2026 window, deduplicate lite metadata "
        "against the formal KG and active staging, then retrieve complete abstracts "
        "only for unseen papers. Case Study IDs are search provenance only."
    ),
    annotations=READ_ONLY_NETWORK,
    structured_output=True,
)
async def search_literature(
    query: str,
    case_study_ids: list[str] | None = None,
    cursor: str = "*",
    page_size: int = 25,
    max_results: int = 10,
) -> dict[str, Any]:
    """Use Europe PMC syntax; pagination uses the returned next_cursor."""

    return await asyncio.to_thread(
        SERVICE.search_literature,
        query=query,
        case_study_ids=case_study_ids or (),
        cursor=cursor,
        page_size=page_size,
        max_results=max_results,
    )


@mcp.tool(
    name="neuroclaw_academic_list_collections",
    title="List Literature Collections",
    description="List collection-only staging campaigns with bounded pagination.",
    annotations=READ_ONLY,
    structured_output=True,
)
async def list_collections(offset: int = 0, limit: int = 25) -> dict[str, Any]:
    """Return compact staging collection summaries."""

    return await asyncio.to_thread(
        SERVICE.list_collections, offset=offset, limit=limit
    )


@mcp.tool(
    name="neuroclaw_academic_get_collection_status",
    title="Get Collection Status",
    description=(
        "Inspect a collection's progress, target counts, files, background job, and "
        "completion seal without changing staging or the formal KG."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def get_collection_status(collection_id: str) -> dict[str, Any]:
    """Use a collection_id returned by list_collections."""

    return await asyncio.to_thread(SERVICE.get_collection_status, collection_id)


@mcp.tool(
    name="neuroclaw_academic_get_batch",
    title="Read an Extraction Batch",
    description=(
        "Read a bounded slice of a sealed or partial 100-paper extraction batch. "
        "This does not perform claim extraction or KG injection."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def get_batch(
    collection_id: str,
    batch_number: int,
    offset: int = 0,
    limit: int = 10,
    include_abstract: bool = True,
) -> dict[str, Any]:
    """Batch numbers are one-based; at most 25 papers are returned per call."""

    return await asyncio.to_thread(
        SERVICE.get_batch,
        collection_id=collection_id,
        batch_number=batch_number,
        offset=offset,
        limit=limit,
        include_abstract=include_abstract,
    )


@mcp.tool(
    name="neuroclaw_academic_validate_collection",
    title="Validate Collection Seal",
    description=(
        "Fail-closed validation of the completion seal, ready-queue hash, batch "
        "count, fixed policy, and formal-KG baseline. Full mode also validates every "
        "paper, abstract, identity, and batch row."
    ),
    annotations=READ_ONLY,
    structured_output=True,
)
async def validate_collection(
    collection_id: str, mode: str = "seal"
) -> dict[str, Any]:
    """Use mode='seal' for fast validation or mode='full' for all rows."""

    return await asyncio.to_thread(
        SERVICE.validate_collection, collection_id=collection_id, mode=mode
    )


@mcp.tool(
    name="neuroclaw_academic_start_sparse_collection",
    title="Start Sparse Case Study Collection",
    description=(
        "Start or resume the frozen sparse-Case-Study literature collector in a "
        "controlled staging directory. It searches lite metadata, deduplicates, then "
        "downloads complete abstracts. It cannot inject or modify the formal KG."
    ),
    annotations=STAGING_WRITE_NETWORK,
    structured_output=True,
)
async def start_sparse_collection(
    campaign_id: str,
    target_unique_papers: int,
    page_size: int = 1000,
) -> dict[str, Any]:
    """Creates only neurooracle/data/case_study_staging/academic_mcp/<campaign>."""

    return await asyncio.to_thread(
        SERVICE.start_sparse_collection,
        campaign_id=campaign_id,
        target_unique_papers=target_unique_papers,
        page_size=page_size,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


__all__ = ["SERVER_INSTRUCTIONS", "mcp", "main"]
