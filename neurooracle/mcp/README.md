# NeuroClaw Academic MCP

Local STDIO MCP server for NeuroOracle literature discovery and collection-only
staging. It wraps the project’s canonical Case Study registry, global paper
identity index, Europe PMC retrieval, extraction batches, and fail-closed
collection seals.

## Frozen workflow

1. Search Europe PMC with the fixed 1980–2026 window using lite metadata.
2. Deduplicate against the formal KG and every active staging corpus.
3. Download the matching complete provider abstract only after deduplication.
4. Treat search targets as provenance, never as Case Study membership.
5. During later claim extraction, assign every applicable Case Study; labels
   are non-exclusive and paper membership is the union of claim memberships.
6. Require a valid seal before extraction. No formal-KG injection tool exists.

The only write-capable tool starts a background collector under
`neurooracle/data/case_study_staging/academic_mcp/`. The collector hashes the
three formal KG files before and after collection and refuses to seal if they
change.

## Tools

- `neuroclaw_academic_list_case_studies`
- `neuroclaw_academic_get_kg_coverage`
- `neuroclaw_academic_check_paper_identity`
- `neuroclaw_academic_search_literature`
- `neuroclaw_academic_list_collections`
- `neuroclaw_academic_get_collection_status`
- `neuroclaw_academic_get_batch`
- `neuroclaw_academic_validate_collection`
- `neuroclaw_academic_start_sparse_collection`

## Reproduce the local environment

From the repository root on Windows:

```powershell
python -m venv .venv-academic-mcp
.\.venv-academic-mcp\Scripts\python.exe -m pip install -r neurooracle\mcp\requirements-lock.txt
.\.venv-academic-mcp\Scripts\python.exe -m neurooracle.mcp.smoke_test
```

A trusted-project `.codex/config.toml` can launch the server with:

```powershell
.\.venv-academic-mcp\Scripts\python.exe -m neurooracle.mcp.academic_server
```

Restart Codex after changing MCP configuration. Use `/mcp` or `codex mcp list`
to inspect the connection.

## Verification

```powershell
.\.venv-academic-mcp\Scripts\python.exe -m unittest neurooracle.src.tests.test_academic_mcp_service -v
.\.venv-academic-mcp\Scripts\python.exe -m neurooracle.mcp.smoke_test
```

`evaluation.xml` contains ten independent, read-only evaluation questions over
the closed 100,000-paper collection.
