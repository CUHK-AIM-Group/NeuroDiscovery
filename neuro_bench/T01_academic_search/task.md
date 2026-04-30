# Benchmark Test Case 1: Academic Search

## Task Description

Search for the most recent papers related to **"world models"** from multiple academic platforms:
- **arXiv**
- **OpenReview**
- **PubMed**

### Constraints
- **Time Range:** Last 180 days
- **Results per Platform:** 30 papers each (90 total minimum)
- **Sorting:** Newest first
- **No Input:** This test case requires no command-line input
- **Robustness:** The workflow should tolerate partial platform failures, access restrictions, and rate limits without failing the whole run

### Expected Output
Results should be saved to `benchmark_results/T01_academic_search/` as a JSON file with the following structure:

```json
{
  "metadata": {
    "query": "world models",
    "timestamp": "ISO-8601 format",
    "total_papers": 90
  },
  "arxiv": [
    {
      "title": "string",
      "authors": ["string"],
      "published": "ISO-8601 format (YYYY-MM-DD or full timestamp)",
      "url": "string",
      "abstract": "string"
    }
  ],
  "pubmed": [...],
  "semantic_scholar": [...],
  "openreview": [...]
}
```

### Success Criteria
- Successfully retrieves 30 papers (or maximum available) from each platform
- Papers are sorted by publication date (newest first)
- Results are saved in JSON format with proper keys
- Output file is timestamped and placed in `benchmark_results/T01_academic_search/`
- **All papers must have publication dates within 180 days of the search date**
- No errors or missing dependencies
- The workflow should preserve partial results when one source is unavailable, and record that degraded condition in output metadata or logs rather than silently failing
- Retrieved records should be normalized and deduplicated consistently across sources before final export

### Implementation Details

The test uses the **academic-research-hub** skill to:
1. Query arXiv API for "world models" papers in the last 180 days
2. Query PubMed database for "world models" papers in the last 180 days
3. Query Semantic Scholar for "world models" papers (date filtering applied)
4. Attempt to collect papers from OpenReview (requires API access)
5. Merge and deduplicate results
6. Save results with full metadata (titles, authors, abstracts, URLs, publication dates)
7. Reuse cached responses or skill-supported local outputs when possible to reduce duplicate requests and rate-limiting risk
8. Apply a thin task-specific normalization layer when source schemas differ from the benchmark export schema

### Notes
- OpenReview may return 0 results due to access restrictions from certain hosts/regions
- PubMed requires a valid email address (set in environment or code)
- All searches are cached to avoid rate limiting issues
- Higher-quality solutions should treat the academic-research-hub capability as the preferred retrieval backbone, then adapt the final export to the benchmark schema rather than reimplementing every source from scratch unless a source is unsupported
- OpenReview failure should be handled as an expected degraded branch with explicit logging/metadata, not as a reason to discard the other platforms
