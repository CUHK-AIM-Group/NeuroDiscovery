# T131_literature_search_dmn_aging: Academic Search: Default Mode Network and Aging
## Task Description

Search for the most recent papers related to **"default mode network aging"**
from multiple academic platforms:

- **arXiv**
- **PubMed**
- **Semantic Scholar**

### Constraints

- **Time Range:** Last 180 days
- **Results per Platform:** 20 papers each (60 total minimum)
- **Sorting:** Newest first
- **Deduplication:** Cross-platform duplicates removed by DOI/title match
- **No Input:** This test case requires no command-line input
- **Robustness:** The workflow should tolerate partial platform failures,
  access restrictions, and rate limits without failing the whole run

### Expected Output

Results should be saved to `benchmark_results/T131_literature_search_dmn_aging/`
as a JSON file with the following structure:

```json
{
  "metadata": {
    "query": "default mode network aging",
    "timestamp": "ISO-8601 format",
    "total_papers": 60,
    "duplicates_removed": 0
  },
  "arxiv": [
    {
      "title": "string",
      "authors": ["string"],
      "published": "ISO-8601 format",
      "url": "string",
      "abstract": "string",
      "doi": "string or null"
    }
  ],
  "pubmed": [],
  "semantic_scholar": []
}
```

## Evaluation

- This test case is manually evaluated.
