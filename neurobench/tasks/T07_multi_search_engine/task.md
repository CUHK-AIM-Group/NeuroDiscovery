# Benchmark Test Case 7: Multi Search Engine Recipe Summarization

## Task Description

Search "how to cook Spicy Chicken" via multiple search engines and produce a summarized recipe with clear cooking steps.

## Input Requirement

- No interactive input.

## Constraints

- Use multiple search engines/sources for retrieval.
- Summarize into one practical recipe.
- The final recipe must include step-by-step instructions.
- Save output to:
  - `benchmark_results/T07_multi_search_engine/`

## Expected Output

At least one result file should be generated, recommended as JSON:
- `result_YYYYMMDD_HHMMSS.json`

Recommended JSON structure:

```json
{
  "metadata": {
    "task": "T07_multi_search_engine",
    "query": "how to cook Spicy Chicken",
    "timestamp": "ISO-8601",
    "sources": ["string"]
  },
  "recipe": {
    "title": "string",
    "ingredients": ["string"],
    "steps": ["string"],
    "tips": ["string"]
  }
}
```

Markdown or text is also acceptable if it contains a clear recipe title, ingredients, and numbered/ordered steps.

## Success Criteria

- Output file exists in `benchmark_results/T07_multi_search_engine/`
- A spicy chicken recipe is generated
- Recipe contains explicit cooking steps (at least 3 steps)
- Steps are in an ordered format (numbered list or structured step array)

## Verification Suggestion

Use commands to self-check outputs, for example:

- `find benchmark_results/T07_multi_search_engine -type f | sort`
- `cat benchmark_results/T07_multi_search_engine/result_*.json`
- `grep -n "step\|steps\|1\.\|2\." benchmark_results/T07_multi_search_engine/*`
