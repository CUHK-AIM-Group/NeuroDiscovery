# T379_citation_style_conversion: Citation Style Conversion
## Task Description

Convert the manuscript bibliography to a different citation style (e.g. APA -> Vancouver) using CSL, verifying every entry renders.

## Input Requirement

Required input(s):

- Zotero export / .bib / PDF folder as applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- CSL file used is recorded.

- Spot-check 10 entries against the style guide.

- Save all generated artifacts to:
  - benchmark_results/T379_citation_style_conversion/


## Expected Output

Expected output artifact(s):

- Converted bibliography

- `style_check.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
