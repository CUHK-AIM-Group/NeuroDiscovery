# T277_container_registry_push: Push Image to GHCR
## Task Description

Tag and push a built analysis image to GitHub Container Registry with semantic version tags and a usage README.

## Input Requirement


- No interactive input.


## Constraints

- Tags: version + sha; no `latest` only.

- README documents run command for the image.

- Save all generated artifacts to:
  - benchmark_results/T277_container_registry_push/


## Expected Output

Expected output artifact(s):

- Push log

- `IMAGE_README.md`

- `tags.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
