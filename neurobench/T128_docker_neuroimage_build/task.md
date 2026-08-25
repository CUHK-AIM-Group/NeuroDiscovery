# T128_docker_neuroimage_build: Docker Image for FSL/AFNI/ANTs Toolchain
## Task Description

Author a Dockerfile for a reproducible neuroimaging analysis image containing
FSL, AFNI, and ANTs, build it, and smoke-test each tool inside the container.

## Input Requirement

- No interactive input.

## Constraints

- Pin base image and tool versions (no `latest` tags).
- Keep the image as small as practical (multi-stage build or neurodocker
  generated recipe preferred; document the choice).
- Smoke tests must run non-interactively: `flirt -version`, `afni -ver`,
  `antsRegistration --version` inside the built image.
- Save all generated artifacts to:
  - benchmark_results/T128_docker_neuroimage_build/

## Expected Output

Expected output artifact(s):

- `Dockerfile` (and neurodocker command if used)
- `build_log.txt` + final image size and tag
- `smoke_test.json` (per-tool: pass/fail + version string)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
