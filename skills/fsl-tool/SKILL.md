---
name: fsl-tool
description: "Plan or run FSL structural, functional and diffusion MRI commands. Choose the requested analysis stage from the available inputs, including tracking from existing bedpostx results."
license: MIT License (NeuroClaw custom skill)
layer: base
skill_type: tool
dependencies:
  - claw-shell
---
# FSL Tool

Use FSL for the requested analysis stage. Existing processed inputs do not
imply that preprocessing should be repeated. Keep the user's output location.
Loading this document provides method guidance; it does not install FSL,
execute commands, launch a delegate, or verify images.

## Select the stage

| Supplied input / requested result | Relevant command |
|---|---|
| T1w / structural preprocessing | `fsl_anat` |
| Brain extraction only | `bet` |
| Tissue segmentation of extracted anatomy | `fast` |
| fMRI motion correction | `mcflirt` |
| Task GLM with an actual design file | `feat design.fsf` |
| Resting-state ICA | `melodic` |
| Raw diffusion with acquisition metadata | `topup` / `eddy` as applicable |
| Corrected diffusion / tensor maps | `dtifit` |
| Corrected diffusion / crossing-fibre model | `bedpostx` |
| Existing bedpostx samples / probabilistic tracking | `probtrackx2` |
| Requested atlas-protocol tract extraction with registration | `xtract` |

Parameters such as BET fraction, ICA dimension and streamline count are
analysis choices, not defaults that apply to every dataset. Check the installed
tool's help and FSL installation before proposing version-sensitive flags.
Use `claw-shell` when it is exposed as the real executor; otherwise provide
concrete external commands for the user or available executor.

## Tracking from existing bedpostx outputs

The samples prefix is `<bedpostx_dir>/merged`, representing the
`merged_th<i>samples`, `merged_ph<i>samples` and `merged_f<i>samples` images.
Use `nodif_brain_mask` as the tracking mask. A raw `bvecs` file is not a samples
prefix; raw bvals/bvecs and a second model-fitting run are unnecessary here.

Specify an actual seed and its coordinate space. When a seed is not supplied,
check for an existing suitable ROI or explicitly make seed selection a required
parameter. Do not silently label whole-brain seeding as a named anatomical tract.
Masks must share the declared seed space; non-diffusion-space masks require the
corresponding transforms. Single-voxel/surface seeds also require a reference.

```bash
# BPX, SEED and OUT are actual user-selected paths; SEED is in diffusion space.
probtrackx2 --samples="$BPX/merged" --mask="$BPX/nodif_brain_mask" \
  --seed="$SEED" --dir="$OUT" --opd --nsamples=5000
```

Check inputs before dispatch, preserve exit status and logs, then check
`fdt_paths.nii.gz` and `waytotal`. These counts are not an anatomical correctness
or scientific connectivity guarantee. [ProbtrackX documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)
and [FSL tracking practical](https://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/fdt2/).

## Optional XTRACT protocol extraction

Choose XTRACT when standardized protocol extraction is actually requested and
its masks and registration are available. Adult-human example:

```bash
xtract -bpx "$BPX" -out "$OUT" -species HUMAN \
  -stdwarp "$STD2DIFF" "$DIFF2STD"
```

`-p` takes a protocol directory, not an `HCP` enumeration. The documented default
is `${FSLDIR}/data/xtract_data/<SPECIES>`; verify the installed resources.
`-str` takes a structures file. Do not invent a `-nthreads` option.
Verify the diffusion-to-standard warp pair instead of assuming a bare bedpostx
folder contains them. Outputs include `tracts/<tract>/density.nii.gz`,
`densityNorm.nii.gz`, `waytotal`, plus commands and logs.
[XTRACT documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/xtract.html).

## Execution and verification

Return the task's missing-input status before attempting an unavailable stage.
Execution code should perform the command, propagate failure and record measured
results; printing a delegation payload is only a plan. For background jobs,
record PID/log/exit-status location and distinguish running, failed and verified
completion. A launched PID or an old output file alone is not completion.

Any `VerificationRunner` integration needs an exposed, implemented caller and
real checks. Reading this skill does not automatically invoke it. Apply output
checks for the requested stage, rather than unrelated BET/registration checks.

Other FSL tools: [official documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/).
