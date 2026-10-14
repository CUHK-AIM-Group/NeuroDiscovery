---
name: freesurfer-tool
description: "Plan or run FreeSurfer recon-all from structural MRI, resume an existing subject, or validate surfaces, segmentations and statistics."
license: MIT License (NeuroClaw custom skill)
layer: base
skill_type: tool
dependencies:
  - claw-shell
---
# FreeSurfer Tool

Choose the reconstruction stage requested by the user. Preserve provided image,
subject and output paths. This document supplies methods; loading it does not
execute reconstruction, install a license, delegate work or invoke verification.

## Inputs and environment

For a new full subject, require the actual local T1w image. Optional T2w or FLAIR
must exist when supplied; report an invalid optional path rather than silently
ignoring it. Do not replace unavailable anatomy with examples or synthetic data.
Check the FreeSurfer installation, setup script and license before execution.
Run environment setup and the command in the same shell/process environment.
Use an available shell executor, including `claw-shell` when exposed; when asked
for code, give a runnable external command rather than an unresolved tool call.

## New subject versus continuation

For an unused subject directory, initialize and run the complete workflow:

```bash
export SUBJECTS_DIR="$OUT/subjects"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
mkdir -p "$SUBJECTS_DIR"
recon-all -sd "$SUBJECTS_DIR" -s "$SUBJECT" -i "$T1W" -all
```

The variables represent user-selected paths. Quote them and validate the subject
identifier. A new import uses `-i`; continuation of an existing initialized
subject uses its existing MRI inputs and omits `-i`. Do not silently overwrite
or import again because a completion marker is absent. A marker must be checked
alongside the actual exit status, inputs and required outputs.

For pial refinement choose **one** available modality: `-T2 "$T2W" -T2pial`,
or `-FLAIR "$FLAIR" -FLAIRpial`. If both are supplied, state the chosen modality
or request the clinically relevant preference; do not enable both pial modes
without verifying version-specific support. Passing `-FLAIR` alone does not
request FLAIR pial refinement. For an already processed subject needing only
refinement, choose the appropriate continuation stages such as `-autorecon3`.
[recon-all reference](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all).

## Long jobs and results

Foreground code can wait for recon-all and preserve its return code. A detached
wrapper must persist PID, command, inputs, stdout/stderr log and terminal exit
status. Launch success is a running state. Only mark completion after the process
finishes successfully and the required artifacts pass checks. Do not conceal
failed child processes with `wait ... || true`.

Under `$SUBJECTS_DIR/$SUBJECT`, check the files required by the requested stage:

| Output | Relative location |
|---|---|
| Segmentation | `mri/aseg.mgz`, `mri/aparc+aseg.mgz` |
| White/pial surfaces | `surf/lh.white`, `surf/rh.white`, `surf/lh.pial`, `surf/rh.pial` |
| Thickness | `surf/lh.thickness`, `surf/rh.thickness` |
| Cortical annotations | `label/lh.aparc.annot`, `label/rh.aparc.annot` |
| Statistics | `stats/aseg.stats`, `stats/lh.aparc.stats`, `stats/rh.aparc.stats` |
| Native logs/status | `scripts/recon-all.log`, `scripts/recon-all.done` |

Build paths with their directory prefixes; the names alone are not relative to
the subject root. Use a JSON serializer for summaries containing paths and logs.
Report missing, running, failed and verified outcomes separately. Image/surface
quality needs content review beyond file existence; a single thickness range is
not a universal pass criterion. [Output-file reference](https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllOutputFiles).

## Repository adapters

`scripts/freesurfer_processor.py` is a historical adapter/example. Inspect its
implementation before choosing it as an executor. A helper that prints commands
or returns a fixed installation status is illustrative, not an executed handoff.
Likewise, verification requires actual implemented checks and an explicit caller;
there is no automatic `VerificationRunner` call when this document is read.

For installation use the [official installation guide](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
and the installed version's help. Do not assume a latest version or local path.
