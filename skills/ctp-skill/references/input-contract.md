# CTP input and numerical contract

## Time-series metadata

NIfTI shape is `[X,Y,Z,T]`, spatial units are `mm`, voxel values are HU **after** any NIfTI scaling. A JSON sidecar supplies acquisition times in seconds (one common time per volume), not a guessed repetition time:

```json
{
  "signal_units": "HU",
  "frame_times_s": [0, 1, 2, 3, 4, 5, 6, 7],
  "baseline_frames": [0, 1],
  "motion_corrected": true
}
```

The short timing array illustrates schema only, not an adequate clinical scan duration. Baseline indices are zero-based, consecutive initial frames with no contrast arrival, at least two. `motion_corrected=true` is an explicit user/researcher QA declaration, not a conclusion made by the deconvolution script. Preserve reviewer/method details in additional metadata fields. Supplied masks are nonempty 0/1 NIfTI; atlas labels are nonnegative integers; all must have the exact analysis geometry. AIF and VOF may lie outside a parenchymal mask but must be distinct, valid vascular regions. Exclude large vessels from tissue measurements.

The motion script uses rigid 3D registration to the first baseline frame, physical LPS coordinates, mutual information and linear resampling. Transform files map reference coordinates to moving coordinates. Its coverage mask excludes any voxel with nonfinite resampled values in any frame. Intersect all relevant masks with coverage and inspect registered images, not only displacement values. It does not correct within-frame motion, table-shuttle timing or contrast leakage.

## Computation

Baseline-subtract tissue and AIF/VOF HU without a logarithmic MR transform. Optionally scale AIF by `area(VOF)/area(AIF)`. Retain signed delta-HU noise. Uniform time spacing is required for convolution; optional linear interpolation never extrapolates and can shorten the last interval. Baseline estimation is on original frames.

The implementation solves `Ct = Ca * k` using rectangle-rule convolution (`dt` included), a zero-padded `2T × 2T` circulant matrix, and relative truncated SVD. Let `H` be the explicitly selected hematocrit scale (default 1, no correction) and `rho` the tissue density in g/mL (default 1.04):

- `CBF = (100 H / rho) * 60 * max(k)` in mL/100g/min.
- `CBV = (100 H / rho) * area(Ct)/area(Ca)` in mL/100g, using trapezoidal finite-window areas.
- `MTT = 60 CBV / CBF` in seconds, not the time of the tissue peak.
- `Tmax` is the time lag of the maximum deconvolved flow-scaled residue, not tissue TTP. The second circulant half represents negative lags and is exported as negative seconds, with QA counts.

Nonpositive enhancement/flow/volume yields invalid maps, not invented normal values. The inverse is shared and voxels processed in chunks. No denoising, leakage correction, adaptive oscillation-index regularization or scanner calibration is implied. Truncated acquisitions and SVD regularization bias estimates; density/AIF/hematocrit assumptions affect absolute units. Do not equate these reference maps to a validated vendor product. For quantitative use, verify with a suitable acquisition-specific phantom and independently validated software.

## Supplied maps

```json
{"units": {"CBF": "mL/100g/min", "CBV": "mL/100g", "MTT": "s", "Tmax": "s"}}
```

Convert vendor scaling and missing-value conventions upstream with documented provenance. Keep source maps unchanged. Negative CBF/CBV/MTT are rejected; signed Tmax is retained. NaNs are counted, not silently imputed. The summary route does not recompute or certify vendor maps.

## Sources

- [Fieselmann et al., theoretical model and implementation, 2011](https://pmc.ncbi.nlm.nih.gov/articles/3166726/).
- [Wu et al., block-circulant arrival-time-insensitive deconvolution, 2003](https://doi.org/10.1002/mrm.10522).
- [SimpleITK registration geometry and framework](https://simpleitk.readthedocs.io/en/master/registrationOverview.html).
