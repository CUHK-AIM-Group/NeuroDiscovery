"""Mass-univariate ROI GLM with multiplicity correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def fit_roi_glm(
    roi_values: np.ndarray,
    design: np.ndarray,
    roi_names: list[str] | None = None,
    contrast_index: int = 1,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for ROI GLM") from exc
    Y = np.asarray(roi_values, dtype=float)
    X = np.asarray(design, dtype=float)
    if Y.ndim != 2 or X.ndim != 2 or len(Y) != len(X):
        raise ValueError("ROI values and design must be aligned 2D matrices")
    if contrast_index >= X.shape[1]:
        raise ValueError("contrast_index exceeds design columns")
    names = roi_names or [f"ROI_{index + 1}" for index in range(Y.shape[1])]
    rows = []
    for index, name in enumerate(names):
        result = sm.OLS(Y[:, index], X, missing="drop").fit(cov_type="HC3")
        rows.append(
            {
                "roi": name,
                "beta": float(result.params[contrast_index]),
                "standard_error": float(result.bse[contrast_index]),
                "t": float(result.tvalues[contrast_index]),
                "p_value": float(result.pvalues[contrast_index]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["q_value"] = multipletests(frame["p_value"], method=correction)[1]
    return frame.sort_values("q_value").reset_index(drop=True)
