"""Mixed-effects and covariate-adjusted inference."""

from __future__ import annotations

from typing import Any

import pandas as pd


def fit_mixed_effects(
    frame: pd.DataFrame,
    formula: str,
    group_col: str,
    re_formula: str | None = None,
    **fit_kwargs: Any,
):
    """Fit a statsmodels linear mixed model and return its result object."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for mixed-effects models") from exc
    model = smf.mixedlm(
        formula=formula,
        data=frame,
        groups=frame[group_col],
        re_formula=re_formula,
    )
    return model.fit(**fit_kwargs)


def fit_ols(frame: pd.DataFrame, formula: str, robust_cov: str | None = "HC3"):
    """Fit formula OLS with optional heteroscedasticity-robust covariance."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for formula OLS") from exc
    result = smf.ols(formula=formula, data=frame).fit()
    return result.get_robustcov_results(cov_type=robust_cov) if robust_cov else result


def fit_glm(
    frame: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    robust_cov: str | None = "HC3",
):
    """Fit a formula GLM for Gaussian, binomial, Poisson, or Gamma outcomes."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for GLM") from exc
    families = {
        "gaussian": sm.families.Gaussian,
        "binomial": sm.families.Binomial,
        "poisson": sm.families.Poisson,
        "gamma": sm.families.Gamma,
    }
    key = family.lower()
    if key not in families:
        raise ValueError(f"Unsupported GLM family: {family}")
    result = smf.glm(
        formula=formula,
        data=frame,
        family=families[key](),
    ).fit(cov_type=robust_cov if robust_cov else "nonrobust")
    return result


def fit_dose_response_mixed_effects(
    frame: pd.DataFrame,
    outcome: str,
    dose: str,
    group_col: str,
    covariates: list[str] | None = None,
    moderator: str | None = None,
    quadratic: bool = True,
    **fit_kwargs: Any,
):
    """Fit a longitudinal dose-response curve with optional moderation."""
    terms = [dose]
    if quadratic:
        terms.append(f"I({dose} ** 2)")
    if moderator:
        terms.extend([moderator, f"{dose}:{moderator}"])
    terms.extend(covariates or [])
    formula = f"{outcome} ~ " + " + ".join(terms)
    return fit_mixed_effects(
        frame,
        formula=formula,
        group_col=group_col,
        re_formula=f"~{dose}",
        **fit_kwargs,
    )
