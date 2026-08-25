"""PLINK2 command construction without hiding shell execution."""

from __future__ import annotations

from pathlib import Path


def build_plink2_command(
    pfile: str | Path,
    phenotype: str | Path,
    phenotype_name: str,
    output_prefix: str | Path,
    covariates: str | Path | None = None,
    covariate_names: list[str] | None = None,
    binary: bool = False,
) -> list[str]:
    command = [
        "plink2",
        "--pfile",
        str(pfile),
        "--pheno",
        str(phenotype),
        "--pheno-name",
        phenotype_name,
        "--glm",
        "hide-covar",
        "cols=+a1freq,+nobs,+beta,+se,+p",
        "--out",
        str(output_prefix),
    ]
    if binary:
        command.extend(["--1"])
    if covariates:
        command.extend(["--covar", str(covariates)])
        if covariate_names:
            command.extend(["--covar-name", ",".join(covariate_names)])
    return command
