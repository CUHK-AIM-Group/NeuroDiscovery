"""Prepare harmonized metadata tables for Case Study 1 external validation.

The exhaustive runner expects the TCP-style columns ``subjectkey``, ``Age``,
``sex``, ``Site``, ``diagnosis_broad_any``, and ``is_genpop``.  This script
maps the frozen ADHD-200, COBRE, HCP-EP, and UCLA metadata into that schema
without using any imaging outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("adhd200", "cobre", "hcpep", "ucla")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _adhd_subject(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return f"{int(float(text)):07d}"
    except ValueError:
        return text.zfill(7)


def _prepare_adhd200(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, dtype={"subject_id": str})
    dx = pd.to_numeric(raw["DX"], errors="coerce")
    out = pd.DataFrame(
        {
            "subjectkey": raw["subject_id"].map(_adhd_subject),
            "Age": pd.to_numeric(raw["Age"], errors="coerce"),
            "sex": raw["Gender"].map(
                lambda value: "M" if _text(value) in {"1", "1.0"} else "F"
                if _text(value) in {"0", "0.0"}
                else ""
            ),
            "Site": raw["Site"].map(_text),
            "diagnosis_broad_any": np.where(dx.isin([1, 2, 3]), "ADHD", ""),
            "is_genpop": dx.eq(0),
            "Group": np.where(dx.eq(0), "genpop", "patient"),
            "source_diagnosis": raw["DX"].map(_text),
            "mean_fd": np.nan,
        }
    )
    return out[dx.isin([0, 1, 2, 3])].copy()


def _prepare_cobre(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, dtype={"subject_id": str})
    diagnosis = raw["dx"].map(_text)
    disease = diagnosis.map(
        {
            "Schizophrenia_Strict": "psychosis_SZ_SZA",
            "Schizoaffective": "psychosis_SZ_SZA",
            "Bipolar_Disorder": "bipolar",
            "No_Known_Disorder": "",
        }
    ).fillna("")
    control = diagnosis.eq("No_Known_Disorder")
    return pd.DataFrame(
        {
            "subjectkey": raw["subject_id"].map(_text),
            "Age": pd.to_numeric(raw["age"], errors="coerce"),
            "sex": raw["sex"].map(
                lambda value: "M" if _text(value).casefold().startswith("m") else "F"
                if _text(value).casefold().startswith("f")
                else ""
            ),
            "Site": raw["study"].map(_text),
            "diagnosis_broad_any": disease,
            "is_genpop": control,
            "Group": np.where(control, "genpop", "patient"),
            "source_diagnosis": diagnosis,
            "mean_fd": np.nan,
        }
    )


def _prepare_hcpep(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, dtype={"subject_id": str})
    diagnosis = raw["phenotype_description"].map(_text)
    disease = diagnosis.map(
        {
            "Non-affective psychosis": "psychosis_SZ_SZA",
            "Affective psychosis": "psychosis_affective_sensitivity",
            "In good health": "",
        }
    ).fillna("")
    control = diagnosis.eq("In good health")
    return pd.DataFrame(
        {
            "subjectkey": raw["subject_id"].map(lambda value: f"sub-{_text(value)}"),
            "Age": pd.to_numeric(raw["interview_age"], errors="coerce") / 12.0,
            "sex": raw["sex"].map(lambda value: _text(value).upper()),
            "Site": raw["site"].map(_text),
            "diagnosis_broad_any": disease,
            "is_genpop": control,
            "Group": np.where(control, "genpop", "patient"),
            "source_diagnosis": diagnosis,
            "mean_fd": np.nan,
        }
    )


def _prepare_ucla(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, dtype={"subject_id": str})
    diagnosis = raw["diagnosis"].map(_text)
    disease = diagnosis.map(
        {
            "SCHZ": "psychosis_SZ_SZA",
            "BIPOLAR": "bipolar",
            "ADHD": "ADHD",
            "CONTROL": "",
        }
    ).fillna("")
    control = diagnosis.eq("CONTROL")
    return pd.DataFrame(
        {
            "subjectkey": raw["subject_id"].map(_text),
            "Age": pd.to_numeric(raw["age"], errors="coerce"),
            "sex": raw["gender"].map(lambda value: _text(value).upper()),
            "Site": raw["ScannerSerialNumber"].map(_text),
            "diagnosis_broad_any": disease,
            "is_genpop": control,
            "Group": np.where(control, "genpop", "patient"),
            "source_diagnosis": diagnosis,
            "mean_fd": np.nan,
        }
    )


PREPARERS = {
    "adhd200": _prepare_adhd200,
    "cobre": _prepare_cobre,
    "hcpep": _prepare_hcpep,
    "ucla": _prepare_ucla,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = PREPARERS[args.dataset](args.source)
    table = table[table["subjectkey"].astype(str).str.len() > 0].copy()
    table = table.drop_duplicates("subjectkey", keep="first")
    table["dataset"] = args.dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    summary = {
        "dataset": args.dataset,
        "source": str(args.source),
        "output": str(args.output),
        "subjects": int(len(table)),
        "controls": int(table["is_genpop"].astype(bool).sum()),
        "diagnosis_counts": {
            str(key or "control"): int(value)
            for key, value in table["diagnosis_broad_any"].value_counts(dropna=False).items()
        },
        "covariates": ["Age", "sex", "Site"],
        "mean_fd_available": bool(pd.to_numeric(table["mean_fd"], errors="coerce").notna().any()),
    }
    summary_path = args.output.with_suffix(".manifest.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
