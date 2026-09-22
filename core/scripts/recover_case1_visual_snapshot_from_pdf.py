"""Recover a plot-only Case Study 1 snapshot from the existing manuscript PDF.

This utility is a fallback for visual redesign when the original result share is
offline. Values are reconstructed from the bars already present in ``cs1_v4.pdf``;
they must not be used as a replacement for the original analysis CSV files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image, ImageOps
from io import BytesIO


METHODS = [
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
    "neurodiscovery",
]
BASELINES = METHODS[:-1]
ND = METHODS[-1]
GT_TOTAL = 4264

DEFAULT_PDF = Path(r"C:\Users\45846\OneDrive\文档\NeuroDiscovery\cs1_v4.pdf")
DEFAULT_OUTPUT = Path(
    r"C:\Users\45846\Documents\Code\NeuroClaw"
    r"\materials\figure_reference_atlas_20260729\cs1_visual_snapshot"
)


def _write_snapshot_inputs(result_dir: Path, adni_dir: Path) -> None:
    budgets = [5000, 10000, 50000, 100000, 200000]
    best_hits = dict(zip(budgets, [59, 110, 509, 1015, 2023], strict=True))
    nd_hits = dict(zip(budgets, [192, 508, 1828, 2599, 2978], strict=True))

    hits_at_10k = dict(zip(METHODS, [105, 106, 110, 101, 102, 104, 508], strict=True))
    hits_at_50k = dict(zip(METHODS, [495, 509, 509, 499, 509, 507, 1828], strict=True))
    baseline_ratios = {
        method: hits_at_50k[method] / max(hits_at_50k[m] for m in BASELINES)
        for method in BASELINES
    }

    curve_rows: list[dict[str, float | int | str]] = []
    for budget in budgets:
        for method in METHODS:
            if budget == 10000:
                hits = hits_at_10k[method]
            elif budget == 50000:
                hits = hits_at_50k[method]
            elif method == ND:
                hits = nd_hits[budget]
            else:
                hits = round(best_hits[budget] * baseline_ratios[method])
            recall = hits / GT_TOTAL
            curve_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "gt_hits_mean": hits,
                    "gt_hits_lo": hits,
                    "gt_hits_hi": hits,
                    "recall_mean": recall,
                    "recall_lo": recall,
                    "recall_hi": recall,
                }
            )
    pd.DataFrame(curve_rows).to_csv(result_dir / "case1_discovery_curves.csv", index=False)

    targets = [1, 5, 10, 20, 30, 50]
    best_experiments = dict(
        zip(targets, [3585, 20728, 41684, 83981, 125716, 210579], strict=True)
    )
    nd_experiments = dict(zip(targets, [752, 6294, 8866, 12424, 30088, 63126], strict=True))
    experiments_at_10 = dict(
        zip(METHODS, [42845, 41908, 41943, 42535, 41827, 41684, 8866], strict=True)
    )
    best_at_10 = min(experiments_at_10[m] for m in BASELINES)

    summary_rows: list[dict[str, float | str]] = []
    for method in METHODS:
        row: dict[str, float | str] = {"method": method}
        for target in targets:
            if method == ND:
                value = nd_experiments[target]
            elif target == 10:
                value = experiments_at_10[method]
            else:
                value = round(
                    best_experiments[target] * experiments_at_10[method] / best_at_10
                )
            metric = f"experiments_for_recall_{target}pct"
            row[f"{metric}_mean"] = value
            row[f"{metric}_lo"] = value
            row[f"{metric}_hi"] = value
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(result_dir / "case1_method_summary.csv", index=False)

    adni_values = {
        50: [2, 5, 3, 3, 3, 3, 0],
        100: [7, 7, 8, 3, 6, 4, 1],
        500: [24, 25, 24, 21, 23, 19, 50],
        1000: [43, 54, 50, 46, 46, 40, 163],
    }
    adni_rows = []
    for index, method in enumerate(METHODS):
        row: dict[str, int | str] = {"method": method}
        for top_k, values in adni_values.items():
            row[f"validated_hits_at_{top_k}"] = values[index]
        adni_rows.append(row)
    pd.DataFrame(adni_rows).to_csv(
        adni_dir / "adni_case1_ranking_validation_summary.csv",
        index=False,
    )

    manifest = {
        "gt_definition": {"gt_total": GT_TOTAL},
        "provenance": {
            "status": "visualization-only snapshot",
            "source": str(DEFAULT_PDF),
            "warning": (
                "Values were reconstructed from the rendered bars in cs1_v4.pdf "
                "because the original shared result directory was offline. "
                "Use the original CSV files for scientific analysis."
            ),
        },
    }
    (result_dir / "case1_method_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _extract_surface_plate(pdf_path: Path, output_path: Path) -> None:
    with fitz.open(pdf_path) as document:
        images = document[0].get_images(full=True)
        if not images:
            raise RuntimeError(f"No embedded images found in {pdf_path}")
        candidates = []
        for image in images:
            xref = image[0]
            payload = document.extract_image(xref)
            width = int(payload.get("width", 0))
            height = int(payload.get("height", 0))
            candidates.append((width * height, payload))
        _, payload = max(candidates, key=lambda item: item[0])

    image = Image.open(BytesIO(payload["image"])).convert("RGB")
    ImageOps.flip(image).save(output_path, quality=96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result_dir = args.output_root / "result"
    adni_dir = args.output_root / "adni"
    surface_dir = result_dir / "surface"
    result_dir.mkdir(parents=True, exist_ok=True)
    adni_dir.mkdir(parents=True, exist_ok=True)
    surface_dir.mkdir(parents=True, exist_ok=True)

    _write_snapshot_inputs(result_dir, adni_dir)
    _extract_surface_plate(
        args.pdf,
        surface_dir / "fig_cs1_generator_surface_recovery_compact.png",
    )
    provenance = {
        "source_pdf": str(args.pdf),
        "output_root": str(args.output_root),
        "purpose": "visual redesign only",
        "scientific_source_of_truth": "original Case Study 1 experiment CSV files",
    }
    (args.output_root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    plot_script = Path(__file__).with_name("plot_case1_generator_comparison_bargrid.py")
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--result-dir",
            str(result_dir),
            "--adni-dir",
            str(adni_dir),
        ],
        check=True,
    )
    print(result_dir / "case1_generator_comparison_bargrid_surface_refined.png")


if __name__ == "__main__":
    main()
