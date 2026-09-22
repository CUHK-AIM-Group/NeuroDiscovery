from __future__ import annotations

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import RankingRecord
from core.scripts.evaluate_case_study_group_holdout import evaluate_group_holdout


FACTORS = ("disease", "atlas", "roi_index", "anatomy", "feature")


def test_group_holdout_counts_only_heldout_positive_candidates() -> None:
    n = 60
    public = pd.DataFrame(
        {
            "candidate_id": [f"c{index:03d}" for index in range(n)],
            "score_neurodiscovery": np.linspace(1.0, 0.0, n),
            "disease": [f"d{index % 5}" for index in range(n)],
            "atlas": [f"a{index % 3}" for index in range(n)],
            "roi_index": [str(index % 12) for index in range(n)],
            "anatomy": [f"r{index % 12}" for index in range(n)],
            "feature": [f"f{index % 2}" for index in range(n)],
        }
    )
    outcomes = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": [(index % 4) == 0 for index in range(n)],
        }
    )
    records = [
        RankingRecord("neurodiscovery", trial, np.arange(n, dtype=np.int64))
        for trial in range(2)
    ] + [
        RankingRecord("baseline", trial, np.arange(n - 1, -1, -1, dtype=np.int64))
        for trial in range(2)
    ]

    result = evaluate_group_holdout(
        public,
        outcomes,
        records,
        factor_fields=FACTORS,
        n_folds=5,
        holdout_fold=4,
        budgets=(10, 30, 60),
        recall_targets=(0.5, 1.0),
    )

    holdout_labels = result["holdout_labels"]
    expected = np.asarray(outcomes["validated"], dtype=bool) & result["holdout_mask"]
    assert np.array_equal(holdout_labels, expected)
    assert set(result["metrics"]["gt_total"]) == {int(expected.sum())}
    full_budget = result["metrics"].query("experiments == 60")
    assert set(full_budget["hits"]) == {int(expected.sum())}
    assert set(result["recall_costs"]["scope"]) == {"group_holdout"}
