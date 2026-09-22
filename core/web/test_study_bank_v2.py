"""v2 extension bank artifact invariants (built artifact checked, not rebuilt)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_expert_study_v2.json"
TRUTH = ROOT / "neurooracle" / "data" / "user_study" / "private_truth" / "case1_tcp_expert_study_v2_truth.json"
MIN_SCORE_GAP = 0.20


class StudyBankV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = json.loads(BANK.read_text(encoding="utf-8"))
        cls.truth = json.loads(TRUTH.read_text(encoding="utf-8"))
        cls.by_id = {c["id"]: c for c in cls.bank["hypotheses"]}

    def test_shape_and_sessions(self):
        bank = self.bank
        self.assertEqual(bank["n_pairs"], 240)
        self.assertEqual(len(bank["pair_schedule"]), 240)
        self.assertEqual(bank["n_hypotheses"], 480)
        self.assertEqual(len(bank["hypotheses"]), 480)
        for n in range(1, 7):
            self.assertEqual(sum(1 for p in bank["pair_schedule"] if p["session_number"] == n), 40, n)
        counts = {d: sum(1 for p in bank["pair_schedule"] if p["difficulty"] == d)
                  for d in ("easy", "medium", "hard")}
        self.assertEqual(counts, {"easy": 80, "medium": 80, "hard": 80})

    def test_gap_floor_and_difficulty_ordering(self):
        gaps = [p["generator_score_gap"] for p in self.bank["pair_schedule"]]
        self.assertGreaterEqual(min(gaps), MIN_SCORE_GAP)
        by = {d: [p["generator_score_gap"] for p in self.bank["pair_schedule"] if p["difficulty"] == d]
              for d in ("easy", "medium", "hard")}
        self.assertGreaterEqual(min(by["easy"]), max(by["medium"]))
        self.assertGreaterEqual(min(by["medium"]), max(by["hard"]))

    def test_pairs_reference_existing_cards_and_sides_are_balanced(self):
        truth_by_pair = {p["pair_id"]: p for p in self.truth["pairs"]}
        sides = {"left": 0, "right": 0}
        for pair in self.bank["pair_schedule"]:
            self.assertIn(pair["left_id"], self.by_id)
            self.assertIn(pair["right_id"], self.by_id)
            self.assertNotEqual(pair["left_id"], pair["right_id"])
            truth = truth_by_pair[pair["pair_id"]]
            confirmed = truth["confirmed_candidate_id"]
            self.assertIn(confirmed, (pair["left_id"], pair["right_id"]))
            self.assertEqual(truth["confirmed_side"], "left" if confirmed == pair["left_id"] else "right")
            self.assertGreater(truth["confirmed_score"], truth["other_score"])
            sides[truth["confirmed_side"]] += 1
        self.assertEqual(sides, {"left": 120, "right": 120})

    def test_no_outcome_columns_in_public_bank(self):
        text = json.dumps(self.bank, ensure_ascii=False)
        for forbidden in ("confirmed_side", "strict_validated", "cv_auc", "adjusted_residual_d", "q_fdr"):
            self.assertNotIn(forbidden, text)
        for card in self.bank["hypotheses"]:
            self.assertNotIn("validated", card)

    def test_cards_bilingual_with_literature(self):
        for card in self.bank["hypotheses"]:
            self.assertTrue(card["title"] and card["title_zh"])
            self.assertTrue(card["summary"] and card["summary_zh"])
            self.assertEqual(len(card["literature"]), 5, card["id"])
        reused = [c for c in self.bank["hypotheses"]
                  if any(item.get("manual_relevance_status") == "completed" for item in c["literature"])]
        self.assertEqual(len(reused), 9)
        auto = [c for c in self.bank["hypotheses"]
                if all(item.get("manual_relevance_status") == "auto_kg_match" for item in c["literature"])]
        self.assertEqual(len(auto) + len(reused), 480)


if __name__ == "__main__":
    unittest.main()
