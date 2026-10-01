import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from neurooracle.src import candidate_ledger_v2 as ledger


def build_database(path, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        "create table attempts(job_id text, work_key text, source_sha256 text, input_sha256 text,"
        " provider text, model text, started_at text, status text, candidate_path text,"
        " finish_reason text, validation_json text, seconds real)")
    for row in rows:
        connection.execute(
            "insert into attempts values(:job_id,:work_key,:source_sha256,:input_sha256,:provider,"
            ":model,:started_at,:status,:candidate_path,:finish_reason,:validation_json,:seconds)", row)
    connection.commit()
    connection.close()


def row(**overrides):
    base = {"job_id": "PAPER:1", "work_key": "PMID:1", "source_sha256": "s", "input_sha256": "i",
            "provider": "ollama", "model": "m", "started_at": "2026-01-01T00:00:00",
            "status": "CANDIDATE_READY", "candidate_path": None, "finish_reason": "stop",
            "validation_json": json.dumps({"structural_pass": True, "hard_errors": []}), "seconds": 1.0}
    base.update(overrides)
    return base


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.candidate = self.root / "cand.json"
        self.candidate.write_text(json.dumps({"paper_id": "PMID:1", "observations": []}))

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, rows, packets=None):
        database = self.root / "ledger.sqlite"
        build_database(database, rows)
        return ledger.load_accepted_candidates(database, self.root, packets)

    def test_accepted_candidate_is_bound(self):
        result = self.load([row(candidate_path=str(self.candidate))])
        self.assertEqual(list(result["accepted"]), ["PAPER:1"])
        self.assertTrue(result["accepted"]["PAPER:1"]["structural_pass"])

    def test_failed_predecessor_does_not_win(self):
        result = self.load([row(candidate_path=str(self.candidate)),
                            row(started_at="2026-01-02T00:00:00", status="HELD",
                                validation_json=json.dumps({"hard_errors": ["bad"]}))])
        self.assertEqual(list(result["accepted"]), ["PAPER:1"])
        self.assertEqual(len(result["rejected"]), 1)

    def test_hard_error_is_rejected(self):
        result = self.load([row(candidate_path=str(self.candidate),
                                validation_json=json.dumps({"structural_pass": False, "hard_errors": ["x"]}))])
        self.assertEqual(result["accepted"], {})
        self.assertEqual(result["rejected"][0]["reason"], "hard_errors")

    def test_packet_digest_mismatch_is_flagged_not_accepted(self):
        packets = {"PAPER:1": {"work_key": "PMID:1", "source_sha256": "different", "input_sha256": "i"}}
        result = self.load([row(candidate_path=str(self.candidate))], packets)
        self.assertEqual(result["accepted"], {})
        self.assertIn("packet_source_sha256_mismatch", result["ambiguous"][0]["problems"])


if __name__ == "__main__":
    unittest.main()
