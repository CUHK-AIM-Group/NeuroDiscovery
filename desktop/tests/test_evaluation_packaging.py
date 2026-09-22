import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluation_stager", ROOT / "desktop/scripts/stage-evaluation-runtime.py")
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)


class EvaluationPackagingTests(unittest.TestCase):
    def test_exact_backend_inventory_and_frozen_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "backend"
            files = stager.stage_backend(ROOT, target)
            self.assertEqual(len(files), 29)
            self.assertNotIn("core/web/server.py", files)
            self.assertFalse(any("organizer" in name or name.endswith(".sqlite3") for name in files))
            for name in stager.BANK_FILES:
                relative = f"neurooracle/data/user_study/{name}"
                self.assertEqual(files[relative], hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
            pack = json.loads(next((target / "core/web/study_materials").glob("*desktop_v1.json")).read_text(encoding="utf-8"))
            self.assertNotIn("organizer", pack)
            self.assertEqual(len(pack["cards"]), 10)
            with self.assertRaises(ValueError):
                stager.stage_backend(ROOT, target)

    def test_unsafe_and_dirty_targets_refused_before_writing(self):
        for target in [ROOT, ROOT.parent, ROOT / "desktop/runtime/backend/new", ROOT / "core/new"]:
            with self.assertRaises(ValueError):
                stager.stage_backend(ROOT, target)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "keep.txt").write_text("untouched")
            with self.assertRaises(ValueError):
                stager.stage_backend(ROOT, target)
            self.assertEqual([path.name for path in target.iterdir()], ["keep.txt"])

    def test_separate_product_and_artifacts(self):
        evaluation = json.loads((ROOT / "desktop/electron-builder.evaluation.json").read_text())
        full = json.loads((ROOT / "desktop/package.json").read_text())["build"]
        self.assertNotEqual(evaluation["appId"], full["appId"])
        self.assertNotEqual(evaluation["directories"]["output"], full["directories"]["output"])
        self.assertNotEqual(evaluation["nsis"]["artifactName"], evaluation["portable"]["artifactName"])
        self.assertEqual(evaluation["files"], ["evaluation-main.js", "evaluation-preload.js", "package.json"])


if __name__ == "__main__":
    unittest.main()
