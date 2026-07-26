from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from core.web.server import _workspace_change_snapshot, _workspace_change_summary


class WorkspaceChangeSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.root, check=True)
        (self.root / "changed.txt").write_text("committed\n", encoding="utf-8")
        (self.root / "untouched.txt").write_text("committed\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.root, check=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reports_only_files_changed_after_the_turn_baseline(self) -> None:
        (self.root / "changed.txt").write_text("dirty before turn\n", encoding="utf-8")
        (self.root / "preexisting.txt").write_text("already untracked\n", encoding="utf-8")
        before = _workspace_change_snapshot(self.root)

        (self.root / "changed.txt").write_text("dirty and changed this turn\n", encoding="utf-8")
        (self.root / "new-this-turn.txt").write_text("new\n", encoding="utf-8")
        changes = _workspace_change_summary(self.root, before)

        self.assertEqual(
            {item["path"] for item in changes},
            {"changed.txt", "new-this-turn.txt"},
        )
        self.assertNotIn("preexisting.txt", {item["path"] for item in changes})

    def test_unchanged_dirty_workspace_produces_no_turn_changes(self) -> None:
        (self.root / "changed.txt").write_text("dirty before turn\n", encoding="utf-8")
        before = _workspace_change_snapshot(self.root)
        self.assertEqual(_workspace_change_summary(self.root, before), [])

    def test_timeline_label_describes_turn_scoped_changes(self) -> None:
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("Changes this turn", index_html)
        self.assertIn("本轮文件变更", index_html)
        self.assertNotIn("tr('Workspace changes')", index_html)
        self.assertIn("message.workspaceChangeScope !== 'turn'", index_html)
        self.assertIn("data.workspace_change_scope === 'turn'", index_html)


if __name__ == "__main__":
    unittest.main()
