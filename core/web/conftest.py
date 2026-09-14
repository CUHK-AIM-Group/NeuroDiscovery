"""Web regressions never read or overwrite a user's durable client history."""
import pytest


@pytest.fixture(autouse=True)
def isolated_workbench_database(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURODISCOVERY_WORKBENCH_DB", str(tmp_path / "workbench.sqlite3"))
