from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from core.scripts.case1_codex_session_shadow_launcher import (
    _load_adapter_module,
    install_shadow_capture,
)


class FakeCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.set_calls = 0

    def _generate_cache_key(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        return "a" * 64

    def get(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    def set(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.set_calls += 1


def fake_adapter() -> ModuleType:
    module = ModuleType("fake_adapter")

    def install(cache: object, *, out_dir: Path, identity_sha256: str) -> None:
        del cache, out_dir, identity_sha256

    def write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    module._install_open_coscientist_cache_audit = install
    module._write_json_atomic = write_json
    return module


def test_shadow_capture_writes_pending_request_not_formal_cache(
    tmp_path: Path,
) -> None:
    module = fake_adapter()
    capture_dir = tmp_path / "codex_session_shadow"
    install_shadow_capture(
        module,
        capture_dir=capture_dir,
        backend_model="gpt-5.6-luna",
        reasoning_effort="max",
        thread_id="01a04745-3f75-7ec0-a67d-323a0274c2a5",
    )
    cache = FakeCache(tmp_path / "formal_cache")
    module._install_open_coscientist_cache_audit(
        cache,
        out_dir=tmp_path,
        identity_sha256="identity-test",
    )

    with pytest.raises(RuntimeError, match="shadow request captured"):
        cache.get(
            "outcome-blind prompt",
            "openai/deepseek-v4-pro",
            0.7,
            4000,
            json_schema={"name": "answer", "schema": {"type": "object"}},
            force_json=True,
        )

    pending = json.loads(
        (capture_dir / "pending" / f"{'a' * 64}.json").read_text(
            encoding="utf-8"
        )
    )
    assert pending["scientific_classification"] == (
        "mixed_backend_user_approved_pending_import"
    )
    assert pending["formal_cache_written"] is False
    assert pending["cache_identity_sha256"] == "identity-test"
    assert pending["requested_backend"]["model"] == "gpt-5.6-luna"
    assert pending["requested_backend"]["reasoning_effort"] == "max"
    assert pending["request"]["prompt"] == "outcome-blind prompt"
    assert pending["request"]["json_schema"]["name"] == "answer"
    assert cache.set_calls == 0
    assert not cache.cache_dir.exists()


def test_shadow_capture_rejects_path_outside_trial(tmp_path: Path) -> None:
    module = fake_adapter()
    install_shadow_capture(
        module,
        capture_dir=tmp_path.parent / "outside",
        backend_model="gpt-5.6-luna",
        reasoning_effort="max",
        thread_id="01a04745-3f75-7ec0-a67d-323a0274c2a5",
    )

    with pytest.raises(RuntimeError, match="must be inside the trial"):
        module._install_open_coscientist_cache_audit(
            FakeCache(tmp_path / "formal_cache"),
            out_dir=tmp_path,
            identity_sha256="identity-test",
        )


def test_load_adapter_module_requires_exact_sealed_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sealed_adapter.py"
    source.write_text("MARKER = 'sealed'\n", encoding="utf-8")
    expected = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv("CS1_SEALED_ADAPTER_SOURCE", str(source))
    monkeypatch.setenv("CS1_SEALED_ADAPTER_SHA256", expected)

    module = _load_adapter_module()

    assert module.MARKER == "sealed"
    assert Path(module.__file__) == source


def test_load_adapter_module_rejects_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sealed_adapter.py"
    source.write_text("MARKER = 'sealed'\n", encoding="utf-8")
    monkeypatch.setenv("CS1_SEALED_ADAPTER_SOURCE", str(source))
    monkeypatch.setenv("CS1_SEALED_ADAPTER_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_adapter_module()
