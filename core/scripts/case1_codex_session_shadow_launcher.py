"""Capture Open Co-Scientist cache misses for a Codex-session shadow backend.

This launcher patches the adapter only in memory.  The sealed formal adapter
source file and its SHA-256 remain unchanged, and captured requests are never
written to the formal LLM cache.  The capture records that the user approved a
mixed-backend continuation; a response still requires validation and a
separate, audited import before it becomes formal cache state.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import ModuleType
from typing import Any

from core.scripts import case_study_official_adapter_client as adapter


CAPTURE_SCHEMA = "case1-codex-session-shadow-request.v1"
SEALED_ADAPTER_MODULE_NAME = "_case1_sealed_official_adapter"


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _load_adapter_module() -> ModuleType:
    """Load an optional byte-sealed adapter without changing the shared source."""

    source_value = os.environ.get("CS1_SEALED_ADAPTER_SOURCE", "").strip()
    if not source_value:
        return adapter
    source = Path(os.path.abspath(source_value))
    if not source.is_file():
        raise RuntimeError(f"sealed adapter source does not exist: {source}")
    expected_sha256 = _required_environment(
        "CS1_SEALED_ADAPTER_SHA256"
    ).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("invalid sealed adapter SHA-256")
    actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "sealed adapter SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        SEALED_ADAPTER_MODULE_NAME,
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sealed adapter source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SEALED_ADAPTER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(SEALED_ADAPTER_MODULE_NAME, None)
        raise
    return module


def install_shadow_capture(
    adapter_module: ModuleType,
    *,
    capture_dir: Path,
    backend_model: str,
    reasoning_effort: str,
    thread_id: str,
) -> None:
    """Wrap the adapter's cache audit with a shadow-only miss capture."""

    if not re.fullmatch(r"[0-9a-fA-F-]{20,80}", thread_id):
        raise RuntimeError("invalid Codex thread id")
    original_install = adapter_module._install_open_coscientist_cache_audit
    # Keep a mapped drive path mapped.  Path.resolve() expands ``R:`` back to
    # the much longer UNC path on Windows, which can cross the legacy MAX_PATH
    # boundary even though the deliberately mapped cache path is short enough.
    capture_dir = Path(os.path.abspath(str(capture_dir)))

    def patched_install(
        cache: Any,
        *,
        out_dir: Path,
        identity_sha256: str,
    ) -> None:
        original_install(
            cache,
            out_dir=out_dir,
            identity_sha256=identity_sha256,
        )
        if getattr(cache, "_neuroclaw_codex_shadow_capture", False):
            return
        resolved_out = Path(os.path.abspath(str(out_dir)))
        if capture_dir != resolved_out and resolved_out not in capture_dir.parents:
            raise RuntimeError(
                "Codex-session shadow capture directory must be inside the trial"
            )

        original_get = cache.get
        capture_lock = threading.Lock()

        def shadow_get(
            prompt: str,
            model_name: str,
            temperature: float,
            max_tokens: int,
            tools: Any = None,
            json_schema: Any = None,
            force_json: Any = None,
        ) -> Any:
            result = original_get(
                prompt,
                model_name,
                temperature,
                max_tokens,
                tools,
                json_schema,
                force_json,
            )
            if result is not None:
                return result
            request_sha256 = cache._generate_cache_key(
                prompt,
                model_name,
                temperature,
                max_tokens,
                tools,
                json_schema,
                force_json,
            )
            pending_path = capture_dir / "pending" / f"{request_sha256}.json"
            with capture_lock:
                if not pending_path.is_file():
                    adapter_module._write_json_atomic(
                        pending_path,
                        {
                            "schema_version": CAPTURE_SCHEMA,
                            "status": "pending",
                            "scientific_classification": (
                                "mixed_backend_user_approved_pending_import"
                            ),
                            "created_at": time.time(),
                            "request_sha256": request_sha256,
                            "cache_identity_sha256": identity_sha256,
                            "formal_cache_written": False,
                            "requested_backend": {
                                "model": backend_model,
                                "reasoning_effort": reasoning_effort,
                                "thread_id": thread_id,
                            },
                            "request": {
                                "prompt": prompt,
                                "prompt_sha256": hashlib.sha256(
                                    prompt.encode("utf-8")
                                ).hexdigest(),
                                "model": model_name,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "tools": tools,
                                "json_schema": json_schema,
                                "force_json": force_json,
                            },
                        },
                    )
            raise RuntimeError(
                f"Codex-session shadow request captured: {request_sha256}"
            )

        cache.get = shadow_get
        cache._neuroclaw_codex_shadow_capture = True

    adapter_module._install_open_coscientist_cache_audit = patched_install


def main() -> None:
    adapter_module = _load_adapter_module()
    install_shadow_capture(
        adapter_module,
        capture_dir=Path(_required_environment("CS1_CODEX_SESSION_CAPTURE_DIR")),
        backend_model=_required_environment("CS1_CODEX_SESSION_MODEL"),
        reasoning_effort=_required_environment(
            "CS1_CODEX_SESSION_REASONING_EFFORT"
        ),
        thread_id=_required_environment("CS1_CODEX_SESSION_THREAD_ID"),
    )
    adapter_module.main()


if __name__ == "__main__":
    main()
