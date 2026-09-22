"""Case-2-specific fixed-setting gateway over the finite DeepSeek V4 Pro router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from typing import Any, Mapping

from neurooracle.scripts.deepseek_v4_pro_gateway import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    GatewayHandler,
    GatewayServer,
)
from neurooracle.scripts.deepseek_v4_pro_router import (
    DEFAULT_KEYS,
    DeepSeekV4ProRouter,
)


FORCED_REASONING_EFFORT = "high"
FORCED_TEMPERATURE = 0.0
FORCED_MAX_OUTPUT_TOKENS = 8192


class _FixedRouter:
    def __init__(self, router: DeepSeekV4ProRouter, request_lock: threading.Lock):
        self._router = router
        self._request_lock = request_lock

    def complete(self, **kwargs: Any) -> Any:
        kwargs["reasoning_effort"] = FORCED_REASONING_EFFORT
        kwargs["temperature"] = FORCED_TEMPERATURE
        kwargs["max_output_tokens"] = FORCED_MAX_OUTPUT_TOKENS
        with self._request_lock:
            return self._router.complete(**kwargs)


class Case2GatewayHandler(GatewayHandler):
    server: "Case2GatewayServer"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in {"", "/health", "/v1/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "benchmark_id": "case2_adni_formal_benchmark_v1",
                    "model": "deepseek-v4-pro",
                    "forced_reasoning_effort": FORCED_REASONING_EFFORT,
                    "forced_temperature": FORCED_TEMPERATURE,
                    "forced_max_output_tokens": FORCED_MAX_OUTPUT_TOKENS,
                    "max_concurrent_upstream_requests": 1,
                    "automatic_route": [
                        "opencode_go_key_1",
                        "opencode_go_key_2",
                        "opencode_go_key_3",
                        "ollama_cloud",
                        "opencode_go_final_pass_once",
                    ],
                    "official_deepseek_enabled": False,
                    "secrets_exposed": False,
                },
            )
            return
        super().do_GET()


class Case2GatewayServer(GatewayServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        keys_path: Path,
        checkpoint_dir: Path,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            address,
            keys_path=keys_path,
            checkpoint_dir=checkpoint_dir,
            timeout_seconds=timeout_seconds,
            allow_official_deepseek=False,
        )
        self.RequestHandlerClass = Case2GatewayHandler
        self._upstream_request_lock = threading.Lock()

    def router(self, request_id: str) -> _FixedRouter:
        router = DeepSeekV4ProRouter(
            keys_path=self.keys_path,
            checkpoint_path=self.checkpoint_dir / f"{request_id}.json",
            timeout_seconds=self.timeout_seconds,
            allow_official_deepseek=False,
            is_quota_exhausted=self.is_quota_exhausted,
            mark_quota_exhausted=self.mark_quota_exhausted,
            clear_quota_exhausted=self.clear_quota_exhausted,
        )
        return _FixedRouter(router, self._upstream_request_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the credential-bearing gateway may bind only to loopback")
    router = DeepSeekV4ProRouter(
        keys_path=args.keys.resolve(),
        checkpoint_path=None,
        timeout_seconds=args.timeout_seconds,
        allow_official_deepseek=False,
    )
    args.checkpoint_dir.resolve().mkdir(parents=True, exist_ok=True)
    safe_status: Mapping[str, Any] = {
        "status": "ok" if args.preflight_only else "listening",
        "benchmark_id": "case2_adni_formal_benchmark_v1",
        "base_url": f"http://{args.host}:{args.port}/v1",
        "model": "deepseek-v4-pro",
        "route_count": len(router.routes()),
        "forced_reasoning_effort": FORCED_REASONING_EFFORT,
        "forced_temperature": FORCED_TEMPERATURE,
        "forced_max_output_tokens": FORCED_MAX_OUTPUT_TOKENS,
        "max_concurrent_upstream_requests": 1,
        "official_deepseek_enabled": False,
        "secrets_printed_or_persisted": False,
    }
    print(json.dumps(dict(safe_status), ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    server = Case2GatewayServer(
        (args.host, args.port),
        keys_path=args.keys.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        timeout_seconds=args.timeout_seconds,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
