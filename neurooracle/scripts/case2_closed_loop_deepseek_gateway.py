"""Fixed DeepSeek V4 Pro gateway for the Case 2 closed-loop pilot."""

from __future__ import annotations

import argparse
from http import HTTPStatus
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.deepseek_v4_pro_gateway import GatewayHandler, GatewayServer
from neurooracle.scripts.deepseek_v4_pro_router import (
    DEFAULT_KEYS,
    OLLAMA_CLOUD_ENDPOINT,
    RETRY_DELAYS_SECONDS,
    DeepSeekV4ProRouter,
    Route,
    Transport,
    urllib_transport,
)
from neurooracle.scripts.case2_closed_loop_credentials import (
    AUTOMATIC_ROUTE,
    load_channel_keys,
)


BENCHMARK_ID = "case2_adni_closed_loop_benchmark_v2_seed1"
FORCED_REASONING_EFFORT = "high"
FORCED_TEMPERATURE = 0.0
FORCED_MAX_OUTPUT_TOKENS = 8192


class Case2Router(DeepSeekV4ProRouter):
    """Finite Ollama-only router for the resumed Case 2 execution."""

    def __init__(
        self,
        *,
        keys_path: Path = DEFAULT_KEYS,
        checkpoint_path: Path | None = None,
        timeout_seconds: float = 300.0,
        retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
        allow_official_deepseek: bool = False,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        is_quota_exhausted: Callable[[str], bool] | None = None,
        mark_quota_exhausted: Callable[[str], None] | None = None,
        clear_quota_exhausted: Callable[[str], None] | None = None,
    ) -> None:
        if allow_official_deepseek:
            raise ValueError("DeepSeek official is disabled for the Case 2 automatic route")
        keys = load_channel_keys(keys_path)
        self._keys = keys
        self._all_secrets = tuple(secret for values in keys.values() for secret in values)
        self.checkpoint_path = checkpoint_path
        self.timeout_seconds = float(timeout_seconds)
        self.retry_delays = tuple(float(value) for value in retry_delays)
        self.allow_official_deepseek = False
        self.transport = transport
        self.sleep = sleep
        self.is_quota_exhausted = is_quota_exhausted or (lambda _key_label: False)
        self.mark_quota_exhausted = mark_quota_exhausted or (lambda _key_label: None)
        self.clear_quota_exhausted = clear_quota_exhausted or (lambda _key_label: None)
        self._checkpoint_lock = threading.Lock()

    def routes(self) -> tuple[Route, ...]:
        ollama = tuple(
            Route(
                channel="ollama_cloud",
                key_group="ollama",
                key_index=index,
                key_label=f"ollama_cloud_key_{index + 1}",
                pass_number=1,
                endpoint=OLLAMA_CLOUD_ENDPOINT,
                model="deepseek-v4-pro:cloud",
            )
            for index in range(2)
        )
        return ollama


class FixedRouter:
    def __init__(self, router: DeepSeekV4ProRouter, request_lock: threading.Lock):
        self._router = router
        self._request_lock = request_lock

    def complete(self, **kwargs: Any) -> Any:
        kwargs["reasoning_effort"] = FORCED_REASONING_EFFORT
        kwargs["temperature"] = FORCED_TEMPERATURE
        kwargs["max_output_tokens"] = FORCED_MAX_OUTPUT_TOKENS
        with self._request_lock:
            return self._router.complete(**kwargs)


class RouterPauseLatch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checkpoint: str | None = None

    def mark(self, checkpoint: str | None) -> None:
        with self._lock:
            if self._checkpoint is None:
                self._checkpoint = str(checkpoint or "checkpointed_router_pause")

    def checkpoint(self) -> str | None:
        with self._lock:
            return self._checkpoint


class Handler(GatewayHandler):
    server: "Server"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        # The shared gateway keeps its historical 503 behavior because it is
        # part of the preserved v1 freeze.  Only this v2 endpoint remaps a
        # safely checkpointed router pause to non-retryable 422, preventing
        # framework backoff from starting a fresh logical request.
        error = payload.get("error")
        if (
            status == HTTPStatus.SERVICE_UNAVAILABLE
            and isinstance(error, dict)
            and error.get("type") == "router_paused"
        ):
            self.server.mark_router_paused(str(error.get("checkpoint") or ""))
            status = HTTPStatus.UNPROCESSABLE_ENTITY
        super()._send_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        checkpoint = self.server.router_pause_checkpoint()
        if checkpoint is not None:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {
                    "error": {
                        "type": "router_paused",
                        "message": "finite Case 2 router is paused; explicit restart required",
                        "checkpoint": checkpoint,
                    }
                },
            )
            return
        super().do_POST()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health", "/v1/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "benchmark_id": BENCHMARK_ID,
                    "model": "deepseek-v4-pro",
                    "forced_reasoning_effort": FORCED_REASONING_EFFORT,
                    "forced_temperature": FORCED_TEMPERATURE,
                    "forced_max_output_tokens": FORCED_MAX_OUTPUT_TOKENS,
                    "max_concurrent_upstream_requests": 1,
                    "route_count": len(AUTOMATIC_ROUTE),
                    "ollama_cloud_key_count": 2,
                    "automatic_route": list(AUTOMATIC_ROUTE),
                    "execution_channel": "ollama_cloud",
                    "opencode_go_enabled": False,
                    "official_deepseek_enabled": False,
                    "router_pause_latch": True,
                    "router_paused": self.server.router_pause_checkpoint() is not None,
                    "secrets_exposed": False,
                },
            )
            return
        super().do_GET()


class Server(GatewayServer):
    def __init__(self, address: tuple[str, int], *, keys: Path, checkpoints: Path, timeout: float):
        super().__init__(
            address,
            keys_path=keys,
            checkpoint_dir=checkpoints,
            timeout_seconds=timeout,
            allow_official_deepseek=False,
        )
        self.RequestHandlerClass = Handler
        self._request_lock = threading.Lock()
        self._router_pause_latch = RouterPauseLatch()

    def mark_router_paused(self, checkpoint: str | None) -> None:
        self._router_pause_latch.mark(checkpoint)

    def router_pause_checkpoint(self) -> str | None:
        return self._router_pause_latch.checkpoint()

    def router(self, request_id: str) -> FixedRouter:
        router = Case2Router(
            keys_path=self.keys_path,
            checkpoint_path=self.checkpoint_dir / f"{request_id}.json",
            timeout_seconds=self.timeout_seconds,
            allow_official_deepseek=False,
            is_quota_exhausted=self.is_quota_exhausted,
            mark_quota_exhausted=self.mark_quota_exhausted,
            clear_quota_exhausted=self.clear_quota_exhausted,
        )
        return FixedRouter(router, self._request_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Credential-bearing gateway may bind only to loopback")
    router = Case2Router(
        keys_path=args.keys.resolve(), checkpoint_path=None,
        timeout_seconds=args.timeout_seconds, allow_official_deepseek=False,
    )
    checkpoints = args.checkpoint_dir.resolve()
    checkpoints.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "ok" if args.preflight_only else "listening",
        "benchmark_id": BENCHMARK_ID,
        "base_url": f"http://{args.host}:{args.port}/v1",
        "model": "deepseek-v4-pro",
        "route_count": len(router.routes()),
        "ollama_cloud_key_count": 2,
        "automatic_route": list(AUTOMATIC_ROUTE),
        "execution_channel": "ollama_cloud",
        "opencode_go_enabled": False,
        "forced_reasoning_effort": FORCED_REASONING_EFFORT,
        "forced_temperature": FORCED_TEMPERATURE,
        "forced_max_output_tokens": FORCED_MAX_OUTPUT_TOKENS,
        "max_concurrent_upstream_requests": 1,
        "official_deepseek_enabled": False,
        "router_pause_latch": True,
        "router_paused": False,
        "secrets_printed_or_persisted": False,
    }
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    server = Server(
        (args.host, args.port), keys=args.keys.resolve(),
        checkpoints=checkpoints, timeout=args.timeout_seconds,
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
