"""Durable, fail-closed accounting for explicitly authorized future experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import time


class BudgetExceeded(RuntimeError):
    pass


class RunGuard:
    def __init__(self, path: Path, authorization: dict):
        self.path = Path(path)
        self.authorization = authorization
        for field in ("authorization_id", "max_runs", "max_total_seconds", "max_run_seconds",
                      "max_gpu_bytes", "max_cpu_bytes", "prior_runs"):
            if field not in authorization:
                raise ValueError(f"missing {field}")
        for field in ("max_runs", "max_gpu_bytes", "max_cpu_bytes", "prior_runs"):
            if type(authorization[field]) is not int or authorization[field] < 0:
                raise ValueError(f"invalid {field}")
        for field in ("max_total_seconds", "max_run_seconds"):
            if isinstance(authorization[field], bool) or not isinstance(authorization[field], (int, float)) or not math.isfinite(authorization[field]) or authorization[field] <= 0:
                raise ValueError(f"invalid {field}")
        binding = json.dumps(authorization, sort_keys=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, key TEXT UNIQUE, state TEXT, reserved REAL, elapsed REAL)")
            connection.execute("INSERT OR IGNORE INTO binding VALUES (1, ?)", (binding,))
            if connection.execute("SELECT payload FROM binding WHERE id=1").fetchone()[0] != binding:
                raise BudgetExceeded("authorization is immutable; cannot reset or increase budget")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def reserve(self, attempt_key: str, seconds: float) -> int:
        if not attempt_key or not math.isfinite(seconds) or not 0 < seconds <= self.authorization["max_run_seconds"]:
            raise BudgetExceeded("invalid per-run reservation")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT state,reserved FROM attempts").fetchall()
            if self.authorization["prior_runs"] + len(rows) >= self.authorization["max_runs"]:
                raise BudgetExceeded("run limit exhausted; failed attempts also consume a slot")
            if any(state != "COMPLETE" for state, _ in rows):
                raise BudgetExceeded("pending, failed or unknown attempt requires review")
            if sum(reserved for _, reserved in rows) + seconds > self.authorization["max_total_seconds"]:
                raise BudgetExceeded("total reserved time exhausted")
            try:
                cursor = connection.execute(
                    "INSERT INTO attempts(key,state,reserved) VALUES (?, 'RESERVED', ?)",
                    (attempt_key, seconds),
                )
            except sqlite3.IntegrityError as exc:
                raise BudgetExceeded("attempt already reserved; no silent resend") from exc
            return cursor.lastrowid

    def finish(self, attempt_id: int, elapsed: float, *, success: bool) -> None:
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid elapsed time")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state,reserved FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row[0] != "RESERVED":
                raise BudgetExceeded("attempt missing or already terminal")
            state = "COMPLETE" if success and elapsed <= row[1] else "HELD"
            connection.execute("UPDATE attempts SET state=?,elapsed=? WHERE id=?", (state, elapsed, attempt_id))

    def claim_worker(self, attempt_id: int, attempt_key: str) -> float:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS worker_claims (attempt_id INTEGER PRIMARY KEY)")
            row = connection.execute("SELECT key,state,reserved FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row[:2] != (attempt_key, "RESERVED"):
                raise BudgetExceeded("worker requires matching durable reservation")
            try:
                connection.execute("INSERT INTO worker_claims VALUES (?)", (attempt_id,))
            except sqlite3.IntegrityError as exc:
                raise BudgetExceeded("worker already claimed; never restart an unknown attempt") from exc
            return row[2]

    def configure_cuda(self, torch_module, device: int = 0) -> None:
        total = torch_module.cuda.get_device_properties(device).total_memory
        cap = self.authorization["max_gpu_bytes"]
        if cap <= 0 or total <= 0:
            raise BudgetExceeded("GPU use not authorized or device unavailable")
        torch_module.cuda.set_per_process_memory_fraction(min(cap / total, 1.0), device)

    def check_usage(self, *, elapsed: float, gpu_bytes: int, cpu_bytes: int, reserved_seconds: float) -> None:
        readings = (elapsed, gpu_bytes, cpu_bytes, reserved_seconds)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in readings):
            raise BudgetExceeded("missing or invalid resource readings")
        if (reserved_seconds > self.authorization["max_run_seconds"] or elapsed >= reserved_seconds
                or gpu_bytes > self.authorization["max_gpu_bytes"]
                or cpu_bytes > self.authorization["max_cpu_bytes"]):
            raise BudgetExceeded("wall-time or memory ceiling reached")

    def supervise(self, process, attempt_id: int, *, clock=time.monotonic, started=None) -> int:
        """Monitor an owned psutil.Popen child; kill it on timeout or RSS overflow."""
        with self._connect() as connection:
            row = connection.execute("SELECT state,reserved FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None or row[0] != "RESERVED":
            raise BudgetExceeded("reserve before launching the worker")
        started = clock() if started is None else started
        try:
            while process.poll() is None:
                self.check_usage(elapsed=clock() - started, gpu_bytes=0,
                                 cpu_bytes=process.memory_info().rss, reserved_seconds=row[1])
                try:
                    process.wait(timeout=min(0.1, max(0.001, row[1] - (clock() - started))))
                except TimeoutError:
                    pass
                except Exception as exc:
                    if type(exc).__name__ not in {"TimeoutExpired"}:
                        raise
            elapsed = clock() - started
            success = process.returncode == 0 and elapsed <= row[1]
            self.finish(attempt_id, elapsed, success=success)
            if process.returncode == 0 and not success:
                raise BudgetExceeded("worker finished after its wall-time ceiling")
            return process.returncode
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            with self._connect() as connection:
                state = connection.execute("SELECT state FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if state and state[0] == "RESERVED":
                self.finish(attempt_id, max(0, clock() - started), success=False)
            raise
