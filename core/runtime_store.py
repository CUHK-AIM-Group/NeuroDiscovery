"""Transactional queue and write-ahead execution records for NeuroDiscovery."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time


class RuntimeStore:
    def __init__(self, path: Path):
        self.path = path
        with self.connect():
            pass

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runtime_queue (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT UNIQUE NOT NULL,
              chat_id TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
              owner TEXT, created REAL NOT NULL, result TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS runtime_one_chat
              ON runtime_queue(chat_id) WHERE status='running';
            CREATE TABLE IF NOT EXISTS runtime_steps (
              request_id TEXT NOT NULL, step_id TEXT NOT NULL, kind TEXT NOT NULL,
              status TEXT NOT NULL, input TEXT NOT NULL, output TEXT,
              PRIMARY KEY(request_id,step_id));
            CREATE TABLE IF NOT EXISTS runtime_checkpoints (
              request_id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS runtime_approvals (
              request_id TEXT NOT NULL, step_id TEXT NOT NULL, tool TEXT NOT NULL,
              arguments TEXT NOT NULL, status TEXT NOT NULL,
              PRIMARY KEY(request_id,step_id));
            CREATE TABLE IF NOT EXISTS runtime_owners (
              owner TEXT PRIMARY KEY, pid INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS runtime_context (
              chat_id TEXT PRIMARY KEY, through_sequence INTEGER NOT NULL, messages TEXT NOT NULL);
        """)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def enqueue(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(body.encode()) > 8 * 1024 * 1024:
            raise ValueError("Queued request exceeds 8 MB")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM runtime_queue WHERE request_id=?", (payload["request_id"],)).fetchone()
            if previous:
                if previous["payload"] != body:
                    raise ValueError("Request ID is already bound to different content")
                return dict(previous)
            first = connection.execute("SELECT payload FROM runtime_queue WHERE chat_id=? ORDER BY sequence LIMIT 1", (payload['chat_id'],)).fetchone()
            if first and json.loads(first['payload']).get('workspace_path') != payload.get('workspace_path'):
                raise ValueError("A queued conversation cannot change workspace; create a new conversation")
            connection.execute("INSERT INTO runtime_queue(request_id,chat_id,payload,status,created) VALUES(?,?,?,?,?)",
                               (payload["request_id"], payload["chat_id"], body, "queued", time.time()))
            return dict(connection.execute("SELECT * FROM runtime_queue WHERE request_id=?", (payload["request_id"],)).fetchone())

    def register_owner(self, owner: str) -> None:
        with self.connect() as connection:
            owners = list(connection.execute("SELECT * FROM runtime_owners"))
            connection.execute("INSERT INTO runtime_owners VALUES(?,?)", (owner, os.getpid()))
        for previous in owners:
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x1000, False, previous['pid'])
                if not handle:
                    if ctypes.get_last_error() == 87:
                        self.recover_owner(previous['owner'])
                    continue
                code = wintypes.DWORD()
                try:
                    if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                        self.recover_owner(previous['owner'])
                finally:
                    kernel.CloseHandle(handle)
                continue
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                self.recover_owner(previous['owner'])
            except PermissionError:
                pass

    def list(self, chat_id: str) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM runtime_queue WHERE chat_id=? ORDER BY sequence", (chat_id,))]

    def get(self, request_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runtime_queue WHERE request_id=?", (request_id,)).fetchone()
            return dict(row) if row else None

    def history_for(self, request_id: str) -> list[dict]:
        current = self.get(request_id)
        rows = self.list(current['chat_id'])
        by_id = {row['request_id']: row for row in rows}
        payloads = {key: json.loads(row['payload']) for key, row in by_id.items()}
        replaced = {body['recovery_from'] for body in payloads.values() if body.get('recovery_from')}
        def position(row):
            source = payloads[row['request_id']].get('recovery_from')
            return position(by_id[source]) if source in by_id else row['sequence']
        history = list(payloads[rows[0]['request_id']].get('history') or [])
        with self.connect() as connection:
            checkpoint = connection.execute('SELECT * FROM runtime_context WHERE chat_id=?', (current['chat_id'],)).fetchone()
        if checkpoint:
            history = json.loads(checkpoint['messages'])
        for earlier in sorted(rows, key=position):
            if checkpoint and earlier['sequence'] <= checkpoint['through_sequence']:
                continue
            if position(earlier) >= position(current):
                break
            if earlier['request_id'] in replaced or not earlier['result']:
                continue
            result = json.loads(earlier['result'])
            history.extend([{'role': 'user', 'content': payloads[earlier['request_id']]['message']},
                            {'role': 'assistant', 'content': result.get('content') or result.get('message') or ''}])
        return history

    def compact_chat(self, chat_id: str, workspace: Path, fallback: list[dict]) -> dict:
        from core.session.context import compact_context
        previous = self.list(chat_id)
        messages = self.history_for(previous[-1]['request_id']) if previous else fallback
        if previous and previous[-1]['result']:
            with self.connect() as connection:
                saved = connection.execute('SELECT through_sequence FROM runtime_context WHERE chat_id=?', (chat_id,)).fetchone()
            if not saved or saved['through_sequence'] < previous[-1]['sequence']:
                result = json.loads(previous[-1]['result'])
                messages += [{'role': 'user', 'content': json.loads(previous[-1]['payload'])['message']},
                             {'role': 'assistant', 'content': result.get('content') or result.get('message') or ''}]
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            rows = list(connection.execute('SELECT * FROM runtime_queue WHERE chat_id=? ORDER BY sequence', (chat_id,)))
            if [dict(row) for row in rows] != previous:
                raise ValueError('Conversation changed; retry compaction')
            if any(row['status'] in {'queued', 'paused', 'running', 'interrupted'} for row in rows):
                raise ValueError('Finish or dismiss pending work before compacting')
            if rows and Path(json.loads(rows[0]['payload'])['workspace_path']).resolve() != workspace.resolve():
                raise ValueError('Conversation workspace mismatch')
            result = compact_context(messages, workspace, budget=2048, force=True)
            if result and result.get('status') == 'compacted':
                connection.execute('INSERT OR REPLACE INTO runtime_context VALUES(?,?,?)', (chat_id, rows[-1]['sequence'] if rows else 0, json.dumps(messages, ensure_ascii=False)))
            return result or {'status': 'unchanged', 'reason': 'Not enough earlier context to compact'}

    def claim(self, chat_id: str, owner: str) -> dict | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM runtime_queue WHERE chat_id=? AND status IN ('running','paused','interrupted')", (chat_id,)).fetchone():
                return None
            row = connection.execute("SELECT * FROM runtime_queue WHERE chat_id=? AND status='queued' ORDER BY CASE WHEN json_extract(payload,'$.recovery_from') IS NOT NULL THEN 0 ELSE 1 END, sequence LIMIT 1", (chat_id,)).fetchone()
            if not row:
                return None
            connection.execute("UPDATE runtime_queue SET status='running',owner=? WHERE request_id=?", (owner, row["request_id"]))
            return dict(row)

    def finish(self, request_id: str, owner: str, result: dict, status: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT chat_id FROM runtime_queue WHERE request_id=? AND owner=? AND status='running'", (request_id, owner)).fetchone()
            if not row:
                raise ValueError("Queue ownership lost")
            connection.execute("UPDATE runtime_queue SET status=?,result=? WHERE request_id=?", (status, json.dumps(result, ensure_ascii=False), request_id))
            if status != "completed":
                connection.execute("UPDATE runtime_queue SET status='paused' WHERE chat_id=? AND status='queued'", (row["chat_id"],))

    def cancel(self, request_id: str, chat_id: str) -> bool:
        with self.connect() as connection:
            return connection.execute("UPDATE runtime_queue SET status='cancelled' WHERE request_id=? AND chat_id=? AND status IN ('queued','paused','interrupted')", (request_id, chat_id)).rowcount == 1

    def continue_queue(self, chat_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM runtime_queue WHERE chat_id=? AND status='interrupted'", (chat_id,)).fetchone():
                raise ValueError("Inspect and dismiss interrupted work before continuing; it will not be replayed")
            connection.execute("UPDATE runtime_queue SET status='queued' WHERE chat_id=? AND status='paused'", (chat_id,))

    def recover_owner(self, owner: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE runtime_queue SET status='interrupted' WHERE owner=? AND status='running'", (owner,))
            connection.execute("UPDATE runtime_queue SET status='paused' WHERE status='queued' AND chat_id IN (SELECT chat_id FROM runtime_queue WHERE owner=? AND status='interrupted')", (owner,))
            connection.execute("UPDATE runtime_approvals SET status='denied' WHERE status='pending' AND request_id IN (SELECT request_id FROM runtime_queue WHERE owner=?)", (owner,))
            connection.execute("DELETE FROM runtime_owners WHERE owner=?", (owner,))

    def checkpoint(self, request_id: str, chat_id: str, body: dict) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO runtime_checkpoints VALUES(?,?,?,?)", (request_id, chat_id, json.dumps(body, ensure_ascii=False), time.time()))

    def begin_step(self, request_id: str, step_id: str, kind: str, data: dict) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO runtime_steps VALUES(?,?,?,?,?,NULL)", (request_id, step_id, kind, "intent", json.dumps(data, ensure_ascii=False)))

    def end_step(self, request_id: str, step_id: str, output: dict, checkpoint: dict | None = None, chat_id: str = '') -> None:
        with self.connect() as connection:
            if connection.execute("UPDATE runtime_steps SET status='recorded',output=? WHERE request_id=? AND step_id=? AND status='intent'", (json.dumps(output, ensure_ascii=False), request_id, step_id)).rowcount != 1:
                raise ValueError("Step is not pending")
            if checkpoint is not None:
                connection.execute("INSERT OR REPLACE INTO runtime_checkpoints VALUES(?,?,?,?)", (request_id, chat_id, json.dumps(checkpoint, ensure_ascii=False), time.time()))

    def recovery(self, request_id: str) -> dict:
        with self.connect() as connection:
            checkpoint = connection.execute("SELECT * FROM runtime_checkpoints WHERE request_id=?", (request_id,)).fetchone()
            steps = [dict(row) for row in connection.execute("SELECT * FROM runtime_steps WHERE request_id=? ORDER BY rowid", (request_id,))]
        return {"checkpoint": json.loads(checkpoint["body"]) if checkpoint else None,
                "chat_id": checkpoint["chat_id"] if checkpoint else None, "steps": steps,
                "uncertain_steps": [row["step_id"] for row in steps if row["status"] == "intent"],
                "automatic_replay": False}

    def resumable(self, request_id: str, chat_id: str) -> dict:
        row = self.get(request_id)
        if not row or row['chat_id'] != chat_id or row['status'] != 'interrupted':
            raise ValueError('Only interrupted work in this conversation can be recovered')
        saved = self.recovery(request_id)
        if saved['uncertain_steps'] or not saved['checkpoint']:
            raise ValueError('Uncertain execution requires manual reconciliation; tools will not be replayed')
        pending = set()
        for message in saved['checkpoint']['messages']:
            pending.update(call['id'] for call in message.get('tool_calls') or [])
            if message.get('role') == 'tool':
                pending.discard(message.get('tool_call_id'))
        if pending:
            raise ValueError('Partial tool transaction requires manual reconciliation')
        messages = saved['checkpoint']['messages']
        if messages and messages[-1].get('role') == 'assistant':
            raise ValueError('A model response is already saved; inspect it rather than generating it again')
        research = saved['checkpoint'].get('autoresearch')
        if research and research.get('status') in {'completed', 'verifying', 'review_required'}:
            raise ValueError('Delivery or review was already recorded; inspect its receipt before explicit continuation')
        return saved['checkpoint']

    def enqueue_recovery(self, payload: dict, source_id: str, checkpoint: dict) -> dict:
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            existing = connection.execute('SELECT * FROM runtime_queue WHERE request_id=?', (payload['request_id'],)).fetchone()
            if existing:
                if json.loads(existing['payload']) == payload:
                    return dict(existing)
                raise ValueError('Recovery request ID is already bound to different content')
            source = connection.execute("SELECT status FROM runtime_queue WHERE request_id=? AND chat_id=?", (source_id, payload['chat_id'])).fetchone()
            if not source or source['status'] != 'interrupted':
                raise ValueError('Recovery already claimed or no longer interrupted')
            current = connection.execute('SELECT body FROM runtime_checkpoints WHERE request_id=?', (source_id,)).fetchone()
            if not current or json.loads(current['body']) != checkpoint:
                raise ValueError('Checkpoint changed; inspect again')
            if connection.execute("SELECT 1 FROM runtime_steps WHERE request_id=? AND status='intent'", (source_id,)).fetchone():
                raise ValueError('Uncertain side effect cannot be replayed')
            connection.execute("INSERT INTO runtime_queue(request_id,chat_id,payload,status,created) VALUES(?,?,?,'queued',?)", (payload['request_id'], payload['chat_id'], json.dumps(payload, sort_keys=True, ensure_ascii=False), time.time()))
            connection.execute("UPDATE runtime_queue SET status='cancelled' WHERE request_id=?", (source_id,))
            connection.execute("UPDATE runtime_queue SET status='queued' WHERE chat_id=? AND status='paused'", (payload['chat_id'],))
            return dict(connection.execute('SELECT * FROM runtime_queue WHERE request_id=?', (payload['request_id'],)).fetchone())

    def request_approval(self, request_id: str, step_id: str, tool: str, arguments: dict) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO runtime_approvals VALUES(?,?,?,?,?)", (request_id, step_id, tool, json.dumps(arguments, ensure_ascii=False), "pending"))

    def approval(self, request_id: str, step_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM runtime_approvals WHERE request_id=? AND step_id=?", (request_id, step_id)).fetchone()
            return row[0] if row else "missing"

    def decide(self, request_id: str, step_id: str, approved: bool) -> bool:
        with self.connect() as connection:
            return connection.execute("UPDATE runtime_approvals SET status=? WHERE request_id=? AND step_id=? AND status='pending'", ("approved" if approved else "denied", request_id, step_id)).rowcount == 1
