"""Bounded, read-only experiment status-file checks; never launch or repair jobs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

from core.runtime_store import RuntimeStore


class ScheduledChecks:
    def __init__(self, path: Path):
        self.store = RuntimeStore(path)
        with self.store.connect() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS scheduled_checks (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, workspace TEXT NOT NULL,
                relative_path TEXT NOT NULL, interval_seconds INTEGER NOT NULL,
                max_checks INTEGER NOT NULL, checks INTEGER NOT NULL, deadline REAL NOT NULL,
                next_check REAL NOT NULL, status TEXT NOT NULL, result TEXT NOT NULL)''')

    def pause_on_restart(self):
        with self.store.connect() as connection:
            connection.execute("UPDATE scheduled_checks SET status='paused', result=? WHERE status IN ('active','checking')",
                               (json.dumps({'reason': 'Backend restarted; explicitly resume after inspection'}),))

    def create(self, chat_id: str, workspace: Path, relative_path: str, interval: int = 10, max_checks: int = 24, hours: int = 24) -> dict:
        if not chat_id or len(chat_id) > 200:
            raise ValueError('Invalid conversation')
        if type(interval) is not int or not 1 <= interval <= 1440:
            raise ValueError('Interval must be 1–1440 minutes')
        if type(max_checks) is not int or not 1 <= max_checks <= 144:
            raise ValueError('Check limit must be 1–144')
        if type(hours) is not int or not 1 <= hours <= 24:
            raise ValueError('Duration must be 1–24 hours')
        relative = Path(relative_path)
        workspace = workspace.resolve()
        if not relative_path.strip() or relative.is_absolute() or relative.drive or ':' in relative_path or '..' in relative.parts:
            raise ValueError('Choose a status file relative to the workspace')
        if not (workspace / relative).resolve().is_relative_to(workspace):
            raise ValueError('Status file must remain in the workspace')
        now = time.time()
        task_id = uuid.uuid4().hex
        with self.store.connect() as connection:
            connection.execute('INSERT INTO scheduled_checks VALUES(?,?,?,?,?,?,0,?,?,?,?)',
                               (task_id, chat_id, str(workspace), str(relative), interval * 60, max_checks,
                                now + hours * 3600, now + interval * 60, 'active', '{}'))
        return next(item for item in self.list(chat_id) if item['id'] == task_id)

    def list(self, chat_id: str) -> list[dict]:
        with self.store.connect() as connection:
            rows = [dict(row) for row in connection.execute('SELECT * FROM scheduled_checks WHERE chat_id=? ORDER BY rowid DESC', (chat_id,))]
        return [{**row, 'result': json.loads(row['result']), 'max_repairs': 0} for row in rows]

    def control(self, task_id: str, chat_id: str, action: str):
        if action not in {'pause', 'resume', 'cancel'}:
            raise ValueError('Unsupported task action')
        with self.store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT * FROM scheduled_checks WHERE id=? AND chat_id=?', (task_id, chat_id)).fetchone()
            if not row or row['status'] in {'completed', 'cancelled'}:
                raise ValueError('Task is missing or terminal')
            if action == 'resume' and (row['status'] != 'paused' or row['checks'] >= row['max_checks'] or time.time() >= row['deadline']):
                raise ValueError('Task cannot resume; paused state and remaining budget are required')
            status = {'pause': 'paused', 'resume': 'active', 'cancel': 'cancelled'}[action]
            connection.execute('UPDATE scheduled_checks SET status=?,next_check=? WHERE id=?',
                               (status, time.time() + row['interval_seconds'], task_id))

    def tick(self, now: float | None = None):
        now = time.time() if now is None else now
        with self.store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute("UPDATE scheduled_checks SET status='completed',result=? WHERE status='active' AND (deadline<=? OR checks>=max_checks)",
                               (json.dumps({'reason': 'Check budget ended; not a scientific completion claim'}), now))
            row = connection.execute("SELECT * FROM scheduled_checks WHERE status='active' AND next_check<=? ORDER BY next_check LIMIT 1", (now,)).fetchone()
            if not row:
                return
            connection.execute("UPDATE scheduled_checks SET status='checking',checks=checks+1 WHERE id=?", (row['id'],))
        status = 'active'
        try:
            workspace = Path(row['workspace']).resolve()
            target = (workspace / row['relative_path']).resolve()
            if not target.is_relative_to(workspace) or not target.is_file():
                raise ValueError('Status file is missing or outside the workspace')
            with target.open('rb') as handle:
                content = handle.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError('Status file exceeds 1 MB; choose a small status receipt')
            reported = None
            try:
                document = json.loads(content)
                reported = document.get('status') if isinstance(document, dict) else None
            except (ValueError, UnicodeError):
                pass
            if reported in ('completed', 'done'):
                status = 'completed'
            elif reported in ('failed', 'blocked', 'error', 'cancelled'):
                status = 'paused'
            result = {'checked_at': now, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                      'reported_status': reported if isinstance(reported, str) else None,
                      'scope': 'Status file observation only; not independent acceptance. No job started or repaired.'}
        except (OSError, ValueError) as exc:
            status, result = 'paused', {'checked_at': now, 'error': str(exc)}
        if row['checks'] + 1 >= row['max_checks'] and status == 'active':
            status = 'completed'
            result['reason'] = 'Check count limit reached; not a scientific completion claim'
        with self.store.connect() as connection:
            connection.execute("UPDATE scheduled_checks SET status=?,result=?,next_check=? WHERE id=? AND status='checking'",
                               (status, json.dumps(result, ensure_ascii=False), now + row['interval_seconds'], row['id']))
