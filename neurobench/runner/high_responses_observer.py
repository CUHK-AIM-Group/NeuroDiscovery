"""Loopback-only, single-attempt native Responses evidence recorder.

The issued application credential stays in memory. Request bodies and native
events are recorded, headers are never written. No retries or body rewriting.
"""
from __future__ import annotations

import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from neurobench.runner.artifacts import EventLog, freeze_text, json_text


class ResponsesObserver:
    def __init__(self, directory: Path, token: str, deadline: float, *,
                 upstream_url: str = 'https://api.openai.com/v1/responses', upstream_token: str | None = None):
        if upstream_url not in ('https://api.openai.com/v1/responses', 'https://api.xnova.online/v1/responses'):
            raise ValueError('unapproved_responses_upstream')
        self.upstream_url = upstream_url
        self.token, self.deadline = token, deadline
        self.upstream_token = upstream_token or token
        self.log = EventLog(directory / 'native_responses_events.jsonl', secrets=[token, self.upstream_token])
        self.directory = directory
        self.lock = threading.Lock()
        self.requests, self.completed, self.errors = [], [], []
        self.client = httpx.Client(trust_env=upstream_url == 'https://api.openai.com/v1/responses',
                                  limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *_):
                pass

            def do_POST(self):
                # Consume bounded request bytes before rejecting credentials.
                # Closing an unread Windows TCP body can reset the connection
                # before the caller receives the HTTP rejection.
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    self.send_error(400)
                    return
                if not 0 < size <= 32_000_000:
                    self.send_error(413)
                    return
                self.connection.settimeout(max(1, min(30, owner.deadline - time.monotonic())))
                raw = self.rfile.read(size)
                if len(raw) != size:
                    self.send_error(400)
                    return
                # The relay cannot serve other providers, files, or arbitrary URLs.
                if self.path != '/v1/responses':
                    self.send_error(404)
                    return
                if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + owner.token):
                    self.send_error(401)
                    return
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    self.send_error(400)
                    return
                if not isinstance(body, dict):
                    self.send_error(400)
                    return
                if body.get('model') != 'gpt-5.6-sol' or (body.get('reasoning') or {}).get('effort') != 'high':
                    owner.record('request_setting_rejected', model=body.get('model'), reasoning=body.get('reasoning'))
                    self.send_error(400)
                    return
                remaining = owner.deadline - time.monotonic()
                if remaining <= 0:
                    self.send_error(408)
                    return
                with owner.lock:
                    index = len(owner.requests) + 1
                    owner.requests.append({'model': body['model'], 'reasoning': body['reasoning']})
                owner.record('model_request', request_index=index, endpoint=owner.upstream_url, body=body)
                sent_headers = False
                try:
                    with owner.client.stream('POST', owner.upstream_url,
                            headers={'Authorization': 'Bearer ' + owner.upstream_token, 'Content-Type': 'application/json', 'Accept-Encoding': 'identity'},
                            content=raw, timeout=httpx.Timeout(remaining, connect=min(30, remaining))) as response:
                        owner.record('http_response', request_index=index, http_status=response.status_code)
                        self.send_response(response.status_code)
                        self.send_header('Content-Type', response.headers.get('content-type', 'text/event-stream'))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        sent_headers = True
                        if response.status_code != 200:
                            data = response.read()
                            owner.record('native_http_error', request_index=index, http_status=response.status_code,
                                         body=data.decode('utf-8', 'replace'), retry_after=response.headers.get('retry-after'))
                            self.wfile.write(data)
                            return
                        for line in response.iter_lines():
                            if time.monotonic() >= owner.deadline:
                                raise TimeoutError('whole_task_deadline_reached')
                            if line.startswith('data:') and line[5:].strip() != '[DONE]':
                                event = json.loads(line[5:].strip())
                                owner.record('native_stream_event', request_index=index, payload=event)
                                if event.get('type') == 'response.completed':
                                    with owner.lock:
                                        owner.completed.append(event['response'])
                            wire = (line + '\n').encode('utf-8')
                            self.wfile.write(wire)
                            self.wfile.flush()
                        owner.record('stream_closed', request_index=index)
                except Exception as error:
                    with owner.lock:
                        owner.errors.append(type(error).__name__)
                    owner.record('transport_error', request_index=index, error_type=type(error).__name__)
                    if not sent_headers:
                        self.send_error(502, 'upstream transport failed; remote outcome may be unknown')
                finally:
                    self.close_connection = True

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True)

    def record(self, event: str, **payload):
        with self.lock:
            self.log.write(event, **payload)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.client.close()
        self.log.close()

    def summary(self):
        models = sorted({r.get('model') for r in self.completed if r.get('model')})
        high = bool(self.completed) and all((r.get('reasoning') or {}).get('effort') == 'high' and
                                          r.get('model') == 'gpt-5.6-sol' for r in self.completed)
        value = {'native_request_count': len(self.requests), 'native_completed_response_count': len(self.completed),
                 'endpoint': self.upstream_url,
                 'official_openai_endpoint': self.upstream_url == 'https://api.openai.com/v1/responses',
                 'native_model_ids': models, 'native_high_echo': high, 'transport_errors': self.errors,
                 'all_requests_completed': len(self.requests) == len(self.completed) and not self.errors,
                 'header_values_logged': False, 'request_body_rewritten': False, 'relay_retries': 0}
        freeze_text(self.directory / 'NATIVE_WIRE_SUMMARY.json', json_text(value))
        return value
