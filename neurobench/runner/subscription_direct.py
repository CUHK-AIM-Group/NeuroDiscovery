"""ChatGPT-plan refresh and native Responses parsing for the new experiment.

The independently verified OAuth registration helper stays immutable. It owns
NeuroClaw's registration only, never Codex's credentials. No inference on import.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from functools import lru_cache
from pathlib import Path

AUTH_SOURCE_SHA256 = 'ba282e637c09dbabb9c5078e0a9909b10a0239cf83972be673da49e98a9bd3e7'
ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def auth_helper():
    path = ROOT / 'tmp/chatgpt_plan_direct_20261009/chatgpt_plan.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUTH_SOURCE_SHA256:
        raise RuntimeError('frozen_neuroclaw_oauth_source_changed')
    spec = importlib.util.spec_from_file_location('neuroclaw_verified_plan_oauth', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_access_record(*, minimum_valid_seconds=60, root=None, client=None):
    """One serialized rotating refresh; ambiguous exchanges are never repeated."""
    auth = auth_helper()
    root = Path(root) if root else auth.credential_root()
    with auth.session_lock(root):
        previous = auth.active_record(root)
        if not previous:
            raise auth.PlanError('neuroclaw_subscription_login_required')
        if previous.get('issuer') != auth.ISSUER or not previous.get('client_id', '').startswith('oaiapp_'):
            raise auth.PlanError('registration_identity_invalid')
        if not {auth.DIRECT_SCOPE, 'resource.invoke'} <= set(previous.get('scopes', [])):
            raise auth.PlanError('chatgpt_plan_permission_not_granted')
        journal = root / 'refresh_transaction.json'
        if journal.exists():
            pending = json.loads(journal.read_text(encoding='utf-8'))
            if pending.get('state') in ('ALLOCATED', 'UNKNOWN'):
                raise auth.PlanError('refresh_remote_unknown_reauthorization_required')
        if previous['expires_at'] > time.time() + minimum_valid_seconds:
            return previous
        owned = client is None
        client = client or auth.http_client()
        try:
            config = auth.discovery(client)
            jwks = auth.checked_json(client.get(config['jwks_uri']), 'refresh_jwks')
            # Keep the exchange reservation before dispatch, without token data.
            auth.atomic_json(journal, {'state': 'ALLOCATED', 'started_at': auth.utc(), 'inference_requests': 0}, private=True)
            try:
                response = client.post(auth.TOKEN, data={
                    'grant_type': 'refresh_token', 'client_id': previous['client_id'],
                    'refresh_token': previous['refresh_token'], 'resource': auth.RESOURCE,
                })
            except Exception as exc:
                auth.atomic_json(journal, {'state': 'UNKNOWN', 'error_type': type(exc).__name__, 'inference_requests': 0}, private=True)
                raise auth.PlanError('refresh_transport_remote_unknown_reauthorization_required') from None
            if response.status_code != 200:
                auth.atomic_json(journal, {'state': 'FAILED_KNOWN_HTTP', 'http_status': response.status_code, 'inference_requests': 0}, private=True)
                raise auth.PlanError('refresh_http_' + str(response.status_code))
            try:
                tokens = response.json()
                # OAuth refresh may omit an unchanged grant scope. No escalation.
                if 'scope' not in tokens:
                    tokens['scope'] = ' '.join(previous['scopes'])
                new = auth.registration(tokens, jwks, previous['client_id'], previous['ext_agent_host_id'], None, previous)
                if not set(new['scopes']) <= set(previous['scopes']):
                    raise auth.PlanError('refresh_grant_scope_changed')
                if new['expires_at'] <= time.time() + minimum_valid_seconds:
                    raise auth.PlanError('refresh_lifetime_too_short')
                auth.save_record(root, new)
                auth.atomic_json(journal, {'state': 'COMPLETE', 'finished_at': auth.utc(), 'identity_signature_verified': True, 'inference_requests': 0}, private=True)
                return new
            except Exception as exc:
                # A successful token response may have rotated the old grant.
                auth.atomic_json(journal, {'state': 'UNKNOWN', 'error_type': type(exc).__name__, 'inference_requests': 0}, private=True)
                raise auth.PlanError('refresh_response_validation_failed_reauthorization_required') from None
        finally:
            if owned:
                client.close()


def visible_models():
    auth = auth_helper()
    record = get_access_record()
    with auth.http_client() as client:
        response = client.get(auth.RESOURCE + '/models', headers={'Authorization': 'Bearer ' + record['access_token']})
        data = auth.checked_json(response, 'model_catalog')
    if not isinstance(data.get('models'), list):
        raise auth.PlanError('subscription_model_catalog_contract_mismatch')
    return {'observed_at': auth.utc(), 'http_status': 200, 'response_sha256': hashlib.sha256(response.content).hexdigest(),
            'models': [{'slug': m.get('slug'), 'display_name': m.get('display_name')}
                       for m in data['models'] if isinstance(m, dict) and m.get('visibility') == 'list'],
            'inference_requests': 0}


def response_body(model, history, function_schemas):
    request = {'model': model, 'input': history, 'store': False, 'stream': True, 'reasoning': {'effort': 'high'},
               'include': ['reasoning.encrypted_content']}
    if function_schemas:
        request['tools'] = [{'type': 'namespace', 'name': 'neuro_runtime',
                             'description': 'Local task tools. Tool output is logged; validation is not workflow execution.',
                             'tools': [dict(type='function', **s) for s in function_schemas]}]
    return request


def terminal_response(events):
    """Validate one completed terminal without loading the frozen OAuth helper."""
    terminal = None
    for event in events:
        kind = event.get("type")
        if kind in ("response.completed", "response.failed", "response.incomplete", "error"):
            if terminal is not None:
                raise ValueError("multiple_response_terminal_events")
            terminal = event
    if terminal is None:
        raise ValueError("response_stream_ended_without_terminal_event_remote_unknown")
    if terminal["type"] != "response.completed":
        raise ValueError("response_not_completed_" + terminal["type"].replace(".", "_"))
    response = terminal.get("response") or {}
    if response.get("status") != "completed":
        raise ValueError("response_terminal_status_mismatch")
    if response.get("error") or response.get("incomplete_details"):
        raise ValueError("completed_response_contains_failure_details")
    return response


def completed_response(events):
    """Complete text OR function calls, preserving stateless reasoning items."""
    events = list(events)
    response = terminal_response(events)
    done = {}
    delta, text_done, arg_delta, arg_done = {}, {}, {}, {}
    for e in events:
        typ = e.get('type')
        if typ == 'response.output_item.done':
            index, item = e.get('output_index'), e.get('item') or {}
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in done:
                raise ValueError('output_item_identity_invalid')
            if item.get('status') not in (None, 'completed'):
                raise ValueError('completed_response_unfinished_item')
            done[index] = item
        elif typ in ('response.output_text.delta', 'response.output_text.done'):
            key = (e.get('item_id'), e.get('output_index'), e.get('content_index'))
            if typ.endswith('.delta'):
                delta.setdefault(key, []).append(e.get('delta', ''))
            else:
                if key in text_done:
                    raise ValueError('duplicate_text_done')
                text_done[key] = e.get('text', '')
        elif typ in ('response.function_call_arguments.delta', 'response.function_call_arguments.done'):
            key = (e.get('item_id'), e.get('output_index'))
            if typ.endswith('.delta'):
                arg_delta.setdefault(key, []).append(e.get('delta', ''))
            else:
                if key in arg_done:
                    raise ValueError('duplicate_arguments_done')
                arg_done[key] = e.get('arguments', '')
    for index, item in done.items():
        for part_index, part in enumerate(item.get('content', [])):
            if part.get('type') != 'output_text':
                continue
            key = (item.get('id'), index, part_index)
            if key in delta and ''.join(delta[key]) != part['text']:
                raise ValueError('text_delta_item_mismatch')
            if key in text_done and text_done[key] != part['text']:
                raise ValueError('text_done_item_mismatch')
        if item.get('type') == 'function_call':
            key = (item.get('id'), index)
            if key in arg_delta and ''.join(arg_delta[key]) != item['arguments']:
                raise ValueError('function_arguments_delta_mismatch')
            if key in arg_done and arg_done[key] != item['arguments']:
                raise ValueError('function_arguments_done_mismatch')
    streamed = [done[k] for k in sorted(done)]
    terminal = response.get('output') or []
    def identities(items):
        return [(i.get('id'), i.get('type'), i.get('name'), i.get('arguments'), i.get('content')) for i in items]
    if terminal and streamed and identities(terminal) != identities(streamed):
        raise ValueError('terminal_stream_item_mismatch')
    items = terminal or streamed
    calls = [i for i in items if i.get('type') == 'function_call']
    if len({i.get('call_id') for i in calls}) != len(calls) or any(not i.get('call_id') for i in calls):
        raise ValueError('function_call_identity_invalid')
    for c in calls:
        if c.get('namespace') not in (None, 'neuro_runtime'):
            raise ValueError('unexpected_function_namespace')
        if not isinstance(json.loads(c.get('arguments', '{}')), dict):
            raise ValueError('function_arguments_not_object')
    text = ''.join(c.get('text', '') for i in items if i.get('type') == 'message' and i.get('role') == 'assistant'
                   for c in i.get('content', []) if c.get('type') == 'output_text')
    if not calls and not text.strip():
        raise ValueError('completed_response_without_text_or_functions')
    return {'response': response, 'output': items, 'text': text, 'calls': calls}
