"""Streaming native channels for the separately frozen high comparison."""
from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path

import httpx

from neurobench.runner.artifacts import EventLog
from neurobench.runner.subscription_direct import auth_helper, completed_response, get_access_record, response_body

_CLIENTS = threading.local()


def credential(provider, *, ollama_slot=1):
    root = Path(os.environ.get('NEUROCLAW_BENCH_REPO') or Path(__file__).resolve().parents[2])
    if provider == 'nvidia':
        value = json.loads((root / '.cc_settings.json').read_text(encoding='utf-8'))['env'].get('ANTHROPIC_AUTH_TOKEN', '')
    elif provider == 'deepseek':
        rows = (Path('C:/Users/45846/OneDrive/keys/deepseek.txt')).read_text(encoding='utf-8-sig').splitlines()
        value = next((r.strip() for r in rows if r.strip().startswith('sk-')), '')
    elif provider == 'ollama':
        if ollama_slot not in (1, 3):
            raise ValueError('ollama_slot_not_authorized_for_this_benchmark')
        rows = [r.strip() for r in Path('C:/Users/45846/OneDrive/keys/ollama.txt').read_text(encoding='utf-8-sig').splitlines() if r.strip()]
        value = rows[ollama_slot - 1] if len(rows) >= ollama_slot else ''
    else:
        raise ValueError('provider_not_supported')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('credential_missing_' + provider)
    return value.strip()


def native_usage(provider, raw):
    if provider == 'anthropic':
        fields = ('input_tokens', 'output_tokens')
        cached = raw.get('cache_read_input_tokens')
        write = raw.get('cache_creation_input_tokens')
        input_tokens = raw.get('input_tokens', 0) + (cached or 0) + (write or 0)
        output_tokens = raw.get('output_tokens', 0)
    elif provider == 'ollama':
        fields = ('prompt_eval_count', 'eval_count')
        input_tokens, output_tokens = raw.get(fields[0], 0), raw.get(fields[1], 0)
        cached = write = None
    else:
        fields = ('input_tokens', 'output_tokens') if provider == 'responses' else ('prompt_tokens', 'completion_tokens')
        input_tokens, output_tokens = raw.get(fields[0], 0), raw.get(fields[1], 0)
        detail = raw.get('input_tokens_details', raw.get('prompt_tokens_details')) or {}
        cached = detail.get('cached_tokens')
        write = detail.get('cache_creation_tokens')
    for value in (input_tokens, output_tokens):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('invalid_native_token_counter')
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'cached_input_tokens': cached,
            'cache_write_tokens': write, 'total_tokens': input_tokens + output_tokens,
            'usage_complete': all(k in raw for k in fields),
            'reasoning_tokens': (raw.get('output_tokens_details') or raw.get('completion_tokens_details') or {}).get('reasoning_tokens')}


def _lines(response, deadline):
    for line in response.iter_lines():
        if time.monotonic() > deadline:
            raise TimeoutError('whole_task_deadline_reached')
        yield line


def _http(body, url, key, log, deadline, *, provider):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('whole_task_deadline_reached_before_dispatch')
    headers = {'Authorization': 'Bearer ' + key}
    if key not in log.secrets:
        log.secrets.append(key)
    if provider == 'anthropic':
        headers = {'x-api-key': key, 'anthropic-version': '2023-06-01'}
    log.write('model_request', endpoint=url, provider=provider, body=body, remaining_seconds=remaining)
    # Keep OpenAI on the existing proxy. Other provider TLS resets were resolved
    # by the direct transport used by the earlier benchmark; no global change.
    clients = getattr(_CLIENTS, 'by_provider', None)
    if clients is None:
        clients = {}; _CLIENTS.by_provider = clients
    if provider not in clients:
        clients[provider] = httpx.Client(trust_env=(provider == 'responses'), limits=httpx.Limits(max_connections=2,max_keepalive_connections=2,keepalive_expiry=60))
    client = clients[provider]
    with client.stream('POST', url, headers=headers, json=body, timeout=httpx.Timeout(remaining, connect=min(30, remaining))) as response:
            log.write('http_response', http_status=response.status_code)
            if response.status_code != 200:
                data = response.read().decode('utf-8', errors='replace')
                log.write('native_http_error', http_status=response.status_code, body=data, retry_after=response.headers.get('retry-after'))
                raise RuntimeError('native_http_' + str(response.status_code))
            lines = _lines(response, deadline)
            if provider == 'ollama':
                for line in lines:
                    if line.strip():
                        e = json.loads(line)
                        log.write('native_stream_event', payload=e)
                        yield e
            else:
                for e in auth_helper().read_sse(lines):
                    log.write('native_stream_event', payload=e)
                    yield e


def responses_round(history, schemas, log, deadline, *, route=None):
    if route is None:
        key = get_access_record()['access_token']
        endpoint, provider = 'https://api.openai.com/v1/responses', 'responses'
    else:
        from neurobench.runner.high_gpt_route import checked_route, route_key
        route = checked_route(route)
        key = route_key(route)
        endpoint, provider = route['endpoint'], route['provider']
    body = response_body('gpt-5.6-sol', history, schemas)
    events = list(_http(body, endpoint, key, log, deadline, provider=provider))
    result = completed_response(events)
    native = result['response']
    if native.get('model') != 'gpt-5.6-sol' or (native.get('reasoning') or {}).get('effort') != 'high':
        raise ValueError('native_model_or_high_echo_mismatch')
    return result, native_usage('responses', native.get('usage') or {})


def chat_round(provider, history, schemas, log, deadline, *, ollama_slot=1):
    key = credential(provider, ollama_slot=ollama_slot)
    if provider == 'ollama':
        body = {'model': 'glm-5.3', 'messages': history, 'stream': True, 'think': 'high', 'options': {'num_predict': 65536}}
        if schemas:
            body['tools'] = [{'type': 'function', 'function': s} for s in schemas]
        events = list(_http(body, 'https://ollama.com/api/chat', key, log, deadline, provider='ollama'))
        if not events or not events[-1].get('done'):
            raise RuntimeError('ollama_stream_without_terminal_remote_unknown')
        terminal = events[-1]
        if terminal.get('error'):
            raise RuntimeError('ollama_native_terminal_error')
        if terminal.get('done_reason') not in ('stop', 'tool_calls'):
            raise RuntimeError('ollama_noncomplete_' + str(terminal.get('done_reason')))
        text = ''.join((e.get('message') or {}).get('content', '') for e in events)
        thinking = ''.join((e.get('message') or {}).get('thinking', '') for e in events)
        calls = [c for e in events for c in (e.get('message') or {}).get('tool_calls', [])]
        message = {'role': 'assistant', 'content': text}
        if thinking:
            message['thinking'] = thinking
        if calls:
            message['tool_calls'] = calls
        normalized = [{'name': c['function']['name'], 'arguments': c['function'].get('arguments', {}), 'call_id': str(i)} for i, c in enumerate(calls)]
        return {'message': message, 'text': text, 'calls': normalized, 'stop_reason': terminal.get('done_reason'), 'model': terminal.get('model')}, native_usage('ollama', terminal)
    if provider != 'deepseek':
        raise ValueError('only_official_deepseek_chat_in_core_matrix')
    body = {'model': 'deepseek-v4-pro', 'messages': history, 'stream': True, 'reasoning_effort': 'high',
            'thinking': {'type': 'enabled'}, 'max_tokens': 65536, 'stream_options': {'include_usage': True}}
    if schemas:
        body['tools'] = [{'type': 'function', 'function': s} for s in schemas]
    text, thinking, calls, usage, stop, model = '', '', {}, None, None, None
    for e in _http(body, 'https://api.deepseek.com/chat/completions', key, log, deadline, provider='deepseek'):
        if e.get('error'):
            raise RuntimeError('deepseek_native_stream_error')
        model = e.get('model', model)
        if e.get('usage'):
            usage = e['usage']
        for choice in e.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('unexpected_chat_choice')
            stop = choice.get('finish_reason') or stop
            d = choice.get('delta') or {}
            text += d.get('content') or ''
            thinking += d.get('reasoning_content') or ''
            for c in d.get('tool_calls', []):
                i = c['index']; row = calls.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                row['id'] += c.get('id') or ''
                for k in ('name', 'arguments'):
                    row['function'][k] += (c.get('function') or {}).get(k) or ''
    if stop not in ('stop', 'tool_calls'):
        raise RuntimeError('deepseek_noncomplete_' + str(stop))
    ordered = [calls[k] for k in sorted(calls)]
    message = {'role': 'assistant', 'content': text, 'reasoning_content': thinking}
    if ordered:
        message['tool_calls'] = ordered
    normalized = [{'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments']), 'call_id': c['id']} for c in ordered]
    return {'message': message, 'text': text, 'calls': normalized, 'stop_reason': stop, 'model': model}, native_usage('deepseek', usage or {})


def anthropic_round(history, schemas, log, deadline):
    key = credential('nvidia')
    body = {'model': 'azure/anthropic/claude-opus-4-8', 'messages': history, 'max_tokens': 65536, 'stream': True,
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}}
    if schemas:
        body['tools'] = [{'name': s['name'], 'description': s['description'], 'input_schema': s['parameters']} for s in schemas]
    blocks, partial_args, usage, stop, ended, model = {}, {}, {}, None, False, None
    for e in _http(body, 'https://inference-api.nvidia.com/v1/messages', key, log, deadline, provider='anthropic'):
        typ = e.get('type')
        if typ == 'error':
            raise RuntimeError('anthropic_native_stream_error')
        if typ == 'message_start':
            m = e.get('message') or {}; usage.update(m.get('usage') or {}); model = m.get('model')
        elif typ == 'content_block_start':
            blocks[e['index']] = dict(e['content_block'])
        elif typ == 'content_block_delta':
            b, d = blocks[e['index']], e['delta']
            if d['type'] == 'text_delta': b['text'] = b.get('text', '') + d['text']
            elif d['type'] == 'thinking_delta': b['thinking'] = b.get('thinking', '') + d['thinking']
            elif d['type'] == 'signature_delta': b['signature'] = b.get('signature', '') + d['signature']
            elif d['type'] == 'input_json_delta': partial_args[e['index']] = partial_args.get(e['index'], '') + d['partial_json']
        elif typ == 'message_delta':
            stop = (e.get('delta') or {}).get('stop_reason') or stop; usage.update(e.get('usage') or {})
        elif typ == 'message_stop': ended = True
    if not ended or stop not in ('end_turn', 'tool_use', 'stop_sequence'):
        raise RuntimeError('anthropic_noncomplete_' + str(stop))
    for i, raw in partial_args.items(): blocks[i]['input'] = json.loads(raw)
    content = [blocks[i] for i in sorted(blocks)]
    text = ''.join(b.get('text', '') for b in content if b.get('type') == 'text')
    calls = [{'name': b['name'], 'arguments': b['input'], 'call_id': b['id']} for b in content if b.get('type') == 'tool_use']
    return {'message': {'role': 'assistant', 'content': content}, 'text': text, 'calls': calls, 'stop_reason': stop, 'model': model}, native_usage('anthropic', usage)
