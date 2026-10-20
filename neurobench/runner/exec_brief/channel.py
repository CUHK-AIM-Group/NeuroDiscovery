"""Exact official Flash model; keep raw SSE and explicit high reasoning."""
import json
from neurobench.runner.high_channels import _http, credential, native_usage

MODEL = 'deepseek-flash'
VALID_ECHOES = {'deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017'}


def flash_round(history, schemas, log, deadline):
    body = {'model': MODEL, 'messages': history, 'stream': True,
            'reasoning_effort': 'high', 'thinking': {'type': 'enabled'},
            'max_tokens': 65536, 'stream_options': {'include_usage': True}}
    if schemas:
        body['tools'] = [{'type': 'function', 'function': s} for s in schemas]
    text, thinking, calls, usage, stop, model = '', '', {}, None, None, None
    for e in _http(body, 'https://api.deepseek.com/chat/completions', credential('deepseek'),
                   log, deadline, provider='deepseek'):
        if e.get('error'):
            raise RuntimeError('deepseek_native_stream_error')
        model = e.get('model', model)
        if e.get('usage'):
            usage = e['usage']
        for choice in e.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('unexpected_chat_choice')
            stop = choice.get('finish_reason') or stop
            delta = choice.get('delta') or {}
            text += delta.get('content') or ''
            thinking += delta.get('reasoning_content') or ''
            for item in delta.get('tool_calls', []):
                row = calls.setdefault(item['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                row['id'] += item.get('id') or ''
                for key in ('name', 'arguments'):
                    row['function'][key] += (item.get('function') or {}).get(key) or ''
    if stop not in ('stop', 'tool_calls'):
        raise RuntimeError('deepseek_noncomplete_' + str(stop))
    if (model or '').casefold() not in VALID_ECHOES:
        raise RuntimeError('native_model_echo_mismatch:' + str(model))
    ordered = [calls[i] for i in sorted(calls)]
    message = {'role': 'assistant', 'content': text, 'reasoning_content': thinking}
    if ordered:
        message['tool_calls'] = ordered
    normalized = [{'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments']),
                   'call_id': c['id']} for c in ordered]
    return {'message': message, 'text': text, 'calls': normalized, 'stop_reason': stop, 'model': model}, native_usage('deepseek', usage or {})
