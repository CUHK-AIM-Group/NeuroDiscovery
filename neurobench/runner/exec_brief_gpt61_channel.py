"""GPT-6.1-Sol/high replacement channel, preserving the frozen GPT-5.6 route.

Only the requested model changes. Endpoint, credentials, prompt construction,
stream parser and usage accounting remain those of the previous direct trial.
"""
from __future__ import annotations

from neurobench.runner.high_channels import _http, native_usage
from neurobench.runner.high_gpt_route import XNOVA_ROUTE, route_key
from neurobench.runner.subscription_direct import completed_response, response_body

MODEL = 'gpt-6.1-sol'
GPT61_ROUTE = {**XNOVA_ROUTE, 'model': MODEL}


def responses_round(history, schemas, log, deadline, *, route=None):
    if route is not None and route != GPT61_ROUTE:
        raise ValueError('unexpected_gpt61_experiment_route')
    body = response_body(MODEL, history, schemas)
    events = list(_http(body, GPT61_ROUTE['endpoint'], route_key(XNOVA_ROUTE),
                        log, deadline, provider=GPT61_ROUTE['provider']))
    result = completed_response(events)
    native = result['response']
    if native.get('model') != MODEL or (native.get('reasoning') or {}).get('effort') != 'high':
        raise ValueError('native_model_or_high_echo_mismatch')
    return result, native_usage('responses', native.get('usage') or {})
