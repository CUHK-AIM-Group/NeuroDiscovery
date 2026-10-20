"""Explicit user-authorized GPT API route; credentials never enter artifacts."""
from __future__ import annotations

from pathlib import Path

XNOVA_ROUTE = {
    'provider': 'xnova_responses',
    'endpoint': 'https://api.xnova.online/v1/responses',
    'model': 'gpt-5.6-sol',
    'reasoning_effort': 'high',
    'credential_path': r'C:\Users\45846\OneDrive\keys\gpt.txt',
    'official_openai_endpoint': False,
    'backend_identity': 'provider_reported_not_independently_attested',
}


def checked_route(route):
    if route != XNOVA_ROUTE:
        raise ValueError('unapproved_gpt_experiment_route')
    return dict(XNOVA_ROUTE)


def route_key(route):
    value = checked_route(route)
    rows = [r.strip() for r in Path(value['credential_path']).read_text(encoding='utf-8-sig').splitlines() if r.strip()]
    if len(rows) != 1 or any(c.isspace() for c in rows[0]):
        raise ValueError('expected_one_plain_gpt_api_key')
    return rows[0]
