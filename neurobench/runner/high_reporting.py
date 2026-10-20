"""Read-only all-denominator metrics. Unknown usage never becomes zero cost."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def _events(p):
    if not p.exists(): return []
    rows=[]
    for line in p.read_text(encoding='utf-8').split('\n'):
        if not line: continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: break  # live writer's unfinished line
    return rows


def summarize(run: Path):
    with sqlite3.connect('file:'+(run/'state.sqlite3').resolve().as_posix()+'?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        slots=[dict(r) for r in con.execute('select * from slots order by model_block,case_id,arm')]
    metrics=[]
    for slot in slots:
        r=read(slot['receipt_path']) if slot.get('receipt_path') else {}
        attempt=run/'attempts'/slot['condition_id']/slot['case_id']
        events=_events(attempt/'events.jsonl') or _events(attempt/'native_responses_events.jsonl')
        usage=r.get('usage') or {}
        # A final request can be unaccounted for even when earlier requests have
        # complete counters. The receipt's observed total is a lower bound.
        unknown=slot['status']=='UNKNOWN'
        known=bool(usage.get('usage_complete')) and not unknown
        calls=[e for e in events if e.get('event')=='model_request']
        native=[e for e in events if e.get('event')=='native_stream_event']
        tools=r.get('tool_calls') or []
        def delta(e):
            p=e.get('payload') or {}
            if p.get('type')=='response.output_text.delta': return bool(p.get('delta'))
            message=p.get('message') or {}
            if message.get('content'): return True
            return any((c.get('delta') or {}).get('content') for c in p.get('choices',[]))
        texts=[e for e in native if delta(e)]
        def elapsed(event):
            if not calls or event is None: return None
            return (datetime.fromisoformat(event['timestamp'])-datetime.fromisoformat(calls[0]['timestamp'])).total_seconds()
        records=r.get('native_usage_records') or []
        normalized_records=[u for u in records if isinstance(u,dict) and 'input_tokens' in u and 'output_tokens' in u]
        caches=[u.get('cached_input_tokens') for u in normalized_records]
        writes=[u.get('cache_write_tokens',u.get('cache_write_input_tokens')) for u in normalized_records]
        reasoning=[u.get('reasoning_tokens',u.get('reasoning_output_tokens')) for u in normalized_records]
        def complete_sum(values):
            return sum(values) if values and all(isinstance(v,int) and not isinstance(v,bool) for v in values) else None
        diagnostic=read(attempt/'DIAGNOSTICS.json') if (attempt/'DIAGNOSTICS.json').exists() else {}
        metric=dict(slot)
        metric.update(
            elapsed_seconds=r.get('elapsed_seconds'), first_native_event_seconds=elapsed(native[0] if native else None),
            first_text_seconds=elapsed(texts[0] if texts else None),
            observed_input_tokens=usage.get('input_tokens') if r else None,
            observed_output_tokens=usage.get('output_tokens') if r else None,
            observed_total_tokens=usage.get('total_tokens') if r else None,
            complete_total_tokens=usage.get('total_tokens') if known else None,
            usage_complete_for_attempt=known, unknown_usage_is_zero=False,
            cached_input_tokens=complete_sum(caches), cache_write_tokens=complete_sum(writes),
            reasoning_output_tokens=complete_sum(reasoning),
            native_request_count=len(calls) if events else r.get('native_request_count'),
            native_tool_calls=len(tools) if r else None,
            skills_loaded=len({(t.get('arguments') or {}).get('name') for t in tools
                               if 'read_skill' in t.get('name','') and (t.get('arguments') or {}).get('name')
                               and (t.get('ok') or (t.get('result') or {}).get('success') is True)}),
            queue_seconds=None, model_wall_seconds=None, tool_wall_seconds=None,
            feedback_repair_used=r.get('feedback_repair_used'),
            blocking_static_errors=diagnostic.get('blocking_error_count'),
            error=r.get('error'), complete_answer=slot['status']=='COMPLETE',
            workflow_executed=False, scientific_correctness_verified=False)
        metrics.append(metric)
    counts=dict(Counter(r['status'] for r in slots))
    pairs={}
    for model in sorted({r['model_block'] for r in slots}):
        rows=[r for r in slots if r['model_block']==model]
        arms=sorted({r['arm'] for r in rows})
        cases=sorted({r['case_id'] for r in rows})
        full=sum(all(next(r for r in rows if r['case_id']==case and r['arm']==arm)['status']=='COMPLETE' for arm in arms) for case in cases)
        pairs[model]={'planned_cases':len(cases),'arms':arms,'complete_bundles':full}
    return metrics,{'planned_primary_slots':len(slots),'status_counts':counts,'complete_bundles':pairs,
                    'available_case_scores_are_not_all_sample_scores':True,
                    'observed_tokens_are_not_actual_subscription_payment':True,
                    'superiority_established':False}


def csv_text(rows):
    if not rows: return ''
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue()
