"""Validate source-bound root extractions and store a separate literature layer.

This module does not extract semantics automatically, call a model, infer support
from a p value, write fixed entities, or publish a graph. Root supplies explicit
observation/claim/evidence decisions after reading the bound source material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from prepare_paper_reconstruction import digest, dump, now, packet, read, settings

ROLES = {'primary_result', 'synthesis_result', 'background', 'hypothesis', 'protocol', 'method'}
RELATIONS = {'supports', 'opposes', 'partial', 'contextual', 'unresolved'}
CONDITIONS = ('species', 'population', 'stage', 'age', 'sex', 'tissue_or_region',
              'modality', 'measurement', 'intervention', 'dose', 'comparator',
              'timepoint', 'adjustment')
SCHEMA = '''
create table if not exists batches(batch_id text primary key, sha256 text, path text, at text);
create table if not exists papers(job_id text primary key, work_key text, identity_status text,
    source_sha256 text, batch_id text, source_packet_json text, extraction_json text);
create table if not exists claims(claim_id text primary key, semantic_key_json text unique,
    statement text, definition_json text);
create table if not exists observations(observation_id text primary key, job_id text,
    role text, statement text, payload_json text,
    foreign key(job_id) references papers(job_id));
create table if not exists evidence(observation_id text, claim_id text, relation text,
    rationale text, scope_match text, primary key(observation_id,claim_id),
    foreign key(observation_id) references observations(observation_id),
    foreign key(claim_id) references claims(claim_id));
create index if not exists evidence_claims on evidence(claim_id,relation);
create index if not exists observation_paper on observations(job_id);
create table if not exists claim_relations(a text,b text,relation text,rationale text,
    primary key(a,b,relation),foreign key(a) references claims(claim_id),
    foreign key(b) references claims(claim_id));
'''


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def claim_id(claim):
    return 'PC:' + digest(claim['semantic_key'])


def text_of(source):
    return source['source_snapshot'].get('abstract_text') or source['source_snapshot'].get('abstract') or ''


def anchor(text, quote):
    if not quote or text.count(quote) != 1:
        raise ValueError('Source anchor must match exactly once: ' + quote[:100])
    start = text.index(quote)
    return {'field': 'abstract', 'start': start, 'end': start + len(quote), 'quote': quote}


def validate(extraction, source, claims):
    if source['old_claims_included'] or source['fixed_layer_write_permission']:
        raise ValueError('Wrong extraction scope')
    if extraction['job_id'] != source['job']['job_id'] or extraction['source_sha256'] != source['source_sha256']:
        raise ValueError('Extraction belongs to another paper snapshot')
    # Historical caches are not authoritative enough to resolve all versions,
    # but an explicit owner-ID contradiction must never pass unnoticed.
    expected = {}
    for publication in source.get('publications', []):
        for kind, value in json.loads(publication['ids_json']).items():
            if value: expected.setdefault(kind.lower(), set()).add(str(value).lower())
    envelope = source['source_snapshot']
    for record in (envelope, envelope.get('paper', {}), envelope.get('own_article_ids') or {}):
        if not isinstance(record, dict): continue
        for kind in ('pmid', 'doi', 'pmcid'):
            value = record.get(kind)
            if value and kind in expected and str(value).lower() not in expected[kind]:
                raise ValueError('Source owning identifier conflicts with paper queue: ' + kind)
    if extraction['review']['reader'] != 'root' or extraction['review']['coverage'] != 'available_abstract':
        raise ValueError('Only root-reviewed available-abstract work is enabled')
    if extraction.get('fixed_entity_mutations'):
        raise ValueError('Fixed entities are read-only')
    if extraction['study']['cohort_overlap_status'] not in {'unknown', 'overlap_known', 'independence_verified'}:
        raise ValueError('Missing study independence status')
    text = text_of(source)
    if not text:
        raise ValueError('No source text')
    seen = set()
    for observation in extraction['observations']:
        oid = observation['observation_id']
        if oid in seen:
            raise ValueError('Duplicate observation identity')
        seen.add(oid)
        if observation['role'] not in ROLES:
            raise ValueError('Unknown passage role')
        if not observation['anchors'] or not observation['statement']:
            raise ValueError('Missing observation or source anchor')
        for a in observation['anchors']:
            if a['field'] != 'abstract' or a['start'] < 0 or a['end'] <= a['start']:
                raise ValueError('Invalid source coordinates')
            if text[a['start']:a['end']] != a['quote']:
                raise ValueError('Source quote does not match snapshot')
        if set(CONDITIONS) - set(observation['conditions']):
            raise ValueError('Condition keys absent; use null and missingness reasons')
        if not isinstance(observation['missing_fields'], dict):
            raise ValueError('Missingness must be explicit')
        if not {'direction', 'significance', 'effect_estimate', 'confidence_interval'} <= observation['result'].keys():
            raise ValueError('Incomplete result structure')
        raw_quotes = '\n'.join(a['quote'] for a in observation['anchors'])
        for statistic in observation['statistics']:
            if not statistic['raw'] or statistic['raw'] not in raw_quotes:
                raise ValueError('Statistic not anchored to source')
            if statistic.get('type') == 'p_value' and statistic.get('operator') not in {'=', '<', '<=', '>', '>='}:
                raise ValueError('Preserve p-value operator')
        linked = set()
        for link in observation['evidence']:
            if link['claim_key'] not in claims or link['claim_key'] in linked:
                raise ValueError('Missing or duplicate target claim')
            linked.add(link['claim_key'])
            if link['relation'] not in RELATIONS or not link['rationale'] or not link['scope_match']:
                raise ValueError('Missing semantic evidence decision')
            if observation['role'] in {'background', 'hypothesis', 'protocol', 'method'} and link['relation'] == 'supports':
                raise ValueError('Non-result passage cannot supply experimental support')
    if not seen:
        raise ValueError('No extracted observations')


def validate_claims(claims):
    for key, claim in claims.items():
        sem = claim['semantic_key']
        if not {'subject', 'relation_kind', 'measurement', 'necessary_qualifiers'} <= sem.keys():
            raise ValueError('Incomplete shared proposition: ' + key)
        if any(k in sem for k in ('pmid', 'doi', 'paper_id', 'sample_size', 'p_value', 'title', 'year')):
            raise ValueError('Paper-specific attributes cannot identify a shared proposition')
        if not claim['statement'] or not claim['scope_rule']:
            raise ValueError('Claim needs an explicit scope rule')


def metrics(c):
    shared = []
    for row in c.execute('''select e.claim_id,c.statement,count(distinct p.work_key)
      from evidence e join observations o using(observation_id) join papers p using(job_id)
      join claims c using(claim_id) where e.relation='supports' and o.role='primary_result'
      and p.work_key is not null and p.identity_status not in ('unresolved','conflict','version_identity_review')
      group by e.claim_id having count(distinct p.work_key)>1 order by count(distinct p.work_key) desc'''):
        shared.append({'claim_id': row[0], 'statement': row[1], 'primary_supporting_work_records': row[2]})
    return {'extracted_papers': c.execute('select count(*) from papers').fetchone()[0],
            'observations': c.execute('select count(*) from observations').fetchone()[0],
            'claims': c.execute('select count(*) from claims').fetchone()[0],
            'evidence_links': c.execute('select count(*) from evidence').fetchone()[0],
            'roles': dict(c.execute('select role,count(*) from observations group by role')),
            'evidence_roles': dict(c.execute('select relation,count(*) from evidence group by relation')),
            'multipaper_claims_with_primary_support': len(shared), 'shared_claims': shared,
            'independent_cohort_replications_verified': 0,
            'counting_note': 'Distinct declared/own-record work identities; independent cohorts are unknown. Partial, opposing, contextual and synthesis evidence is retained and excluded from primary-support counts.',
            'production_claims_added': 0}


def sync_queue(c, out):
    q = sqlite3.connect(out / 'PAPERS.sqlite')
    with q:
        for job, source, result in c.execute('select job_id,source_sha256,extraction_json from papers'):
            row = q.execute('select result_json from extraction_results where job_id=?', (job,)).fetchone()
            if row and row[0] != result:
                raise ValueError('Conflicting previous queue extraction; no silent overwrite')
            q.execute('insert or ignore into extraction_results values(?,?,?,?,?)',
                      (job, source, result, 'root_source_reviewed_staging', now()))
            q.execute("update paper_jobs set status='abstract_extracted_staging' where job_id=?", (job,))
    counts = dict(q.execute('select status,count(*) from paper_jobs group by status'))
    q.close()
    return counts


def ingest(path):
    _, out, _, _ = settings()
    path = Path(path).resolve()
    if not path.is_relative_to(out):
        raise ValueError('Extraction ledger must be in reconstruction workspace')
    batch = read(path)
    batch_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_claims(batch['claims'])
    sources = {}
    for paper in batch['papers']:
        if paper['job_id'] in sources:
            raise ValueError('Duplicate paper in batch')
        sources[paper['job_id']] = packet(paper['job_id'])
        validate(paper, sources[paper['job_id']], batch['claims'])
    c = sqlite3.connect(out / 'LITERATURE.sqlite')
    c.execute('pragma foreign_keys=ON')
    c.executescript(SCHEMA)
    prior = c.execute('select sha256 from batches where batch_id=?', (batch['batch_id'],)).fetchone()
    if prior and prior[0] != batch_sha:
        raise ValueError('Registered batch is immutable; submit an explicit separate revision')
    receipt=out/(batch['batch_id']+'_IMPORTED.json')
    if prior and receipt.exists():
        sync_queue(c,out)
        c.close()
        report=read(receipt)
        print(json.dumps(report,ensure_ascii=False))
        return report
    ids = {k: claim_id(v) for k, v in batch['claims'].items()}
    if not prior:
        with c:
            c.execute('insert into batches values(?,?,?,?)', (batch['batch_id'], batch_sha, str(path), now()))
            for key, claim in batch['claims'].items():
                old = c.execute('select definition_json from claims where claim_id=?', (ids[key],)).fetchone()
                if old and old[0] != json_text(claim):
                    raise ValueError('Claim scope changed; explicit reconciliation required')
                c.execute('insert or ignore into claims values(?,?,?,?)',
                          (ids[key], json_text(claim['semantic_key']), claim['statement'], json_text(claim)))
            for paper in batch['papers']:
                source = sources[paper['job_id']]
                identity = source['job']['identity_status']
                if source['job']['status'] == 'version_identity_review':
                    identity = 'version_identity_review'
                c.execute('insert into papers values(?,?,?,?,?,?,?)',
                          (paper['job_id'], source['job']['work_key'], identity, paper['source_sha256'],
                           batch['batch_id'], json_text(source), json_text(paper)))
                for observation in paper['observations']:
                    c.execute('insert into observations values(?,?,?,?,?)',
                              (observation['observation_id'], paper['job_id'], observation['role'],
                               observation['statement'], json_text(observation)))
                    for link in observation['evidence']:
                        c.execute('insert into evidence values(?,?,?,?,?)',
                                  (observation['observation_id'], ids[link['claim_key']], link['relation'],
                                   link['rationale'], link['scope_match']))
            for link in batch.get('claim_relations', []):
                c.execute('insert or ignore into claim_relations values(?,?,?,?)',
                          (ids[link['a']], ids[link['b']], link['relation'], link['rationale']))
    statuses = sync_queue(c, out)
    report = {'at': now(), 'batch_id': batch['batch_id'], 'batch_sha256': batch_sha,
              'staging_only': True, **metrics(c)}
    if c.execute('pragma integrity_check').fetchone()[0] != 'ok' or c.execute('pragma foreign_key_check').fetchall():
        raise ValueError('Evidence store integrity failure')
    c.close()
    dump(out / (batch['batch_id'] + '_IMPORTED.json'), report)
    state = read(out / 'STATE.json')
    partition = out / 'FIXED_BOUNDARY.json'
    if not partition.exists():partition=out/'FIXED_PARTITION_READY.json'
    state.update(status='SOURCE_EXTRACTION_VALIDATION_IN_PROGRESS', at=now(),
                 **{k:v for k,v in report.items() if k not in {'at','batch_id','batch_sha256','shared_claims'}},
                 latest_batch=batch['batch_id'], status_counts=statuses,
                 fixed_partition_status=read(partition)['status'] if partition.exists() else 'partition_running',
                 next='review provenance holds and source extraction coverage; then scale full paper queue')
    dump(out / 'STATE.json', state)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch')
    ingest(p.parse_args().batch)
