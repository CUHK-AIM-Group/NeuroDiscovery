"""Independent arithmetic, artifact seal and replay readback; no model calls."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import re
import statistics

from neurobench.runner.exec_dev.common import file_map, read_json, sha, write_json, read_csv


def seal(root):
    target = root / 'DELIVERY_BINDINGS.json'
    if target.exists():
        raise ValueError('delivery_already_sealed')
    assert (root / 'RUN_END.json').is_file()
    assert read_json(root / 'delivery_r1/VERIFIED.json')['passed']
    assert read_json(root / 'HISTORICAL_PRESERVATION_FINAL.json')['passed']
    write_json(target, {'scope':'Entire finite trial before independent readback',
        'files':file_map(root), 'reader_source':str(Path(__file__).resolve()),
        'reader_sha256':sha(__file__)})
    print(json.dumps({'sealed_files':len(read_json(target)['files'])}),flush=True)


def readback(root):
    checks, errors = [], []
    def check(name, condition, detail=None):
        row = {'check':name,'passed':bool(condition)}
        if detail is not None:
            row['detail'] = detail
        checks.append(row)
        if not condition:
            errors.append(row)
    binding = read_json(root / 'DELIVERY_BINDINGS.json')
    for p, h in binding['files'].items():
        check('sealed:'+p, (root / p).is_file() and sha(root / p) == h)
    check('reader_unchanged', sha(binding['reader_source']) == binding['reader_sha256'])
    manifest = read_json(root / 'MANIFEST.json')
    check('predeclared_documentation_drift',manifest['original_p0_binding_count']==len(manifest['p0_bindings'])+len(manifest['preexisting_documentation_drift']))
    # Reporting helpers are pinned by the pre-existing sealed deliveries that
    # were hashed before this run, rather than by a newly chosen implementation.
    for package in read_json(root/'HISTORICAL_BINDINGS.json')['packages']:
        original = read_json(package['binding'])
        if 'report_source' in original:
            source = original['report_source']
            check('historical_report_source:'+source['path'], sha(source['path'])==source['sha256'])
        for path, h in original.get('analysis_sources',{}).items():
            check('historical_analysis_source:'+path,sha(path)==h)
    for p, h in manifest['source_bindings'].items():
        check('source:'+p, Path(p).is_file() and sha(p)==h)
    data = read_json(root / 'delivery_r1/RESULTS.json')
    rows, summary, pairs = data['rows'], data['summary'], data['pairs']
    check('rows_and_identity', len(rows)==36 and {r['id'] for r in rows}=={s['id'] for s in manifest['slots']})
    check('only_gpt61', {r['model'] for r in rows}=={'gpt-6.1-sol'})
    check('all_original_once', read_json(root/'RUN_END.json')['finished_attempts']==36)
    for arm in ('A','B0','B1'):
        group = [r for r in rows if r['arm']==arm]
        normal = [r for r in group if r['case_id']!='N12']
        item = next(s for s in summary if s['arm']==arm)
        check(arm+':denominators',len(group)==12 and len(normal)==11)
        check(arm+':completed', item['processable_pass']==sum(r['final_pass'] for r in normal))
        check(arm+':old_caps', item['observed_processable_pass_within_old_caps']==sum(r['observed_pass_within_old_caps'] for r in normal))
        check(arm+':extended', item['processable_pass_beyond_old_caps']==sum(r['pass_beyond_old_caps'] for r in normal))
        check(arm+':median',item['median_all_case_seconds']==statistics.median(r['elapsed_seconds'] for r in group))
        check(arm+':requests',item['native_requests']==sum(r['native_request_count'] for r in group))
        check(arm+':status',item['status_counts']==dict(Counter(r['status'] for r in group)))
        check(arm+':block',item['correct_block']==sum(r['final_pass'] for r in group if r['case_id']=='N12'))
    mapping = {(r['case_id'],r['arm']):r['final_pass'] for r in rows}
    for item in pairs:
        on, off = item['comparison'].split('-')
        actual = Counter()
        for i in range(1,12):
            a,b = mapping[f'N{i:02d}',off],mapping[f'N{i:02d}',on]
            actual['both_pass' if a and b else 'on_only' if b else 'off_only' if a else 'both_fail'] += 1
        check('pair:'+item['comparison'],all(item[k]==actual[k] for k in ('both_pass','on_only','off_only','both_fail')))
    with (root/'delivery_r1/CASE_RESULTS.csv').open(encoding='utf-8-sig',newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    check('csv_identity',len(csv_rows)==36 and {r['id'] for r in csv_rows}=={r['id'] for r in rows})
    for csv_row in csv_rows:
        row=next(r for r in rows if r['id']==csv_row['id'])
        check(row['id']+':csv_selection',csv_row['final_pass']==str(row['final_pass']))
    # Re-execute N07 scripts in new directories using the frozen original policy.
    from neurobench.runner.exec_budget_gpt61_trial import replay_split
    for row in rows:
        if row['case_id'] != 'N07' or not row['submitted']:
            continue
        destination = root/'readback_replays'/row['id']
        destination.mkdir(parents=True,exist_ok=False)
        actual = replay_split(root/'workspaces'/row['id'],destination)
        check(row['id']+':independent_replay',actual['passed']==row['split_reproducibility']['passed'])
        if actual['passed']:
            split=lambda p:{r['subject_id']:r['split'] for r in read_csv(p)}
            check(row['id']+':actual_replay_membership',split(destination/'split_reproducibility/output/split.csv')==split(root/'workspaces'/row['id']/'output/split.csv'))
    report=root/'delivery_r1/REPORT.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if '://' not in target:
            check('link:'+target,(report.parent/target).is_file())
    from PIL import Image
    with Image.open(root/'delivery_r1/COMPARISON.png') as picture:
        check('figure_dimensions',picture.width>=1200 and picture.height>=600)
        picture.verify()
    check('pdf_header',(root/'delivery_r1/COMPARISON.pdf').read_bytes().startswith(b'%PDF-'))
    check('svg_header','<svg' in (root/'delivery_r1/COMPARISON.svg').read_text(encoding='utf-8'))
    result={'passed':not errors,'checks':len(checks),'errors':errors,'sealed_files':len(binding['files']),
            'records':checks,'native_requests_added':0,'status':'VERIFIED_WITH_PREEXISTING_DOCUMENTATION_DRIFT','full_historical_preservation_passed':False,'scope':'Independent file/arithmetic/replay/format readback; exposed development only.'}
    write_json(root/'DELIVERY_READBACK.json',result)
    print(json.dumps({k:result[k] for k in ('passed','checks','errors','sealed_files','native_requests_added')},ensure_ascii=False),flush=True)
    return not errors


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('seal','readback'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='seal':
        seal(args.root.resolve())
    else:
        raise SystemExit(0 if readback(args.root.resolve()) else 1)
