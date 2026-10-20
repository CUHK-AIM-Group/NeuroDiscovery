"""Separate-process readback of the sealed replacement trial and archived rows."""
from pathlib import Path
import argparse
import csv
import json
import statistics

from neurobench.runner.exec_dev.common import read_json, sha, write_json


def verify(root):
    root = root.resolve()
    seal = read_json(root / 'DELIVERY_BINDINGS.json')
    checks, errors = 0, []

    def check(name, passed):
        nonlocal checks
        checks += 1
        if not passed:
            errors.append(name)

    for p, expected in seal['files'].items():
        path = Path(p) if Path(p).is_absolute() else root / p
        check('new_file:' + p, path.is_file() and sha(path) == expected)
    manifest = read_json(root / 'MANIFEST.json')
    for p, expected in manifest['source_bindings'].items():
        check('live_source:' + p, sha(p) == expected)
    for p, expected in read_json(root / 'REPORT_SOURCE_BINDINGS.json')['report_sources'].items():
        check('report_source:' + p, sha(p) == expected)
    old_results_path = Path(manifest['comparison_reference'])
    check('archived_results_binding', sha(old_results_path) == manifest['comparison_reference_sha256'])
    old_root = old_results_path.parent.parent
    old_seal = read_json(old_root / 'DELIVERY_BINDINGS.json')
    check('prior_seal_unchanged', sha(old_root / 'DELIVERY_BINDINGS.json') ==
          read_json(root / 'PRIOR_DELIVERY_PRESERVATION.json')['old_bindings_sha256'])
    for p, expected in old_seal['files'].items():
        path = Path(p) if Path(p).is_absolute() else old_root / p
        check('prior_file:' + p, path.is_file() and sha(path) == expected)
    final = read_json(root / 'delivery_r1/RESULTS.json')
    old = read_json(old_results_path)
    archived_by_id = {r['id']: r for r in old['rows'] if r['model'] == 'deepseek-v4.1-flash'}
    new_rows = [r for r in final['rows'] if r['result_origin'] == 'new_gpt61_attempt']
    old_rows = [r for r in final['rows'] if r['result_origin'] == 'archived_flash_no_new_request']
    check('exact_36_new_rows', len(new_rows) == 36 and len({r['id'] for r in new_rows}) == 36)
    check('exact_36_archived_rows', len(old_rows) == 36 and len({r['id'] for r in old_rows}) == 36)
    check('only_gpt61_dispatched', all(s['model'] == 'gpt-6.1-sol' for s in manifest['slots']))
    for row in old_rows:
        fields = {k: v for k, v in row.items() if k not in ('result_origin', 'source_results')}
        check('archived_row_unchanged:' + row['id'], fields == archived_by_id.get(row['id']))
    for row in new_rows:
        raw = read_json(root / 'attempts' / row['id'] / 'RESULT.json')
        check('raw_identity:' + row['id'], raw['model'] == row['model'] == 'gpt-6.1-sol')
        check('raw_status:' + row['id'], raw['status'] == row['status'] and raw['error'] == row['error'])
        check('raw_pass_preserved:' + row['id'], raw['final_pass'] == row['raw_final_pass'])
        check('time_preserved:' + row['id'], raw['elapsed_seconds'] == row['elapsed_seconds'])
        check('tokens_preserved:' + row['id'], raw['tokens_observed'] == row['tokens_observed'])
    for summary in final['summary']:
        group = [r for r in final['rows'] if (r['model'], r['arm']) == (summary['model'], summary['arm'])]
        normal = [r for r in group if r['case_id'] != 'N12']
        blocked = [r for r in group if r['case_id'] == 'N12']
        check('summary_count:' + summary['model'] + summary['arm'],
              len(normal) == summary['processable_total'] == 11 and
              sum(bool(r['final_pass']) for r in normal) == summary['processable_pass'] and
              sum(bool(r['final_pass']) for r in blocked) == summary['correct_block'])
        check('summary_median:' + summary['model'] + summary['arm'],
              statistics.median(r['elapsed_seconds'] for r in group) == summary['median_all_case_seconds'])
        check('summary_resources:' + summary['model'] + summary['arm'],
              sum(r['tokens_observed'] for r in group) == summary['observed_tokens'] and
              sum(r['native_request_count'] for r in group) == summary['native_requests'])
    with (root / 'delivery_r1/CASE_RESULTS.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = list(csv.DictReader(stream))
    check('csv_coverage', len(table) == 72 and {r['id'] for r in table} == {r['id'] for r in final['rows']})
    json_rows = {r['id']: r for r in final['rows']}
    for row in table:
        check('csv_pass:' + row['id'], (row['final_pass'] == 'True') == json_rows[row['id']]['final_pass'])
    check('cold_audit_passed', read_json(root / 'delivery_r1/VERIFIED.json')['passed'])
    check('original_slots_ended', read_json(root / 'RUN_END.json')['finished_attempts'] == 36)
    qa = root / 'readback_r1/VISUAL_AND_LINK_QA.json'
    check('visual_and_links_passed', qa.is_file() and read_json(qa)['passed'])
    result = {'passed': not errors, 'checks': checks, 'errors': errors,
              'new_sealed_files': len(seal['files']), 'prior_sealed_files': len(old_seal['files']),
              'independently_recomputed_summaries': len(final['summary']),
              'archived_rows_unchanged': len(old_rows),
              'scope': 'File hashes, row inheritance, raw status preservation, summary/CSV arithmetic and figure/link QA; not expert scientific or backend-weight acceptance.'}
    write_json(root / 'DELIVERY_READBACK.json', result)
    print(json.dumps(result), flush=True)
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(0 if verify(arguments.root) else 1)
