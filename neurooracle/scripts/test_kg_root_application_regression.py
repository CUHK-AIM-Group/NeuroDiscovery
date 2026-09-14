"""R70 current KG regressions, with explicit obsolete-live R59 exclusions."""
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from apply_kg_root_repairs import OUTPUT, R69


def main():
    require(not (OUTPUT / 'BUILD_STATE.json').exists(), 'regression evidence already frozen')
    OUTPUT.mkdir(exist_ok=True)
    paths = set()
    for name in ('WIDE_REGRESSION_TEST_RESULTS.xml', 'INGESTION_TEST_RESULTS.xml',
                 'REPAIR_UNIT_TEST_RESULTS.xml', 'VALIDATOR_TEST_RESULTS.xml', 'REPORT_TEST_RESULTS.xml'):
        for case in ET.parse(R69 / name).getroot().iter('testcase'):
            bits = case.get('classname').split('.')
            while bits:
                path = Path('/'.join(bits) + '.py')
                if path.is_file():
                    paths.add(path.as_posix()); break
                bits.pop()
            else:
                raise ValueError('test module unavailable')
    paths.update(['neurooracle/tests/test_kg_root_application.py', 'neurooracle/tests/test_relation_evidence_dossier.py'])
    scope = j.read_json(j.OUTPUT / 'round67_relation_semantic_consolidation/REGRESSION_SCOPE.json')
    excludes = scope['unselected_obsolete_live_r59_cases']
    require(len(excludes) == 18, 'unexpected exclusion scope')
    j.atomic_json(OUTPUT / 'REGRESSION_SCOPE.json', dict(at=j.utc_now(), selected_modules=sorted(paths),
        unselected_obsolete_live_r59_cases=excludes, runner_code=j.fingerprint(Path(__file__)),
        no_cases_falsely_reported_as_passing=True, replacements='Independent complete R70 candidate, census, metadata and live evidence queries'))
    command = [sys.executable, '-X', 'utf8', '-m', 'pytest', *sorted(paths), '-q', '--junitxml=' + str(OUTPUT / 'TEST_RESULTS.xml')]
    for selector in excludes:
        command.extend(['--deselect', selector])
    sys.exit(subprocess.call(command))


if __name__ == '__main__':
    main()
