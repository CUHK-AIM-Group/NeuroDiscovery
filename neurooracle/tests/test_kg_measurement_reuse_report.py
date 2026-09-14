from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_measurement_reuse import restored_pairs, validate_retirement_paths, PAIRS, DERIVATIVES


def groups():
    return [dict(id='REL:'+str(i), members=[dict(claim_id=cid, paper_key='pmid:39829963', paper_status='verified')
                                          for cid in pair]) for i,pair in enumerate(PAIRS)]


def test_same_paper_restoration_is_not_new_independent_evidence():
    assert all(r['new_independent_sources'] == 0 for r in restored_pairs(groups()))


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'unverified', 'different_paper'])
def test_restoration_requires_exact_complete_pair_and_source(problem):
    data = groups()
    if problem == 'missing': data[0]['members'].pop()
    if problem == 'duplicate': data.append(data[0])
    if problem == 'unverified': data[0]['members'][0]['paper_status'] = 'unverified'
    if problem == 'different_paper': data[0]['members'][0]['paper_key'] = 'pmid:11111111'
    with pytest.raises(Exception): restored_pairs(data)


def files(tmp_path):
    previous = tmp_path/'previous'; previous.mkdir()
    paths = [previous/name for name in DERIVATIVES]
    for p in paths: p.touch()
    return previous, paths, [dict(path=str(p)) for p in paths]


def test_retirement_only_exact_four_old_derivatives(tmp_path):
    previous, paths, fps = files(tmp_path)
    assert validate_retirement_paths(fps, set(), previous, tmp_path) == paths
    assert all(p.exists() for p in paths)


def test_current_file_can_never_be_retired(tmp_path):
    previous, paths, fps = files(tmp_path)
    with pytest.raises(Exception): validate_retirement_paths(fps, {paths[0]}, previous, tmp_path)


def test_retirement_cannot_escape_root_or_use_incomplete_set(tmp_path):
    previous, paths, fps = files(tmp_path)
    with pytest.raises(Exception): validate_retirement_paths(fps, set(), previous, previous/'wrong-root')
    with pytest.raises(Exception): validate_retirement_paths(fps[:3], set(), previous, tmp_path)

