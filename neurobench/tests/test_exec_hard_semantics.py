"""Regressions for observed public-contract interpretation defects."""
from neurobench.runner.exec_hard.semantic import *


def test_identity_is_not_json_storage_type_or_float_rounding():
    assert participant_id(50003)=='50003'
    assert participant_id('050003')=='050003'
    assert participant_id(50003.0)==50003.0
    assert participant_id(True) is True


def test_quality_envelopes_preserve_supplied_roi_evidence():
    records=[{'participant_id':'50003','atlas':'aal','invalid_roi_ids':['2001']}]
    assert qc_records({'participants':records})==records
    assert qc_records({'invalid_roi_ids':{'50003':{'aal':['2001']}}})==records


def test_covariate_meanings_accept_labels_and_order_but_not_missing_terms():
    expected=['intercept','label','age','sex2','mean_fd','site:NYU']
    assert covariates_equivalent(['C(site)[T.NYU]','const','DX_GROUP','AGE_AT_SCAN','sex_2','func_mean_fd'],expected)
    assert not covariates_equivalent(['const','label','age','sex','fd','age'],expected)


def test_public_external_metadata_is_not_a_hidden_test_metric():
    rows=[{'participant_id':'50003','split':'external','site':'NYU'}]
    base={'external_metrics':'pending_host_evaluation'}
    assert external_metadata_valid(dict(base,n_external=1,external_sites=['NYU']),rows)
    assert not external_metadata_valid(dict(base,external_accuracy=.99),rows)
    assert not external_metadata_valid(dict(base,n_external=2),rows)


def test_step_envelope_does_not_invent_missing_actions():
    assert step_records([{'step_id':'fc:50003','action':'computed'}])==[{'name':'fc:50003','action':'execute'}]
    assert step_records([{'step':'group','action':'run'}])==[{'name':'group','action':'execute'}]
    original={'fc:50003':{'inputs':['inputs/series.npy'],'outputs':['output/fc.npz'],'config':{}}}
    assert dependency_records({'steps':original,'metadata':'extra'})==original
    annotated={'metric:50003':{'config':{'threshold':.2,'threshold_source':'inputs/config.json'}}}
    assert dependency_records(annotated)['metric:50003']['config']=={'threshold':.2}
    assert 'threshold_source' in annotated['metric:50003']['config']
    assert default_group_config({'aggregation':'mean_of_participant_means_and_sum_of_counts','metrics':'all'})
    assert not default_group_config({'aggregation':'median','metrics':'all'})


def test_density_uses_exact_decimal_ceiling_at_integer_boundary():
    assert density_edge_count(116)==667
    assert density_edge_count(115)==656
    assert density_edge_count(200)==1990


def test_fit_invariance_uses_public_precision_and_feature_identity(tmp_path):
    import numpy as np
    a=dict(feature_ids=np.array(['a','b']),mean=np.array([0.,1.]),scale=np.array([1.,2.]),coef=np.array([.2,.3]),intercept=np.array(.1))
    np.savez(tmp_path/'a.npz',**a)
    a['coef'][0]+=4.7e-7;np.savez(tmp_path/'b.npz',**a)
    assert model_state_equivalent(tmp_path/'a.npz',tmp_path/'b.npz')
    a['coef'][0]+=.01;np.savez(tmp_path/'c.npz',**a)
    assert not model_state_equivalent(tmp_path/'a.npz',tmp_path/'c.npz')
