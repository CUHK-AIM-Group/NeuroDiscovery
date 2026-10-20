"""Public-contract equivalence helpers, without changing scientific values."""
import re


def participant_id(value):
    # Strings retain padding and namespace. Floats/bools are not valid ID coercions.
    return str(value) if isinstance(value,int) and not isinstance(value,bool) else value


def participant_ids(values):
    return {participant_id(v) for v in values}


def qc_records(value):
    if isinstance(value,list):return value
    for key in ('records','participants','series'):
        if isinstance(value.get(key),list):return value[key]
    if isinstance(value.get('invalid_roi_ids'),dict):
        return [dict(participant_id=sid,atlas=atlas,invalid_roi_ids=ids)
                for sid,atlases in value['invalid_roi_ids'].items() for atlas,ids in atlases.items()]
    raise ValueError('missing_participant_atlas_quality_records')


def covariates_equivalent(labels,expected):
    aliases={'intercept':{'intercept','const','constant'},'label':{'label','diagnosis','dx','dxgroup'},
             'age':{'age','ageatscan'},'sex2':{'sex','sex2','sexindicatorsex2','female'},
             'mean_fd':{'meanfd','funcmeanfd','fd'}}
    result=[]
    for label in labels:
        token=re.sub('[^a-z0-9]','',str(label).lower())
        found=next((key for key,values in aliases.items() if token in values),None)
        if found is None:
            sites=[key for key in expected if key.startswith('site:') and
                   token.endswith(re.sub('[^a-z0-9]','',key[5:].lower())) and
                   ('site' in token or token==re.sub('[^a-z0-9]','',key[5:].lower()))]
            if len(sites)==1:found=sites[0]
        result.append(found)
    return len(result)==len(expected) and None not in result and len(set(result))==len(result) and set(result)==set(expected)


def external_metadata_valid(evaluation,rows):
    if evaluation.get('external_metrics')!='pending_host_evaluation':return False
    external=[r for r in rows if r['split']=='external']
    for key,value in evaluation.items():
        if 'external' not in key or key=='external_metrics':continue
        if key in ('n_external','n_external_participants','external_n','external_count'):
            if isinstance(value,bool) or not isinstance(value,int) or value!=len(external):return False
        elif key in ('external_site','external_sites'):
            if set(value if isinstance(value,list) else [value])!={r['site'] for r in external}:return False
        elif key=='external_participant_ids':
            if not isinstance(value,list) or len(value)!=len(external) or participant_ids(value)!={r['participant_id'] for r in external}:return False
        else:return False
    return True


def step_records(value):
    rows=value if isinstance(value,list) else value['steps']
    actions={'execute':'execute','run':'execute','recompute':'execute','computed':'execute','recomputed':'execute',
             'reuse':'reuse','reused':'reuse','cached':'reuse'}
    result=[]
    for row in rows:
        name=next((row[key] for key in ('name','step','step_id') if key in row),None)
        if name is None or row.get('action') not in actions:raise ValueError('invalid_step_record')
        result.append({'name':name,'action':actions[row['action']]})
    return result


def dependency_records(value):
    records=value['steps'] if isinstance(value.get('steps'),dict) else value
    result={name:dict(row) for name,row in records.items()}
    for row in result.values():
        config=row.get('config')
        if isinstance(config,dict) and set(config)=={'threshold','threshold_source'} and config['threshold_source']=='inputs/config.json':
            row['config']={'threshold':config['threshold']}
    if 'group' in result and default_group_config(result['group'].get('config')):
        result['group']['config']={}
    return result


def default_group_config(value):
    return value=={} or value=={'aggregation':'mean_of_participant_means_and_sum_of_counts','metrics':'all'}


def model_state_equivalent(first,second,atol=5e-6,rtol=5e-5):
    import numpy as np
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        names=list(map(str,a['feature_ids']));other=list(map(str,b['feature_ids']))
        if len(names)!=len(set(names)) or len(other)!=len(set(other)) or set(names)!=set(other):return False
        order=[other.index(name) for name in names]
        return all(np.allclose(a[key],b[key][order],atol=atol,rtol=rtol,equal_nan=False) for key in ('mean','scale','coef')) and bool(np.allclose(a['intercept'],b['intercept'],atol=atol,rtol=rtol,equal_nan=False))


def density_edge_count(nodes):
    """ceil(0.1 * choose(n,2)) in exact decimal/integer arithmetic."""
    pairs=nodes*(nodes-1)//2
    return (pairs+9)//10
