"""Materialize bounded copies; never modify source cohorts or historical packets."""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .common import *


def build(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    evidence = read_json(ROOT / 'dev/runs/benchmark_hard_design_20261020/READINESS.json')
    rows = read_csv(ROOT / evidence['phenotype']['path'])
    meta = {r['SUB_ID']: r for r in rows}
    pairs = {}
    source_bindings = {}
    for entry in evidence['payloads']:
        source = ROOT / entry['path']
        if sha(source) != entry['sha256']:
            raise ValueError('audited_source_changed')
        source_bindings[entry['path']] = entry['sha256']
        pairs.setdefault(entry['subject_id'], {})[entry['atlas']] = entry
    participants = []
    for sid, pair in sorted(pairs.items()):
        row = meta[sid]
        participants.append(dict(participant_id=sid, site=row['SITE_ID'],
            label=int(row['DX_GROUP'] == '1'), age=float(row['AGE_AT_SCAN']),
            sex=int(row['SEX']), mean_fd=float(row['func_mean_fd'])))
    small = []
    for site in sorted({r['site'] for r in participants}):
        for label in (0, 1):
            small += [r for r in participants if r['site'] == site and r['label'] == label][:3]
    payload = {}
    roi_ids = {}
    for sid, pair in pairs.items():
        payload[sid] = {}
        roi_ids[sid] = {}
        for atlas, entry in pair.items():
            source = ROOT / entry['path']
            header = source.read_text(encoding='utf-8-sig').splitlines()[0]
            ids = [s.lstrip('#').replace('Mean_', '') for s in header.split()]
            a = np.loadtxt(source)
            if len(ids) != a.shape[1] or len(ids) != len(set(ids)):
                raise ValueError('invalid_source_roi_header')
            payload[sid][atlas] = a
            roi_ids[sid][atlas] = ids
    specs = {r['case_id']: r for r in read_json(ROOT / 'neurobench/designs/execution_tasks.json')['tasks']}
    index = []
    private = destination / 'private'
    private.mkdir()
    transformations = []

    def new(case_id, protocol, required):
        path = destination / 'public/cases' / case_id
        inp = path / 'inputs'
        inp.mkdir(parents=True)
        write_json(inp / 'protocol.json', {
            'data_scope': 'Development derivatives of public ABIDE ROI data or supplied atlas data. Controlled record/layout/grid variants are engineering fixtures, not new acquisitions.',
            'entrypoint': {'path': 'project/main.py', 'arguments': '--input INPUT_DIRECTORY --output OUTPUT_DIRECTORY',
                'environment': 'Python 3.11, numpy, scipy, nibabel, pandas, networkx, scikit-learn, statsmodels; CPU only; no package installs or external access.',
                'runtime_module': 'The common host supplies runtime.run_step; its API is documented in context/RUNTIME.txt. The same interface is supplied during clean replay.'},
            'comparison': 'Same inputs, protocol, optional method library and execution interface in both conditions.',
            'budget': {'task_seconds': 900, 'model_rounds': 48, 'tool_calls': 192, 'python_call_seconds': 120,
                       'max_public_submissions': 2, 'post_terminal_retries': 0, 'max_managed_estimator_fits': 64},
            'numerical_tolerance': {'absolute': 5e-6, 'relative': 5e-5},
            **protocol,
        })
        return path, inp, required

    def seal(case_id, path, required):
        spec = specs[case_id]
        (path / 'task.md').write_text(spec['task'] + '\n', encoding='utf-8')
        contract = {'case_id': case_id, 'title': spec['title'], 'family_id': {
            'H01': 'cohort_integrity', 'H02': 'connectome_analysis', 'H03': 'prediction',
            'H05': 'statistical_inference', 'H07': 'image_space', 'H09': 'workflow_execution',
            'H10': 'workflow_execution', 'H11': 'cohort_integrity'}[case_id],
            'required_outputs': required + ['project/main.py'], 'input_sha256': file_map(path / 'inputs'),
            'public_protocol': 'inputs/protocol.json', 'expected_status': 'complete',
            'output_formats': 'Schemas and scientific definitions are in protocol.json. Output ordering is free; identities, not row positions, define alignment.'}
        write_json(path / 'CONTRACT.json', contract)
        index.append({'case_id': case_id, 'task_sha256': sha(path / 'task.md'), 'contract_sha256': sha(path / 'CONTRACT.json')})

    def table(inp, selected, filename='cohort.csv'):
        write_csv(inp / filename, selected, list(participants[0]))

    def series(inp, selected, atlases=('aal',), permute=True):
        listing = []
        for r in selected:
            sid = r['participant_id']
            for atlas in atlases:
                x = payload[sid][atlas]
                ids = roi_ids[sid][atlas]
                order = np.random.default_rng(int(sid) + len(atlas)).permutation(x.shape[1]) if permute else np.arange(x.shape[1])
                rel = f'series/{sid}_{atlas}.npy'
                (inp / 'series').mkdir(exist_ok=True)
                np.save(inp / rel, x[:, order], allow_pickle=False)
                write_json(inp / rel.replace('.npy', '.json'), {'participant_id': sid, 'atlas': atlas, 'roi_ids': [ids[i] for i in order]})
                listing.append({'record_id': f'R{len(listing):04d}', 'participant_id': sid, 'atlas': atlas, 'path': rel})
        return listing

    fc_definition = {'estimand': 'Pearson correlation over time between ROI columns. Constant ROI has peak-to-peak <=1e-12 and is invalid. Nonfinite series are unsupported.',
        'roi_order': 'Canonical output order is ascending numeric ROI ID from each series sidecar; do not assume stored columns follow that order.',
        'matrix_policy': 'Keep all ROI IDs. Set invalid rows/columns to zero including their diagonal. Valid diagonal is 1. Return valid boolean mask in this same ROI order.',
        'npz_schema': {'fc': 'R by R finite float matrix', 'roi_ids': 'R Unicode strings', 'valid': 'R booleans'},
        'mean_fc': 'Arithmetic mean of valid strict-upper-triangle coefficients.'}

    # H01: duplicate/conflicting identity records and an unresolved acquisition.
    path, inp, required = new('H01', {'identity': 'Participant key is participant_id in this ABIDE baseline namespace; repeated identical phenotype or acquisition records do not add participants.',
        'eligibility': 'A unique consistent phenotype (site,label,age,sex,mean_fd), age>0, sex in {1,2}, mean_fd>=0, and existing readable AAL+CC200 series are required. Conflicting phenotype means uncertain; absent pairing means excluded. Acquisitions without phenotype are uncertain.',
        'cohort_columns': ['participant_id','site','label','age','sex','mean_fd','aal','cc200'],
        'audit_schema': {'records': 'One object per source row in phenotype.csv and acquisitions.csv: source,record_id,participant_id,status,reason. status is included/excluded/uncertain. Include all duplicate source rows in audit; included participants occur once in cohort.csv.'}}, ['cohort.csv','audit.json'])
    phen = [{**r, 'record_id': f'P{i:04d}'} for i, r in enumerate(participants)]
    phen.append({**phen[0], 'record_id': 'P0128'})
    phen.append({**phen[1], 'label': 1-phen[1]['label'], 'record_id': 'P0129'})
    listing = series(inp, participants, ('aal','cc200'))
    listing = [r for r in listing if not (r['participant_id']==participants[2]['participant_id'] and r['atlas']=='cc200')]
    for suffix in ('.npy','.json'):
        (inp/f'series/{participants[2]["participant_id"]}_cc200{suffix}').unlink()
    listing.append({**listing[6], 'record_id': 'R0256'})
    listing.append({'record_id':'R0257','participant_id':'599991','atlas':'aal','path': listing[0]['path']})
    write_csv(inp/'phenotype.csv', list(reversed(phen)), ['record_id']+list(participants[0]))
    write_csv(inp/'acquisitions.csv', listing[::2]+listing[1::2], ['record_id','participant_id','atlas','path'])
    transformations += [{'case_id':'H01','kind':'controlled duplicate/conflicting phenotype, absent acquisition and unsupported identity','affected':[participants[i]['participant_id'] for i in range(3)]}]
    seal('H01',path,required)

    # H02: paired real multiscale ROI derivatives with independent storage permutations.
    path, inp, required = new('H02', {'connectivity':fc_definition,
        'comparison_columns':['participant_id','atlas','valid_rois','n_edges','mean_fc','global_transitivity'],
        'network': 'For valid ROIs, select up to ceil(pair_count/10) strictly positive edges, where pair_count=n_valid*(n_valid-1)/2 is an integer and division/ceiling use exact decimal arithmetic (116 valid nodes -> 667 edges). Descending correlation, exact ties broken by ascending pair of numeric ROI IDs. Undirected binary graph, all valid ROIs retained. Global transitivity = 3*triangles / connected unordered neighbor triples, or 0 when denominator is zero.',
        'deliveries':'connectivity/<participant_id>_<atlas>.npz for both atlases of every cohort row; comparison.csv; qc.json with participant/atlas invalid ROI ID lists.'},['comparison.csv','qc.json'])
    table(inp,small[::-1]);series(inp,small,('aal','cc200'))
    seal('H02',path,required)

    # Shared fixed feature identities; transformed missing entries are documented privately.
    edge_pairs = list(itertools.combinations(range(12),2))[::2][:24]
    features = []
    for r in participants:
        x=payload[r['participant_id']]['aal']
        x=x[:,np.argsort([int(i) for i in roi_ids[r['participant_id']]['aal']])]
        z=x-x.mean(axis=0); norms=np.sqrt((z*z).sum(axis=0)); norms[norms==0]=1
        corr=(z.T@z)/np.outer(norms,norms)
        features.append([float(np.arctanh(np.clip(corr[i,j],-.999999,.999999))) for i,j in edge_pairs])
    features=np.asarray(features)
    feature_ids=[f'edge_{i:02d}' for i in range(24)]
    canonical_ids=sorted(roi_ids[participants[0]['participant_id']]['aal'],key=int)
    feature_defs=[{'feature_id':name,'roi_i':canonical_ids[i],'roi_j':canonical_ids[j]} for name,(i,j) in zip(feature_ids,edge_pairs)]

    path, inp, required = new('H03', {'target':'label=1 corresponds to source DX_GROUP=1, label=0 to DX_GROUP=2. Each row is one independent participant. External-site labels are unavailable to the candidate.',
        'feature_schema':'features.csv has participant_id and named connection features; align by participant and feature names. Values are Fisher-z transformed correlations; feature_definitions.json states identities.',
        'candidate_models': {'families':['logistic_l2','ridge'], 'hyperparameters':[0.1,1.0,10.0],
            'definitions':'logistic_l2 uses L2 logistic regression C; ridge uses RidgeClassifier alpha. Intercept fitted. Use equal sample weights and deterministic solves.'},
        'validation':'Leave one development site out per validation fold. Model choice maximizes unweighted mean of site balanced accuracy across all 6 family/hyperparameter candidates. Any exact tied maximum is allowed. External-site labels and features cannot affect any fitted preprocessing or model choice.',
        'preprocessing_state':'Featurewise arithmetic means from fitting rows replace nonfinite entries; scale by fitting-row population standard deviation after imputation, replacing zero scale by 1. Means/scales are fitted separately for every validation fold and the final development fit.',
        'prediction_schema':'predictions.csv: participant_id,split,score,prediction. score is logistic decision_function or ridge decision_function, prediction is int(score>=0). Include all development participants with their out-of-site fold predictions from the chosen candidate, plus external predictions.',
        'model_npz_schema':'model/state.npz: feature_ids Unicode strings,mean,scale,coef vectors,intercept scalar; values describe the final development fit.',
        'evaluation_schema':'evaluation.json: family,hyperparameter,mean_site_balanced_accuracy,external_metrics="pending_host_evaluation". fit_evidence.json: final_fit_participant_ids,validation_folds [{site,fit_ids,validation_ids}],candidate_scores [{family,hyperparameter,mean_site_balanced_accuracy}]. No minimum accuracy is required.'},['predictions.csv','evaluation.json','model/state.npz','fit_evidence.json'])
    test_site=sorted({r['site'] for r in participants})[-1]
    cohort=[{**r,'split':'external' if r['site']==test_site else 'development','label':'' if r['site']==test_site else r['label']} for r in participants]
    write_csv(inp/'cohort.csv',cohort[::-1],list(cohort[0]))
    order=np.random.default_rng(151).permutation(24)
    frows=[{'participant_id':r['participant_id'],**{feature_ids[j]:float(features[i,j]) for j in order}} for i,r in enumerate(participants)]
    write_csv(inp/'features.csv',frows[::2]+frows[1::2],['participant_id']+[feature_ids[j] for j in order])
    write_json(inp/'feature_definitions.json',feature_defs)
    write_json(private/'H03_EXTERNAL_LABELS.json',{r['participant_id']:r['label'] for r in participants if r['site']==test_site})
    seal('H03',path,required)

    path, inp, required = new('H05', {'estimand':'Each specified Fisher-z connection is the outcome in a participant-level linear model. Target is adjusted coefficient for diagnosis label 1 versus 0.',
        'covariates':['intercept','label','age','sex indicator (sex==2)','mean_fd','site categorical indicators with lexically first site as baseline'],
        'eligibility':'Use unique participants with finite outcomes and all finite numeric covariates; sex in {1,2}. Audit every cohort row included/excluded. Connections are one multiple-comparison family, not independent participants.',
        'uncertainty':'OLS coefficients with HC3 heteroscedasticity-consistent standard errors. Two-sided standard-normal p values and 95% normal confidence intervals. Benjamini-Hochberg q values across all 12 specified label coefficients. Full design rank is required.',
        'estimates_columns':['feature_id','beta','se','ci_low','ci_high','p','q'],
        'design_schema':'design.json: participant_ids,columns,n,rank. sample_audit.csv: participant_id,status,reason. report.json: n,feature_count,significant_feature_ids (q<0.05); no positive finding required.'},['estimates.csv','design.json','sample_audit.csv','report.json'])
    cohort=[dict(r) for r in participants]
    cohort[3]['age']='';cohort[45]['mean_fd']=''
    write_csv(inp/'cohort.csv',cohort[::2]+cohort[1::2],list(cohort[0]))
    # Reverse records without changing their participant-feature identity.
    write_csv(inp/'connectivity.csv',[{'participant_id':r['participant_id'],**{feature_ids[j]:features[i,j] for j in range(12)}} for i,r in reversed(list(enumerate(participants)))],['participant_id']+feature_ids[:12])
    write_json(inp/'feature_definitions.json',feature_defs[:12])
    transformations.append({'case_id':'H05','kind':'controlled missing covariates','participants':[cohort[3]['participant_id'],cohort[45]['participant_id']]})
    seal('H05',path,required)

    path,inp,required=new('H07', {'space':'All inputs are registered atlas derivatives in the same declared MNI world coordinates in millimetres, not patient acquisitions. No registration estimation is requested.',
        'sampling':'Output grid is exactly reference.nii.gz. Labels are categorical nearest-neighbor samples; values are continuous trilinear samples. World transform is identity. Constant zero outside support; values inside support retain their physical numeric values.',
        'measurements_columns':['roi_id','voxel_count','volume_mm3','mean_value'],
        'measurements':'One row per positive label actually present after alignment. volume_mm3 uses absolute affine determinant; mean_value averages aligned values over the aligned ROI.',
        'outputs':'images/labels.nii.gz, images/values.nii.gz, measurements.csv; spatial_audit.json gives reference_shape,reference_affine,voxel_volume_mm3,space="MNI",units="mm".'},['images/labels.nii.gz','images/values.nii.gz','measurements.csv','spatial_audit.json'])
    (inp/'images').mkdir()
    source=ROOT/'data/atlas/aal_116/atlas.nii.gz';a=nib.load(source)
    label=a.as_reoriented(np.array([[1,1],[0,-1],[2,1]]))
    nib.save(label,inp/'images/labels.nii.gz')
    source_bindings[str(source.relative_to(ROOT))]=sha(source)
    source=ROOT/'data/atlas/msdl_39/atlas.nii.gz';v=nib.load(source)
    val=nib.Nifti1Image(np.asarray(v.dataobj[...,0],dtype=np.float32),v.affine)
    val=val.as_reoriented(np.array([[0,-1],[2,1],[1,-1]]))
    val.header.set_xyzt_units('mm');nib.save(val,inp/'images/values.nii.gz')
    source_bindings[str(source.relative_to(ROOT))]=sha(source)
    affine=np.array([[3.5,.4,0,-96],[0,3.3,.2,-132],[.15,0,3.2,-71],[0,0,0,1]])
    ref=nib.Nifti1Image(np.zeros((56,70,61),dtype=np.uint8),affine);ref.header.set_xyzt_units('mm');nib.save(ref,inp/'reference.nii.gz')
    write_json(inp/'spatial_metadata.json',{'space':'MNI','units':'mm','label_semantics':'categorical AAL region IDs','value_semantics':'first supplied MSDL atlas component','source_to_reference_world_transform':np.eye(4).tolist()})
    transformations.append({'case_id':'H07','kind':'axis reorientation preserving original world samples, first MSDL component, controlled nonorthogonal reference grid'})
    seal('H07',path,required)

    path,inp,required=new('H09',{'connectivity':fc_definition,
        'research_outputs':'connectivity/<participant_id>.npz with fc,roi_ids,valid; summary.csv: site,label,n,mean_fc (mean of participant mean_fc). execution.json: entrypoint and analyzed_participant_ids.',
        'repair_scope':'The supplied project is editable by copying it to output/project/. Equivalent valid implementations are accepted. Final project/main.py must work in a fresh host workspace with only bound inputs and delivered project files.'},['summary.csv','execution.json'])
    table(inp,small[::-1]);series(inp,small)
    (inp/'project').mkdir()
    (inp/'project/main.py').write_text('import argparse\nfrom connectomes import analyze\np=argparse.ArgumentParser();p.add_argument("--input");p.add_argument("--output");a=p.parse_args()\nanalyze(a.input,a.output)\n',encoding='utf-8')
    (inp/'project/connectome.py').write_text('from pathlib import Path\nimport csv, numpy as np\ndef analyze(inp,out):\n    inp,out=Path(inp),Path(out);out.mkdir(exist_ok=True,parents=True)\n    rows=list(csv.DictReader((inp/"cohort.csv").open()))\n    for r in rows:\n        x=np.load(inp/"series"/(r["participant_id"]+"_aal.npy"))\n        a=np.corrcoef(x)\n        np.save(out/(r["participant_id"]+".npy"),a)\n    (out/"summary.csv").write_text("site,label,n,mean_fc\\nALL,0,"+str(len(rows))+",0\\n")\n',encoding='utf-8')
    transformations.append({'case_id':'H09','kind':'predeclared project import/interface, correlation-axis and summary defects'})
    seal('H09',path,required)

    path,inp,required=new('H10',{'connectivity':fc_definition,
        'outputs':'results/<participant_id>_fc.npz (fc,roi_ids,valid), results/<participant_id>_metric.json (mean_fc,above_threshold), results/group.json (n,mean_fc,above_threshold_sum). dependencies.json maps step names to input paths/config. execution.json lists step names and actual intended actions.',
        'metric':'mean_fc as defined for connectivity; above_threshold counts valid upper-triangle correlations strictly greater than that participant configuration threshold. group averages participant means and sums counts.',
        'reuse':'Use content/config equivalence and recorded dependencies; preserve every still-valid prior artifact byte-for-byte. Recompute invalid artifacts and their dependents. Current inputs are in series/ and config.json; previous products and their bindings are in prior/.',
        'step_evidence':'Every FC, metric and group step must use runtime.run_step(name, inputs, outputs, config, code, reuse=False). inputs/outputs are workspace-relative paths. code is Python implementing that one step. reuse=True records existing outputs without executing code; copies of prior products may be staged first. The host records actual input/output hashes and action outside candidate output. This common primitive does not decide cache validity for you.',
        'step_names':'fc:<participant_id>, metric:<participant_id>, group. FC depends on series and its ROI sidecar; metric depends on that FC and its per-participant threshold; group depends on all metrics.'},['results/group.json','dependencies.json','execution.json'])
    selected=small[::2];table(inp,selected);series(inp,selected,permute=False)
    current={r['participant_id']:{'threshold':.2} for r in selected}
    old={k:dict(v) for k,v in current.items()};current[selected[1]['participant_id']]['threshold']=.35
    write_json(inp/'config.json',current);write_json(inp/'prior_config.json',old)
    from .reference import create_prior
    create_prior(inp, selected)
    sid=selected[0]['participant_id'];p=inp/f'series/{sid}_aal.npy';x=np.load(p);x[:,0]*=-1;np.save(p,x)
    transformations.append({'case_id':'H10','kind':'one real derivative column sign transformation and one threshold change with unchanged filenames','participants':[sid,selected[1]['participant_id']]})
    seal('H10',path,required)

    path,inp,required=new('H11',{'connectivity':fc_definition,
        'eligibility':'A record is supported only when exactly one matching metadata record exists, its series and ROI sidecar exist, series is finite 2D with at least 100 timepoints and at least two nonconstant ROIs, and ROI IDs are unique and match columns. Unknown/conflicting metadata is uncertain; absent, nonfinite, truncated or invalid-shape series is unsupported.',
        'status_columns':['record_id','participant_id','status','reason','artifact'],
        'status_vocabulary':['complete','unsupported','uncertain'],
        'deliveries':'Every record must appear once in status.csv. Supported records produce outputs/<record_id>.npz (fc,roi_ids,valid). Other records have empty artifact and nonempty evidence-based reason; do not invent replacements. summary.json: total,complete,unsupported,uncertain.'},['status.csv','summary.json'])
    selected=small[::2]+small[1:9:2];table(inp,selected);listing=series(inp,selected)
    records=[{'record_id':f'Q{i:03d}','participant_id':r['participant_id'],'path':r['path']} for i,r in enumerate(listing)]
    (inp/records[1]['path']).unlink()
    p=inp/records[3]['path'];np.save(p,np.load(p)[:30])
    p=inp/records[5]['path'];x=np.load(p);x[4,5]=np.nan;np.save(p,x)
    records.append({'record_id':'Q016','participant_id':'599992','path':records[0]['path']})
    write_csv(inp/'records.csv',records[::-1],['record_id','participant_id','path'])
    transformations.append({'case_id':'H11','kind':'controlled absent/truncated/nonfinite derivatives and unresolved identity','record_ids':['Q001','Q003','Q005','Q016']})
    seal('H11',path,required)

    write_json(destination/'INDEX.json',{'revision':REVISION,'status':'MATERIALIZED_QUALIFICATION_PENDING','cases':index})
    write_json(private/'PROVENANCE.json',{'source_bindings':source_bindings,'transformations':transformations,
        'participant_count':len(participants),'source_data_mutations':0,'eligibility':'Existing public-source local research derivatives; no new acquisition or clinical/generalization claim. Authored development fixtures.',
        'source_metadata':{'path':evidence['phenotype']['path'],'sha256':evidence['phenotype']['sha256']}})
    shutil.copytree(ROOT/'tmp/benchmark_high_continuation_20261016_r5/materials',destination/'public/library')
    return destination


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args()
    print(build(args.destination))
