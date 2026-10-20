"""Generate new synthetic inputs without reading or modifying research datasets."""
from __future__ import annotations

import json
import shutil
import numpy as np
import nibabel as nib
from .common import *

SEEDS = {'O01': 202610201401, 'O02': 202610201512, 'O03': 202610201623, 'O04': 202610201734}

SCORING = {
    'primary': 'Report each case validity and its scientific endpoints separately, plus valid-delivery counts per model/arm. Do not pool heterogeneous endpoint scales into a superiority score.',
    'O01': 'Validity requires all disclosed identity, probability, site-validation reproduction and unseen-label boundaries. Host-held external participant balanced accuracy and ROC AUC are continuous endpoints, never a validity threshold. No required model, candidate grid, tuning rule or positive result.',
    'O02': 'Validity requires all disclosed estimates, uncertainty and adjusted tests to reproduce the declared admissible analysis. No required significant association or preferred admissible method.',
    'O03': 'Validity requires complete evidence-based status and numerical measurements under the declared admissible interpolation.',
    'O04': 'Validity requires correct identity-linked numerical results and successful final-entrypoint execution; diagnosis prose is retained for review, not keyword-scored.',
    'common': 'On-time immutable submission, bound unchanged inputs/context, successful final program execution, and clean replay are required. Float absolute tolerance 5e-6, relative tolerance 5e-5. JSON key order, CSV row order, extra descriptive metadata, integer-versus-string identifiers that denote the same identifier, and equivalent numeric encodings are accepted. Missing/duplicate identities, nonfinite numerical answers, fabricated metrics, altered science or missing required fields fail. No method/library/tool-use bonus. Failed and unknown attempts remain in the denominator; never resend them.',
    'interpretation': 'One realization per slot, four authored synthetic development cases. Descriptive calibration only; neither preregistered clinical superiority nor independent expert acceptance. Runtime cost includes native calls, all in-session failed processes and all submissions; clean evaluation replay is reported separately.',
}


def build(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'private').mkdir()
    index = []

    def new(case_id, task, required, rules):
        root = destination/'public/cases'/case_id
        inp = root/'inputs'; inp.mkdir(parents=True)
        (root/'task.md').write_text(task+'\n', encoding='utf-8')
        protocol = {
            'revision': REVISION, 'case_id': case_id,
            'provenance': 'New fixed-seed synthetic development material, generated before inference; no real participant or clinical-validity claim.',
            'data_boundary': 'Read only supplied inputs/context and installed packages. Labels absent from public external data are held by the evaluator and must never be inferred as known truth. Source/evaluator/other runs/network are unavailable.',
            'execution': 'project/main.py --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY must rebuild every required output, including report.md, from supplied inputs and delivered project files in a fresh output directory. Reusable project files only; no workspace-specific absolute paths. CPU numpy/scipy/pandas/sklearn/statsmodels/nibabel/networkx available.',
            'required_outputs': required+['project/main.py','report.md'],
            'acceptance_and_scoring': SCORING,
            **rules,
        }
        write_json(inp/'protocol.json', protocol)
        return root, inp, protocol

    def seal(case_id, root, protocol):
        contract = {'case_id':case_id,'title':(root/'task.md').read_text(encoding='utf-8').strip(),
            'required_outputs':protocol['required_outputs'], 'input_sha256':file_map(root/'inputs'),
            'public_protocol':'inputs/protocol.json','expected_status':'complete',
            'output_formats':'protocol.json; additional metadata and equivalent representations accepted'}
        write_json(root/'CONTRACT.json',contract)
        index.append({'case_id':case_id,'contract_sha256':sha(root/'CONTRACT.json'),'task_sha256':sha(root/'task.md')})

    root, inp, protocol = new('O01',
        'Produce participant-level diagnostic probabilities for an unseen acquisition site and defensible out-of-site evidence for how those probabilities will transfer. Decide which supplied information can support that inference, choose and justify the learner, and state what remains unresolved.',
        ['predictions.csv','validation.csv','metrics.json','project/learner.py'], {
        'data_dictionary': 'train.csv has participant_id,site,visit,label and f00..f15. external.csv has the same columns except label. Two acquisitions per participant, stable identity. Features are synthetic connectivity-like measurements; metadata are not additional independent participants.',
        'learner_interface': 'project/learner.py must expose fit_predict(train, test): pandas DataFrames -> dict mapping every unique test participant_id to a finite probability of label=1 in [0,1]. Test has no label. Fit using only the passed train rows, never test values or other files; no file I/O in fit_predict. Fix internal random seeds. Any finite CPU learner and internal selection procedure is allowed. Aggregate acquisitions by a method of your choice. The evaluator independently calls this same function on the declared public site partitions and on reordered inputs.',
        'validation_boundary': 'For every development site, exclude all its participants from training, call fit_predict with remaining train rows and the held site without labels, and retain one score per participant. This is the external-site evaluation boundary, not a required learner or tuning procedure. Final external scores fit all development participants. Internal model/parameter decisions must use only the passed training participants, respecting participant identity.',
        'outputs': {'predictions.csv':['participant_id','score','prediction'],
            'validation.csv':['participant_id','fold','score','prediction'],
            'metrics.json':'validation_balanced_accuracy and external_metrics=null (external labels are host-held). Additional metadata accepted.'},
        'definitions': 'One prediction per participant; prediction=int(score>=0.5). fold is the omitted site. Validation balanced accuracy is computed over pooled participant-level out-of-site predictions. Evaluator reproduces validation scores from learner.py within float tolerance. It checks row-order and single-test-versus-batched inference invariance using the same training data. No threshold on hidden external accuracy, no single accepted method or model-state layout.',
        'additional_acceptance': 'report.md explains chosen method, validation evidence and limits. Only actual numerical evidence and published boundaries are machine graded; narrative requires later expert review.',
    })
    rng=np.random.default_rng(SEEDS['O01']); rows=[]; private_labels=[]
    for s, site in enumerate(['A','B','C','D','E']):
        for i in range(64 if s<4 else 80):
            sid=f'P{s+1}{i:04d}'; x=rng.normal(size=16)
            latent=1.6*x[0]*x[1]+.65*x[2]+.4*(x[3]**2-1)+rng.normal(scale=.6)
            label=int(latent>0)
            # Acquisition nuisance shifts are independent of the stable diagnostic mechanism.
            x[6:]+=((s-2)*.55)
            for visit in ['v1','v2']:
                obs=x+rng.normal(scale=.12,size=16)
                r={'participant_id':sid,'site':site,'visit':visit,'label':label}
                r.update({f'f{k:02}':float(v) for k,v in enumerate(obs)})
                rows.append(r)
            private_labels.append({'participant_id':sid,'site':site,'label':label})
    rng.shuffle(rows)
    write_csv(inp/'train.csv',[r for r in rows if r['site']!='E'],list(rows[0]))
    external=[{k:v for k,v in r.items() if k!='label'} for r in rows if r['site']=='E']
    write_csv(inp/'external.csv',external,list(external[0]))
    write_json(destination/'private/O01_LABELS.json',private_labels)
    seal('O01',root,protocol)

    root, inp, protocol = new('O02',
        'Using the supplied repeated observations, decide whether any candidate connectivity marker has a defensible participant-level association with diagnosis after accounting for the stated demographic and acquisition differences. Choose an analysis whose uncertainty is honest for these data, and state what the evidence does and does not support.',
        ['estimates.csv','method.json'], {
        'data_dictionary':'observations.csv contains participant_id,visit,site,label,age,sex,mean_fd and six markers m0..m5. Three acquisitions per participant; age, sex, site and diagnosis are participant-level. mean_fd can vary by acquisition. Data are synthetic.',
        'estimand':'For each marker separately, the diagnosis coefficient in an additive linear conditional-mean model with intercept, diagnosis (label), age, sex, mean_fd and categorical site. This defines the scientific target, not the computational solver. No interactions or outcome-derived covariate/model selection. Positive and negative estimates are equally valid.',
        'admissible_choices':'Equal credit is given to a participant-level analysis using participant means with HC3 covariance, or a record-level analysis using participant-cluster covariance with finite-sample CR1 correction G/(G-1)*(N-1)/(N-K). Either Wald normal-reference or t-reference inference is accepted (N-K df for participant_mean, G-1 for records), as are fdr_bh, holm or bonferroni for the six two-sided p-values. The public contract fixes the estimand and what counts as reproducible evidence, not a preferred solver; declaring a method outside this set is unsupported.',
        'method.json':{'unit':'participant_mean or records','covariance':'HC3 or cluster respectively','inference':'normal or t','adjustment':'fdr_bh or holm or bonferroni','claim_type':'association'},
        'estimates.csv':['marker','beta','se','p','q','ci_low','ci_high','decision'],
        'definitions':'95% two-sided confidence intervals and tests of diagnosis coefficient=0. q denotes the selected multiplicity-adjusted p-value for all six markers. decision is supported iff q<0.05, otherwise inconclusive. Report one row per marker with numeric values; no unreported significance threshold or required number of discoveries. All fields are independently recomputed for the chosen analysis. A causal claim is unsupported.',
    })
    rng=np.random.default_rng(SEEDS['O02']); observations=[]
    beta=np.array([0,1.4,-1.1,0,0,0]); site_effect=np.array([-.9,-.2,.3,1.1])
    for i in range(144):
        s=i%4; label=int(rng.random()<[.2,.4,.65,.8][s]); age=22+4*s+3*label+rng.normal(scale=3)
        sex=int(rng.random()<.45); base=rng.normal(scale=1.1,size=6)
        for v in range(3):
            fd=.08+.03*label+.01*s+abs(rng.normal(scale=.025))
            marks=base+beta*label+.025*age+.1*sex+2*fd+site_effect[s]+rng.normal(scale=.18,size=6)
            row={'participant_id':f'S{i:04d}','visit':v,'site':f'center{s}', 'label':label,'age':age,'sex':sex,'mean_fd':fd}
            row.update({f'm{k}':float(x) for k,x in enumerate(marks)});observations.append(row)
    rng.shuffle(observations);write_csv(inp/'observations.csv',observations,list(observations[0]));seal('O02',root,protocol)

    root, inp, protocol = new('O03',
        'Deliver atlas-space regional measurements for the supplied image collection. Decide which records provide enough spatial evidence to support a measurement, quantify only what the data support, and document why any record is unresolved or unsupported.',
        ['status.csv','measurements.csv','method.json'], {
        'data_dictionary':'atlas.nii.gz has positive integer region labels; atlas.json gives its coordinate frame. records.csv links image paths and coordinate-frame metadata. NIfTI spatial units and affine describe physical coordinates; a numeric voxel index is not a physical coordinate. No unlisted transform is available.',
        'target':'For each supported image, evaluate its scalar field at atlas voxel centers, then report the mean for every positive atlas label and its atlas-native voxel count and physical volume in mm^3. Spatial units must be known and converted consistently. Nearest and linear sampling are both accepted alternatives when declared and reproduced; the choice does not change record admissibility. Every region center must lie within the input field of view. Never interpolate region identities.',
        'status':'One row per record. complete means finite readable 3D image, known physical spatial units, established coordinate-frame correspondence and atlas-region coverage. uncertain means units or frame correspondence cannot be established with supplied evidence. unsupported means corrupt/unreadable, nonfinite or wrong-dimensional image. No fabricated registration or missing data.',
        'status.csv':['record_id','status','reason'],
        'measurements.csv':['record_id','region_id','voxel_count','volume_mm3','mean_value'],
        'method.json':{'interpolation':'nearest or linear'},
        'definitions':'Only complete records produce measurements, one per atlas region. Every status needs a nonempty evidence-based reason. Record IDs, region IDs, counts and physical/numerical answers are checked; free-text reasons are retained for review and are not keyword graded.',
    })
    rng=np.random.default_rng(SEEDS['O03']); atlas=np.zeros((12,14,10),np.int16)
    atlas[2:6,2:7,2:6]=3;atlas[7:10,3:10,2:8]=7;atlas[2:6,9:12,4:8]=11
    aff=np.diag([2.,2.5,3.,1.]);aff[:3,3]=[-10,-14,-12]
    a=nib.Nifti1Image(atlas,aff);a.header.set_xyzt_units('mm');nib.save(a,inp/'atlas.nii.gz')
    write_json(inp/'atlas.json',{'coordinate_frame':'template_calibration_v1'})
    (inp/'images').mkdir();records=[]
    shape=(24,24,24)
    for i in range(5):
        mm_aff=np.array([[1.5,0,0,-20],[0,1.5,0,-20],[0,0,1.5,-20],[0,0,0,1.]])
        if i==1:mm_aff=np.array([[0,-1.5,0,16],[1.5,0,0,-20],[0,0,1.5,-20],[0,0,0,1.]])
        ijk=np.indices(shape).reshape(3,-1).T; xyz=nib.affines.apply_affine(mm_aff,ijk)
        field=(2+.015*xyz[:,0]+.025*xyz[:,1]-.01*xyz[:,2]+.0004*xyz[:,0]*xyz[:,1]).reshape(shape)
        physical_aff=mm_aff.copy();units='mm'
        if i==1:physical_aff[:3,:]*=.001;units='meter'
        if i==2:units='unknown'
        if i==4:field=field.copy();field[0,0,0]=np.nan
        image=nib.Nifti1Image(field.astype(np.float64),physical_aff);image.header.set_xyzt_units(units)
        rel=f'images/I{i+1}.nii.gz';nib.save(image,inp/rel)
        records.append({'record_id':f'I{i+1}','path':rel,'coordinate_frame':'native_unregistered' if i==3 else 'template_calibration_v1'})
    write_csv(inp/'records.csv',records[::-1],list(records[0]));seal('O03',root,protocol)

    root, inp, protocol = new('O04',
        'Audit the supplied connectivity project and legacy summary. Decide which results are trustworthy, diagnose the causes of any unreliable results, deliver a repaired reusable project, and support the corrected results with executable evidence.',
        ['summary.csv','diagnosis.json'], {
        'data_dictionary':'records.csv links participant_id to series .npz and confounds .csv. The .npz stores signal (frames x ROI), frame_ids, roi_ids; confounds have frame_id,fd and six motion columns. Row positions and stored ROI order are not identities. The existing project and legacy_summary.csv are unverified inputs.',
        'scientific_target':'The public estimand is the Pearson correlation across retained, motion-adjusted ROI time courses for each participant. Use the supplied frame identities to apply the published fd<0.35 retention rule and the matched six motion columns as the nuisance model, adjusting for intercept plus those columns on retained frames. A ROI has no estimable time course when its centered nuisance-adjusted residual has Euclidean norm<=1e-8. Correlations touching such a ROI are zero, and a retained ROI diagonal is one; clip numerical roundoff to [-1,1]. Each delivered matrix identifies its actual ROI order. No other preprocessing or imputation. The repair and diagnosis are yours; this is the target definition, not an implementation recipe.',
        'outputs':'connectivity/<participant_id>.npz contains fc,roi_ids,valid. summary.csv contains participant_id,n_frames,n_valid_rois,mean_fc (mean of valid strict upper-triangle Pearson correlations). diagnosis.json is a nonempty list of objects with evidence and explanation fields documenting actual observed failures; no keyword rubric or required bug count. Extra fields are accepted.',
        'acceptance':'Every input participant appears exactly once in summary and has a correctly identified matrix. Numerical values, retained counts and valid masks are independently checked by identities, irrespective of output ROI/row order. Final project must also reproduce on a private reorder of confound rows and series columns with identities preserved. Diagnosis prose is retained for expert review; successful numbers and metamorphic replay provide execution evidence.',
    })
    rng=np.random.default_rng(SEEDS['O04']);(inp/'series').mkdir();(inp/'confounds').mkdir();records=[]
    for i in range(10):
        sid=f'R{i:03d}';frames=np.arange(160);roi=np.array([2,5,9,12,17,23,28,31,37,43,48,55])
        motion=rng.normal(size=(160,6));latent=rng.normal(size=(160,3));signal=latent@rng.normal(size=(3,12))+.45*rng.normal(size=(160,12))+motion@rng.normal(size=(6,12))
        if i%3==0:signal[:,-1]=2+motion@np.arange(6)*.2
        perm=rng.permutation(12);np.savez(inp/f'series/{sid}.npz',signal=signal[:,perm],frame_ids=frames,roi_ids=roi[perm])
        fd=rng.uniform(.02,.5,size=160);cr=[]
        for f in rng.permutation(160):
            row={'frame_id':int(f),'fd':float(fd[f])};row.update({f'm{k}':float(motion[f,k]) for k in range(6)});cr.append(row)
        write_csv(inp/f'confounds/{sid}.csv',cr,list(cr[0]));records.append({'participant_id':sid,'series':f'series/{sid}.npz','confounds':f'confounds/{sid}.csv'})
    write_csv(inp/'records.csv',records[::-1],list(records[0]));(inp/'project').mkdir()
    legacy='''import argparse,csv,json\nfrom pathlib import Path\nimport numpy as np\n\ndef analyze(inp,out):\n    inp,out=Path(inp),Path(out);(out/'connectivity').mkdir(parents=True,exist_ok=True)\n    rows=list(csv.DictReader((inp/'records.csv').open()));summary=[]\n    for row in rows:\n        data=np.load(inp/row['series']);x=data['signal']\n        conf=list(csv.DictReader((inp/row['confounds']).open()))\n        keep=np.array([float(c['fd'])<.35 for c in conf])\n        z=np.array([[1]+[float(c['m'+str(k)]) for k in range(6)] for c in conf])[keep]\n        residual=x[keep]-z@np.linalg.lstsq(z,x[keep],rcond=None)[0]\n        fc=np.nan_to_num(np.corrcoef(residual));ids=np.sort(data['roi_ids'])\n        np.savez(out/'connectivity'/(row['participant_id']+'.npz'),fc=fc,roi_ids=ids,valid=np.ones(len(ids),bool))\n        summary.append(dict(participant_id=row['participant_id'],n_frames=int(keep.sum()),n_valid_rois=len(ids),mean_fc=float(fc.mean())))\n    with (out/'summary.csv').open('w',newline='') as f:\n        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)\n\nif __name__=='__main__':\n    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args();analyze(a.input,a.output)\n'''
    (inp/'project/main.py').write_text(legacy,encoding='utf-8')
    write_csv(inp/'legacy_summary.csv',[{'participant_id':r['participant_id'],'mean_fc':0.31} for r in records],['participant_id','mean_fc'])
    seal('O04',root,protocol)

    # Reuse only the public optional library, identically available to both arms.
    library=ROOT/'dev/runs/benchmark_hard_trial_20261020_r1/packet/public/library'
    if not library.is_dir():raise FileNotFoundError('existing_optional_library_missing')
    shutil.copytree(library,destination/'public/library')
    write_json(destination/'INDEX.json',{'revision':REVISION,'cases':index})
    write_json(destination/'private/PROVENANCE.json',{
        'source':'New synthetic arrays and observations, no real source data read or written.', 'seeds':SEEDS,
        'source_bindings':{},'library_source':str(library),'library_hashes':file_map(library),
        'unexposed_before_dispatch':True,'clinical_or_independent_holdout_claim':False,
        'controlled_defects':'O03 unknown units, frame mismatch and nonfinite field; O04 positional confounds, correlation axis, ROI identity and invalid residual handling. These are not supplied as task hints.',
    })
    print(json.dumps({'packet':str(destination),'cases':CASE_IDS,'synthetic':True}),flush=True)
    return destination
