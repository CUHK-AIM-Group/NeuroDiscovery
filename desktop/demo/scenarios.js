/* Offline presentation fixtures. Never import the live research runtime. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DemoScenarios = api;
})(globalThis, function () {
  'use strict';
  const disclosure = 'Offline demonstration: responses, tool traces, scores and experiment results are scripted simulations. They do not represent live model calls, patient data or scientific validation.';
  const topic = 'Cerebellar–default mode network connectivity and attention symptoms in children with ADHD';
  const datasets = [
    {id:'adhd200', name:'ADHD-200', modalities:'T1w · rs-fMRI · Phenotypes',
      source:'https://fcon_1000.projects.nitrc.org/indi/adhd200/',
      downloaded:'NIfTI images and phenotype CSV files organized by site and participant. Raw images still require registration, denoising and connectivity matrix construction.',
      pipeline:'BIDS organization → T1 segmentation and registration → BOLD motion correction and confound regression → AAL116 ROI time series including the cerebellum → Fisher-z FC → participant and site splits',
      fixture:'12 synthetic participants; simplified T1 arrays, simulated 80×12 ROI time series and a phenotype CSV. The toy demo12 ROIs are not AAL116.',
      ready:'fc_z [12,12,12] · t1 [12,1,12,12,12]', models:'BrainGNN / BrainNetCNN / BNT', skill:'adhd200-skill'},
    {id:'abide', name:'ABIDE', modalities:'T1w · rs-fMRI · Phenotypes',
      source:'https://preprocessed-connectomes-project.org/abide/download.html',
      downloaded:'PCP preprocessed derivatives: ROI time series (.1D), images (.nii.gz) and phenotype CSV files. Preprocessing alone does not align labels or define training splits.',
      pipeline:'Verify pipeline, atlas and GSR settings → QC and phenotype ID alignment → time series to FC → training-fold normalization → independent-site evaluation',
      fixture:'12 synthetic participants with toy T1 arrays, ROI time series and ASD/control labels. This cohort is separate from ADHD-200.',
      ready:'fc_z [12,12,12] · t1 [12,1,12,12,12]', models:'BrainNetCNN / BNT / BrainGNN', skill:'abide-skill'},
    {id:'adni', name:'ADNI', modalities:'T1w · rs-fMRI · Clinical measures',
      source:'https://adni.loni.usc.edu/data-samples/adni-data/',
      downloaded:'Real ADNI data require approved IDA access. Images may be DICOM or NIfTI, with clinical tables exported separately. Match participant and visit IDs, and verify which modalities exist at each visit.',
      pipeline:'Convert DICOM if needed → match visits → BIDS → T1/BOLD preprocessing → ROI features → keep all visits from one participant in the same split',
      fixture:'12 synthetic participants, one visit each, with T1, ROI time series and MCI/control labels. No controlled ADNI records were downloaded.',
      ready:'fc_z [12,12,12] · t1 [12,1,12,12,12]', models:'CNN3D / BrainGNN / BNT', skill:'adni-skill'},
    {id:'seediv', name:'SEED-IV', modalities:'EEG · Emotion labels',
      source:'https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html',
      downloaded:'Official packages distinguish raw EEG (.mat) from extracted DE/PSD features. Eye tracking is also available; this workflow uses EEG. Precomputed features must not be filtered as raw waveforms.',
      pipeline:'Check sampling rate, channels and trials → resample, filter and inspect raw-waveform artifacts → 4-second windows → training-only normalization → group every window by participant',
      fixture:'12 synthetic participants and 24 segments; 62 channels at 200 Hz, 4 seconds each, four emotion classes. No real EEG or eye-tracking records are included.',
      ready:'epochs [24,62,800] · labels [24]', models:'temporal-models (TCN / Transformer)', skill:'seed-iv-skill'},
  ];
  const hypotheses = [
    {id:'H1', text:'In children with ADHD, cerebellar–default mode network connectivity improves attention-symptom prediction in outer cross-validation after controlling for age, sex, site and head motion.', novelty:100, structural:86, gnn:89, statistical:91, clinical:89, methodological:90, status:'Selected', type:'Substantive extension'},
    {id:'H2', text:'Under a fixed held-out-site protocol, a prespecified subset of cerebellar–default mode network connections generalizes more consistently for ADHD classification than whole-brain features.', novelty:100, structural:81, gnn:84, statistical:87, clinical:84, methodological:87, status:'Selected', type:'Substantive extension'},
    {id:'H3', text:'With identical participant splits and matched model capacity, adding T1 gray-matter volume features improves ADHD classification over rs-fMRI connectivity alone.', novelty:70, structural:88, gnn:90, statistical:90, clinical:91, methodological:86, status:'Selected', type:'Potential new relationship'},
    {id:'H4', text:'ADHD is associated with differences in default mode network functional connectivity.', novelty:0, structural:94, gnn:96, statistical:94, clinical:94, methodological:91, status:'Replication reference', type:'Established finding'},
    {id:'H5', text:'A cerebellar connectivity measure has the same ADHD prediction effect across all age groups.', novelty:20, structural:70, gnn:69, statistical:75, clinical:70, methodological:77, status:'More evidence needed', type:'Uncertain'},
    {id:'H7', text:'With fixed participant splits, the predictive benefit of cerebellar Crus I–posterior cingulate connectivity for inattention scores remains stable after age stratification.', novelty:70, structural:83, gnn:86, statistical:85, clinical:84, methodological:85, status:'Reserve', type:'Potential new relationship'},
    {id:'H8', text:'In leave-one-site-out evaluation, left–right asymmetry of cerebellar–default mode connectivity predicts attention symptoms better than mean connectivity alone.', novelty:70, structural:81, gnn:83, statistical:84, clinical:81, methodological:84, status:'Reserve', type:'Potential new relationship'},
    {id:'H9', text:'After training-fold site adjustment, the ADHD classification benefit of cerebellar–default mode network participation coefficients is consistent across head-motion thresholds.', novelty:70, structural:80, gnn:81, statistical:81, clinical:80, methodological:82, status:'Reserve', type:'Potential new relationship'},
  ].map(row => {
    const review = (row.statistical + row.clinical + row.methodological) / 3;
    return {...row, review, total: +(0.4 * row.novelty + 0.2 * row.structural + 0.2 * row.gnn + 0.2 * review).toFixed(2)};
  }).sort((a,b) => b.total - a.total);
  const h6={id:'H6',text:'After removing prespecified motion-sensitive connections, the association between cerebellar–default mode network connectivity and attention symptoms persists in independent validation.',novelty:70,structural:84,gnn:87,statistical:87,clinical:89,methodological:92,review:268/3,total:80.07,status:'Validated (simulated)',type:'Potential new relationship'};
  const finalHypotheses=[...hypotheses.filter(h=>['H1','H2'].includes(h.id)).map(h=>({...h,status:'Validated (simulated)'})),h6].sort((a,b)=>b.total-a.total);
  // Presentation candidates only; do not read or alter the scientific graph/candidate pool.
  const poolFeatures=['Crus I–medial prefrontal connectivity','Crus II–posterior cingulate connectivity','cerebellar lobule VI–precuneus connectivity','left cerebellum–right angular gyrus connectivity','right cerebellum–left angular gyrus connectivity','mean vermis–default mode connectivity','bilateral cerebellar–default mode connectivity laterality','cerebellar–default mode between-module participation','the positive-to-negative cerebellar–default mode connectivity ratio','individual deviation in cerebellar–default mode connectivity','joint cerebellar gray-matter volume and default mode connectivity features','the interaction between cerebellar structure and default mode node connectivity'];
  const poolOutcomes=['inattention scores','hyperactivity/impulsivity scores','total symptom scores','ADHD versus control status','predominantly inattentive subtype','combined subtype','high attention-symptom burden'];
  const poolProtocols=['leave-one-site-out evaluation','participant-grouped cross-validation stratified by age','participant-grouped cross-validation stratified by sex','sensitivity analysis with strict motion thresholds','training-fold site adjustment','a prespecified capacity-matched ablation'];
  const pendingHypotheses=poolFeatures.flatMap(feature=>poolOutcomes.flatMap(outcome=>poolProtocols.map(protocol=>({feature,outcome,protocol}))))
    .slice(0,500-hypotheses.length).map(({feature,outcome,protocol},i)=>({
      id:'H'+(i+10),text:`In the ADHD-200 child cohort, ${feature} improves prediction of ${outcome} under ${protocol}, beyond a demographic-only baseline after controlling for age, sex, site and head motion.`,
      novelty:null,structural:null,gnn:55+(i*17)%35,statistical:null,clinical:null,methodological:null,review:null,total:null,
      status:'Awaiting detailed review',type:null,reviewStatus:'not_reviewed',validationStatus:'not_tested',
    }));
  const hypothesisPool=[...hypotheses.map(h=>({...h,reviewStatus:'reviewed',validationStatus:'not_tested'})),...pendingHypotheses];
  function fullHypothesisPool(options={}) {
    const success=options.fullOutcome!=='limit';
    const passed=new Set(success?finalHypotheses.map(h=>h.id):[]);
    return [...hypothesisPool,h6].map(h=>passed.has(h.id)?{...h,status:'Validated (simulated)',reviewStatus:'reviewed',validationStatus:'passed'}:
      ['H1','H2','H3','H6'].includes(h.id)?{...h,status:'Not validated',reviewStatus:'reviewed',validationStatus:'insufficient'}:h)
      .sort((a,b)=>(b.total??-1)-(a.total??-1));
  }
  const scoreNote = 'Balanced ranking: total = novelty × 40% + graph structure × 20% + GNN × 20% + mean of three reviewers × 20%. Scores are on a 0–100 scale. Novelty uses category priorities (100/70/20/0), not probabilities of correctness. Candidates must pass validity review before selection.';
  const definitions = [
  {
    "id": "chat",
    "name": "Chat",
    "mode": "off",
    "icon": "✦",
    "subtitle": "Meet your research assistant",
    "titles": {
      "zh": "认识 NeuroDiscovery",
      "en": "Getting to Know NeuroDiscovery"
    },
    "prompt": "Who are you, and what can you help me with?",
    "aliases": [
      "你是谁？你能做什么？",
      "你是谁？你能做什么？请介绍 NeuroDiscovery、NeuroOracle 和 NeuroRuntime，以及 Idea、Data、Experiment、Full 四种模式。"
    ]
  },
  {
    "id": "idea",
    "name": "Idea",
    "mode": "idea",
    "icon": "◇",
    "subtitle": "From graph evidence to research hypotheses",
    "titles": {
      "zh": "ADHD 小脑网络研究假设",
      "en": "ADHD Cerebellar Network Hypotheses"
    },
    "prompt": "Help me develop research hypotheses about cerebellar–default mode network connectivity and attention symptoms in children with ADHD. Show your workflow.",
    "aliases": [
      "我想研究 ADHD 儿童的小脑—默认模式网络与注意症状的关系。请帮我提出有价值的研究假设，并展示工作过程。",
      "请围绕“ADHD 儿童的小脑—默认模式网络功能连接与注意症状”构建 500 条候选 hypothesis 并进行全量筛查。可用数据是 ADHD-200 的 T1w、rs-fMRI 和表型。展示图谱搜索、GNN 评分、新颖性判断，以及统计、临床、方法学三个 agent 的重点评审过程。回复中展示 8 条已评审假设及各项分数，注明候选总数并提供完整表格，筛选 3 个候选并输出 /ideas/adhd_network/IDEA.md。",
      "请围绕“ADHD 儿童的小脑—默认模式网络功能连接与注意症状”生成研究 idea，展示图谱检索、GNN、新颖性和三个 agent 的评分，筛选 3 个 hypothesis 并按总分排序。",
      "请围绕“ADHD 儿童的小脑—默认模式网络功能连接与注意症状”提出 5 个可检验的 hypothesis。可用数据是 ADHD-200 的 T1w、rs-fMRI 和表型。展示图谱搜索、结构/GNN 评分、新颖性判断，以及统计、临床、方法学三个 agent 的评审过程。按总分从高到低给出假设和各项分数，筛选 3 个候选并输出 /ideas/adhd_network/IDEA.md。"
    ]
  },
  {
    "id": "data",
    "name": "Data",
    "mode": "data",
    "icon": "▦",
    "subtitle": "From input data to model-ready tensors",
    "titles": {
      "zh": "研究数据整理与质控",
      "en": "Research Data Preparation and Quality Control"
    },
    "prompt": "Please prepare the datasets in /data/demo and show your workflow.",
    "aliases": [
      "请帮我处理 /data/demo 中的数据，并展示工作过程。",
      "请处理 /data/demo 下的 adhd200、abide、adni、seediv 四套演示数据。先说明各自的模态、真实下载包的状态及离模型输入还差哪些步骤，再展示检查、处理、质控和划分过程。各队列独立保存，避免受试者泄漏，最后展示 model_ready 的张量形状和完整目录组织。",
      "请处理 /data/demo 下的 adhd200、abide、adni、seediv，说明数据模态和预处理步骤，展示处理过程及最终目录结构。"
    ]
  },
  {
    "id": "experiment",
    "name": "Experiment",
    "mode": "model",
    "icon": "⌘",
    "subtitle": "Model selection, implementation and experiments",
    "titles": {
      "zh": "ADHD 网络假设实验验证",
      "en": "Testing ADHD Network Hypotheses"
    },
    "prompt": "Use the data in /data/demo/adhd200 to test the hypotheses in /ideas/adhd_network/IDEA.md. Show your workflow and results.",
    "aliases": [
      "请用 /data/demo/adhd200 的数据验证 /ideas/adhd_network/IDEA.md 中的研究假设，并展示工作过程和结果。",
      "请读取 /ideas/adhd_network/IDEA.md 和 /data/demo/adhd200/model_ready，验证其中 H1–H3。根据项目模型库选择 BrainGNN、BrainNetCNN、BNT 及必要基线，展示模型代码、超参数、训练和验证集调参过程。使用预先固定的受试者/站点划分，测试集只在最终方案冻结后评估一次。输出结果表、假设验证状态和 /experiments/adhd_network 下的产物。",
      "请根据 /ideas/adhd_network/IDEA.md，使用 /data/demo/adhd200/model_ready 完成模型选择、代码编写、调参和实验，输出结果表。"
    ]
  },
  {
    "id": "full",
    "name": "Full",
    "mode": "end_to_end",
    "icon": "∞",
    "subtitle": "From research question to iterative validation",
    "titles": {
      "zh": "ADHD 小脑网络探索与验证",
      "en": "ADHD Network Discovery and Validation"
    },
    "prompt": "Explore and test hypotheses about cerebellar–default mode network connectivity and attention symptoms in children with ADHD, using /data/demo/adhd200. Show your workflow and results.",
    "aliases": [
      "我想研究 ADHD 儿童的小脑—默认模式网络与注意症状的关系，数据在 /data/demo/adhd200。请帮我探索并验证有价值的研究假设，展示工作过程和结果。",
      "研究 topic 是“ADHD 儿童的小脑—默认模式网络功能连接与注意症状”，数据位于 /data/demo/adhd200。请依次完成 Idea、Data、Experiment，并检查是否得到有价值且验证通过的假设。若没有或少于 3 个，就根据失败原因继续循环；获得至少 3 个通过验证门槛的不同假设，或总循环达到 20 次时停止。不能把预测分数当验证，也不能反复查看同一测试集来调参。展示每轮决策及最终结果。",
      "请围绕“ADHD 儿童的小脑—默认模式网络功能连接与注意症状”，使用 /data/demo/adhd200 完成 Idea、Data、Experiment 和验证；不足 3 个有效且验证通过的假设就继续循环，达到 3 个或循环 20 次时停止。"
    ]
  }
];
  const results = [
    ['Demographic logistic baseline', '—', '0.642', '0.611', '0.596'],
    ['BrainNetCNN', '1e-3 / 0.3', '0.761', '0.715', '0.701'],
    ['BNT', '5e-4 / 0.2', '0.784', '0.737', '0.724'],
    ['BrainGNN (selected)', '1e-3 / 0.3', '0.809', '0.754', '0.743'],
    ['BrainGNN + T1 (ablation)', '1e-3 / 0.3', '0.814', '0.758', '0.747'],
  ];
  const resultHeaders = ['Model', 'Learning rate / Dropout', 'Validation AUROC', 'Final test AUROC', 'Final test balanced accuracy'];
  const modelCode = `# BrainNetCNN: connectivity classification and validation-set model selection.
from pathlib import Path
import json
import numpy as np
import torch
from models.brainnetcnn.net.brainnetcnn import BrainNetCNN

root = Path("data/demo/adhd200/model_ready")
fc_z = np.load(root / "fc_z.npy", allow_pickle=False)
# FC contract: store Fisher-z; restore Pearson correlations and clear the diagonal before model input.
x = torch.from_numpy(np.tanh(fc_z).astype("float32"))
x.diagonal(dim1=-2, dim2=-1).zero_()
y = torch.from_numpy(np.load(root / "labels.npy")).long()
split = json.loads((root / "splits.json").read_text())
train, valid = split["train"], split["validation"]
torch.manual_seed(42)
model = BrainNetCNN(n_roi=x.shape[-1], nclass=2, dropout=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                             weight_decay=1e-4)
best_loss, best_state = float("inf"), None
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x[train]), y[train])
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = torch.nn.functional.cross_entropy(model(x[valid]), y[valid])
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
# Evaluate the test set separately, once all hypotheses and settings are frozen.
torch.save({"state_dict": best_state}, "model.pt")
`;
  const config = {demo:true, executed:false, atlas:'demo12', real_study_atlas:'AAL116',
    candidates:['BrainGNN','BrainNetCNN','BNT'], seeds:[42,43,44], epochs:30, batch_size:8,
    learning_rates:[0.001,0.0005], dropout:[0.2,0.3], weight_decay:0.0001,
    early_stopping_patience:5, selection:'validation only', test_policy:'single final evaluation',
    full:{max_rounds:20, target_distinct_hypotheses:3, interim:'development nested validation; no test feedback'}};
  const {demo: _demo, executed: _executed, ...trainingConfig} = config;
  const datasetTree = `/data/demo/
├── MANIFEST.json                  # SHA-256 hashes and tensor shapes
├── README.md
├── load_demo.py                   # NumPy → Tensor; optional project .pt export
├── adhd200/
│   ├── source/                    # T1 / ROI time series / phenotype.csv
│   └── model_ready/
│       ├── fc_z.npy               # [12, 12, 12], Fisher-z
│       ├── t1.npy                 # [12, 1, 12, 12, 12], float32
│       ├── labels.npy             # [12], int64
│       ├── subjects.csv           # ID / site / label / split
│       ├── splits.json            # train 8 / validation 2 / test 2
│       └── qc.json                # Finite values, shapes, participant overlap
├── abide/                         # Separate source/ and matching model_ready/ layout
├── adni/                          # Separate source/ and matching model_ready/ layout
└── seediv/
    ├── source/epochs.npy          # EEG; 24 × 62 × 800
    └── model_ready/
        ├── epochs.npy            # Normalized using training participants only
        ├── labels.npy            # Four emotion classes
        ├── subjects.csv          # All windows from one person share a split
        ├── splits.json
        └── qc.json`;
  const graph = {
    nodes:[['adhd','ADHD',80,130],['cerebellum','Cerebellum',255,64],['dmn','Default mode network',425,100],['attention','Attention symptoms',570,180],['motion','Motion / site',255,250],['t1','T1 gray-matter volume',435,290]],
    edges:[['adhd','cerebellum','Co-occurrence'],['cerebellum','dmn','Candidate connection'],['dmn','attention','Association to test'],['motion','dmn','Confound control'],['t1','cerebellum','Multimodal evidence'],['adhd','attention','Phenotype']],
  };
  const message = text => ({kind:'message', text});
  const tool = (name, input, output, delay=1100) => ({kind:'tool', name, input, output, delay});
  function ideaEvents() {
    return [message(`I will focus on children with ADHD, resting-state connectivity and attention symptoms. I will screen a pool of ${hypothesisPool.length} hypotheses, then review ${hypotheses.length} priority candidates for novelty and from three scientific perspectives. The complete pool will remain available as an artifact.`),
      tool('NeuroOracle · topic_evidence', {topic, graph:'Topic evidence subgraph', evidence:'Study sources and conditions'}, 'Retrieve ADHD, cerebellar, default mode and attention evidence while preserving population, measurements, direction and study conditions.'),
      {kind:'graph', graph},
      tool('Hypothesis · generate', {candidates:hypothesisPool.length, required:['population','predictor','outcome','comparison','falsification']}, `Generated ${hypothesisPool.length} candidates with stable IDs across connectivity features, clinical outcomes and evaluation protocols. Screen the entire pool.`),
      tool('GNN · score_paths', {model:'GraphSAGE + DistMult',pool_size:hypothesisPool.length,review_batch:hypotheses.length}, `Screened graph paths and GNN scores for ${hypothesisPool.length} candidates. Retain all candidates and prioritize ${hypotheses.length} for detailed review. An existing graph connection does not establish a novel hypothesis.`),
      tool('Novelty · classify', {policy:'reviewed-novelty-v2 / balanced',review_batch:hypotheses.length}, 'H1/H2: substantive extensions. H3/H7/H8/H9: potential new relationships. H4: replication reference. H5: insufficient evidence. Other candidates remain awaiting review.'),
      tool('Agent 1 · Statistical review', {perspective:'statistical',candidates:hypotheses.length}, `Reviewed ${hypotheses.length} priority candidates. H1 needs ΔR² and confidence intervals; H2 needs fixed held-out sites. Correct for multiple hypotheses and report all seeds, not just the best run.`,900),
      tool('Agent 2 · Clinical review', {perspective:'clinical',candidates:hypotheses.length}, `Reviewed ${hypotheses.length} priority candidates. Define the age range and symptom scale, distinguish symptom prediction from diagnosis, and require H3 to demonstrate benefit beyond additional model capacity.`,950),
      tool('Agent 3 · Methodological review', {perspective:'methodological',candidates:hypotheses.length}, `Reviewed ${hypotheses.length} priority candidates. Control motion and site, prevent participant or visit leakage, use paired splits and prespecified ablations, and keep observational associations separate from causal claims.`,1000),
      {kind:'scores',rows:hypotheses,pool:hypothesisPool,artifact:'hypotheses.csv',summary:`The pool contains **${hypothesisPool.length} candidates**, all screened. **${hypotheses.length} priority candidates** have completed detailed review and appear below in descending total-score order. The remaining **${hypothesisPool.length-hypotheses.length} candidates** and their screening status are in the complete table; unreviewed scores remain blank.`},
      message(`Select H1, H2 and H3. Keep H7/H8/H9 in reserve, H4 as a replication reference, and defer H5. All ${hypothesisPool.length} hypotheses are in hypotheses.csv. IDEA.md records falsification criteria, data requirements and open questions. Selection at the Idea stage is not experimental validation.`)];
  }
  function dataEvents() {
    return [message('I will inspect the modalities, input state and preparation needs of all four datasets, then organize preprocessing, quality control and splits for each cohort separately.'),
      {kind:'datasets'},
      tool('NeuroRuntime · inspect_data', {root:'/data/demo', datasets:datasets.map(d=>d.id)}, 'Check modalities, sampling rates, IDs, labels and missing values. MRI inputs contain 12 participants and 12 ROIs; EEG contains 12 participants and 24 windows.'),
      tool('Dataset skills · plan', {skills:datasets.map(d=>d.skill)}, 'Raw MRI follows BIDS → image preprocessing → ROI extraction. Check the existing pipeline for ABIDE PCP derivatives, and distinguish raw SEED-IV waveforms from precomputed features.'),
      tool('MRI · prepare_features', {atlas:'demo12', output:'float32 Fisher-z FC + T1 tensors'}, 'Compute Pearson and Fisher-z matrices from ROI time series; normalize T1 within each sample and align tensor shapes with label order.',1600),
      tool('EEG · prepare_epochs', {channels:62, sfreq:200, window_seconds:4}, 'Organize 4-second EEG segments and normalize channels using training-participant statistics. Output shape: [24,62,800].',1300),
      tool('QC · freeze_splits', {group:'subject_id', counts:'8 train / 2 validation / 2 test subjects'}, 'Check finite values, FC symmetry, label ranges and participant overlap. Reserve the two participants at site C for testing. Both EEG windows from one participant remain in the same split.'),
      {kind:'tree', text:datasetTree},
      message('Model inputs are ready: project adapters can provide dense FC to BrainNetCNN/BNT, and load_demo.py can export the .pt fields required by BrainGNN. T1 tensors support CNN3D; EEG epochs support temporal models.')];
  }
  function experimentEvents() {
    return [message('I will read IDEA.md and the dataset directory, match each research question to its model inputs, then implement and tune the models. Classification, symptom regression and modality ablations will be evaluated separately.'),
      tool('Experiment · inspect_contract', {idea:'/ideas/adhd_network/IDEA.md', data:'/data/demo/adhd200/model_ready'}, 'H1 → symptom regression; H2 → site generalization; H3 → FC/T1 ablation. Align ROI order and use tanh to restore correlations from Fisher-z features before model input.'),
      tool('Model registry · select', {candidates:['BrainGNN','BrainNetCNN','BNT']}, 'Use BrainGNN for node and edge interpretation, BrainNetCNN as a dense-connectivity baseline, and BNT for global network representations. Include a demographic baseline to assess confounding.'),
      {kind:'code', filename:'model_example.py', text:modelCode},
      {kind:'code', filename:'config.json', text:JSON.stringify(trainingConfig,null,2)},
      tool('Experiment · tune_validation', {trials:4, seeds:config.seeds, selection:'Validation AUROC; test data excluded from selection'}, 'Compare four learning-rate/dropout configurations and select by validation performance.',1500),
      {kind:'training'},
      tool('Experiment · evaluate_frozen', {test:'One evaluation after freezing', checks:['Baseline','Paired ablation','Confidence intervals','Multiple comparisons']}, 'Freeze candidates, settings and tests, then generate the final report.',1200),
      {kind:'results'},
      {kind:'validation', rows:[['H1','ΔR² = 0.08; 95% CI [0.02,0.13]; q = 0.03','Validated'],['H2','ΔAUROC = 0.05; 95% CI [0.01,0.09]; q = 0.04','Validated'],['H3','ΔAUROC = 0.004; 95% CI [-0.02,0.03]; q = 0.42','Insufficient evidence']]},
      message('Only H1 and H2 meet the validation criteria. The H3 improvement is small and its interval includes zero, so it remains unvalidated. Strong prediction or GNN scores cannot replace hypothesis testing. Full mode would continue from this feedback.')];
  }
  function decideLoop({round, acceptedIds, target=3, maxRounds=20}) {
    const count = new Set(acceptedIds).size;
    if (count >= target) return {stop:true, reason:'target_reached', count};
    if (round >= Math.min(20,maxRounds)) return {stop:true, reason:'max_rounds', count};
    return {stop:false, reason:'insufficient_validated', count};
  }
  const roundHeaders = ['Hypothesis / check','Comparison','Development results','Assessment'];
  function roundDetails(round, hardCase) {
    if(round===1) return {
      title:'Test the initial candidates and locate unstable results',
      plan:'Test three questions: whether cerebellar–default mode connectivity helps predict attention symptoms, whether a prespecified connection subset generalizes to held-out sites, and whether T1 adds useful information. Compare each candidate with its baseline using the same participant splits.',
      dataReview:'Check participant IDs, symptom scales and imaging labels. Apply the same age, sex and motion controls. Fit normalization within training folds, leave development sites out in turn, and keep the final test site sealed.',
      experiment:'Compare BrainGNN, BrainNetCNN, BNT and a demographic baseline using seeds 42, 43 and 44. Choose classification settings in inner validation and use a separate symptom-regression task. Summarize outer-fold results rather than the best run.',
      comparisons:hardCase ? [
        ['H1 · Symptom prediction','Demographic baseline vs added network connectivity','R² 0.180 → 0.204; ΔR² +0.024; 95% CI [-0.019, 0.067]; q = 0.23','Gain interval includes zero'],
        ['H2 · Cross-site classification','Whole-brain features vs prespecified connection subset','ΔAUROC +0.018; 95% CI [-0.014, 0.051]; q = 0.21; opposite directions at two development sites','Unstable across sites'],
        ['H3 · Multimodal ablation','Functional connectivity alone vs connectivity + T1','AUROC 0.809 → 0.814; Δ +0.005; 95% CI [-0.018, 0.027]; q = 0.41','Added benefit not established'],
      ] : [
        ['H1 · Symptom prediction','Demographic baseline vs added network connectivity','R² 0.180 → 0.250; ΔR² +0.070; 95% CI [0.020, 0.120]; q = 0.03','Supported in development'],
        ['H2 · Cross-site classification','Whole-brain features vs prespecified connection subset','AUROC 0.768 → 0.786; Δ +0.018; 95% CI [-0.014, 0.051]; q = 0.21','Unstable across sites'],
        ['H3 · Multimodal ablation','Functional connectivity alone vs connectivity + T1','AUROC 0.809 → 0.814; Δ +0.005; 95% CI [-0.018, 0.027]; q = 0.41','Added benefit not established'],
      ],
      finding:(hardCase ? '**H1: not counted as supported.** The mean predictive gain is positive, but its uncertainty interval still includes no improvement. This is insufficient to establish a benefit.' : '**H1: retain for final validation.** Adding network connectivity increases explained symptom variance from 0.180 to 0.250. The gain interval excludes zero and the corrected test meets the threshold. This supports further validation of symptom prediction, without establishing diagnostic utility.')+
        '\n\n**H2: the mean improvement does not generalize consistently.** Relative to whole-brain features, the subset improves AUROC by 0.052 at held-out development site A but reduces it by 0.016 at site B. The aggregate score hides this disagreement, so H2 does not yet pass.'+
        '\n\n**H3: added information from T1 is not established.** The improvement is only 0.005 AUROC and its interval includes zero. The fusion model also has more parameters, leaving information gain and model capacity unresolved.',
      nextStep:'Address H2 site instability while keeping the connection subset and holdout rules fixed: check training-fold normalization, confound control and site weighting. Add a capacity-matched single-modality control for H3. Keep the H1 research question and validation criteria frozen.',
    };
    if(round===2) return {
      title:'Follow up on site instability and model capacity',
      plan:'Reuse the first-round participant splits and connection subset to reassess H2 and H3. For H2, standardize training-fold normalization, age/sex/motion adjustment and site weighting. For H3, add a functional-connectivity model with comparable parameter count.',
      dataReview:'Reuse the quality-controlled matrices without selecting different participants or changing held-out sites. Estimate all adjustment coefficients from training data. Held-out development sites are evaluation-only, and the final test set remains sealed.',
      experiment:'Train paired whole-brain and fixed-subset BrainGNN models at each held-out development site using the same seeds and search range. Separately train capacity-matched FC-only and FC+T1 controls. Summarize site-level results and paired gain intervals.',
      comparisons:hardCase ? [
        ['H2 · Site follow-up','Whole brain vs fixed subset; consistent training pipeline','ΔAUROC +0.022; 95% CI [-0.012, 0.059]; q = 0.18','Generalization benefit remains uncertain'],
        ['H2 · Site-level check','Same model and subset; each development site held out','Site A: Δ +0.045; site B: Δ -0.001','A near-zero/negative result remains'],
        ['H3 · Capacity-matched ablation','Capacity-matched FC-only vs FC + T1','AUROC 0.807 → 0.811; Δ +0.004; 95% CI [-0.016, 0.026]; q = 0.45','Fusion benefit not established'],
      ] : [
        ['H2 · Site follow-up','Whole brain vs fixed subset; consistent training pipeline','AUROC 0.741 → 0.790; Δ +0.049; 95% CI [0.011, 0.083]; q = 0.04','Supported in development'],
        ['H2 · Site-level check','Same model and subset; each development site held out','Site A: Δ +0.055; site B: Δ +0.043','Consistent direction at both sites'],
        ['H3 · Capacity-matched ablation','Capacity-matched FC-only vs FC + T1','AUROC 0.807 → 0.811; Δ +0.004; 95% CI [-0.016, 0.026]; q = 0.45','Fusion benefit not established'],
      ],
      finding:(hardCase ? '**H2: remains unsupported.** The consistent pipeline improves the mean difference, but site B remains near zero and the overall interval includes zero. Cross-site stability is still unresolved.' : '**H2: the paired comparison supports further validation.** Both development sites now show positive gains. The overall gain is 0.049 and the corrected test meets the threshold. This evaluates subset versus whole-brain features within this round; absolute AUROCs from different training pipelines must not be directly subtracted across rounds.')+
        '\n\n**H3: stop tuning this direction.** With matched capacity, T1 still adds only 0.004 AUROC and the interval includes zero. Retain the unsuccessful result instead of changing splits or selecting favorable seeds to obtain significance.'+
        '\n\n**Next, investigate head motion.** The network findings still need to answer whether motion explains the observed association. This is a more informative next test than adding further modalities.',
      nextStep:'Register H6: does the cerebellar–default mode association with attention symptoms persist after removing prespecified motion-sensitive connections? Freeze removal rules, association tests and a stricter-motion sensitivity analysis before running the experiment. H3 remains insufficient.',
    };
    if(round===3) return {
      title:'Test whether the network association survives motion control',
      plan:'Introduce H6 to test dependence on motion-sensitive connections. Identify these connections within training folds and freeze their removal before evaluating the network–symptom association in held-out development data. Prespecify stricter motion thresholds and an equal-count connection-removal control.',
      dataReview:'Align motion measures and symptom scores by participant. Learn removal masks from training data only. Apply identical age, sex and site controls, and account for uncertainty from the smaller sample after stricter QC.',
      experiment:'Evaluate removal of motion-sensitive connections, stricter motion thresholds, and removal of an equal number of other connections using fixed seeds. Report standardized coefficients, intervals and corrected tests. These checks all support H6 and count as one hypothesis.',
      comparisons:hardCase ? [
        ['H6 · Primary test','Network–attention association after removing motion-sensitive connections','Standardized β = 0.090; 95% CI [-0.040, 0.220]; q = 0.19','Association remains uncertain'],
        ['H6 · Stricter motion control','Repeat the prespecified test with a stricter motion threshold','Standardized β = 0.050; 95% CI [-0.080, 0.180]; q = 0.38','Not maintained under stricter QC'],
        ['H6 · Equal-count removal','Remove an equal number of other connections with fixed seeds','Standardized β = 0.100; 95% CI [-0.030, 0.230]; q = 0.17','Specificity not established'],
      ] : [
        ['H6 · Primary test','Network–attention association after removing motion-sensitive connections','Standardized β = 0.240; 95% CI [0.090, 0.380]; q = 0.012','Supported in development'],
        ['H6 · Stricter motion control','Repeat the prespecified test with a stricter motion threshold','Standardized β = 0.220; 95% CI [0.050, 0.360]; q = 0.026','Direction retained; interval excludes zero'],
        ['H6 · Equal-count removal','Remove an equal number of other connections with fixed seeds','Standardized β = 0.230; 95% CI [0.080, 0.370]; q = 0.018','Association also survives other removals'],
      ],
      finding:hardCase ? '**H6: not supported yet.** After removing motion-sensitive connections, the effect interval includes zero. The estimate weakens under stricter motion control, and the equal-count control is also inconclusive. These results do not establish that motion has been ruled out.\n\nThe limitations of H1, H2 and H3 remain. This round clarifies the failure mechanisms without increasing the number of supported hypotheses.' :
        '**H6: the primary and sensitivity analyses agree in direction.** The standardized association is 0.240 after removing motion-sensitive connections and 0.220 under stricter motion control; both intervals exclude zero. The association also survives equal-count removal of other connections. It is not dependent on one removal scheme, but this neither eliminates all motion effects nor establishes causality.\n\n**Count one new hypothesis.** H1 concerns symptom-prediction gain, H2 cross-site classification, and H6 the persistence of association under motion control. The three H6 checks address one question and are not counted separately. Shared data also mean these tests are not statistically independent; the final analysis corrects across the candidate family.\n\n**H3 remains unvalidated.** This round does not establish a multimodal benefit.',
      nextStep:hardCase ? 'Continue bounded development checks for influential participants, site differences, QC thresholds and model complexity. Keep the final test set sealed rather than using it to search for improvement.' :
        'Three distinct research questions now have development support. Freeze the connection subset, motion-removal rules, preprocessing, model settings and test family. Evaluate the sealed test set once. Any candidate that fails final validation remains unvalidated; do not tune against that result.',
    };
    const probes=[
      ['Influential participants','Leave out influential participants in turn and recheck H1 prediction gains','H1 · Robustness','ΔR² +0.016; 95% CI [-0.022, 0.055]; q = 0.31','After removing influential participants, the gain is still indistinguishable from zero and cannot be assumed to generalize.'],
      ['Site heterogeneity','Recheck H2 training weights and calibration with fixed site holdouts','H2 · Generalization','ΔAUROC +0.014; 95% CI [-0.019, 0.048]; q = 0.29','Site-level performance still disagrees. Reweighting has not established a stable subset advantage.'],
      ['Motion control','Recheck the H6 direction and interval at prespecified motion thresholds','H6 · Association','Standardized β = 0.070; 95% CI [-0.050, 0.190]; q = 0.27','Uncertainty remains substantial under strict QC. A stricter threshold alone does not establish a positive association.'],
      ['Model complexity','Recheck H3 with stronger regularization and capacity-matched controls','H3 · Multimodal','ΔAUROC +0.003; 95% CI [-0.020, 0.025]; q = 0.51','After reducing overfitting risk, the additional T1 benefit remains unestablished. Repeated adjustments have not added support.'],
    ];
    const [title,comparison,hypothesis,result,feedback]=probes[(round-4)%probes.length];
    return {title:'Follow-up: '+title,plan:comparison+'. Keep earlier unsuccessful results and use development data only. Repeating a hypothesis does not increase the count.',
      dataReview:'Reuse quality-controlled data, frozen splits and training-fold processing. Compare with prior records to detect dependence on incidental settings.',
      experiment:comparison+'. Report the paired effect, interval and corrected test. The final test set remains unopened.',
      comparisons:[[hypothesis,comparison,result,'Insufficient evidence']],finding:feedback+'\n\nNo additional hypothesis meets the criteria, and the criteria have not been relaxed. Repeated observations of the same result are not new independent evidence.',
      nextStep:round===20?'Stop at the 20-round limit. Retain all unsuccessful results and settings. Seek additional matched participants and independent sites, or narrow the research question. Do not label the current candidates as validated.':'Continue the next development check guided by the recorded failure reasons. Stop at 20 rounds if the available data still cannot support a stable effect.'};
  }
  function roundDecision(event) {
    const state=event.verdict.reason==='target_reached'?'Development target reached. Freeze the protocol and proceed to final validation.':event.verdict.reason==='max_rounds'?'Reached the 20-round limit. Stop and retain the unvalidated status.':'Not enough supported hypotheses yet. Continue to the next round.';
    return `**Round ${event.round} decision**\n\n${event.finding}\n\n**Next step**\n\n${event.nextStep}\n\nDistinct hypotheses supported in development: **${event.verdict.count}/3**${event.accepted.length?' ('+event.accepted.join(', ')+')':''}. ${state}`;
  }
  function fullEvents(options={}) {
    const hardCase = options.fullOutcome === 'limit';
    const events = [message('Start the research cycle: seek three distinct hypotheses that meet the validation criteria, with a maximum of 20 rounds. Each round covers Idea → Data → Experiment → Review. Iteration uses nested development validation only; the final test set stays sealed until every analysis choice is frozen.'),
      tool('Full · set_protocol', {topic,target:3,max_rounds:20,mode:hardCase?'Persistent insufficient evidence':'Target reached in round 3'}, 'Fix participant splits, confound controls, test families and stopping rules. Repeated support for the same hypothesis does not increase the count.'),
      ...ideaEvents().slice(0,-1), ...dataEvents().slice(1,-1),
      {kind:'code', filename:'model_example.py', text:modelCode},
      {kind:'code', filename:'config.json', text:JSON.stringify(trainingConfig,null,2)},
      tool('Full · choose_and_tune', {candidates:config.candidates, split:'Nested development validation only'}, 'Select BrainGNN, BrainNetCNN, BNT and the demographic baseline. Tune within development data and prepare paired ablations.',1100),
      {kind:'training'}];
    const acceptedIds=[];
    for (let round=1; round<=20; round++) {
      if (!hardCase && round===1) acceptedIds.push('H1');
      if (!hardCase && round===2) acceptedIds.push('H2');
      if (!hardCase && round===3) acceptedIds.push('H6');
      const verdict=decideLoop({round,acceptedIds});
      events.push({kind:'round', round, phases:['Idea','Data','Experiment','Review'], delay:500,
        ...roundDetails(round,hardCase),accepted:[...acceptedIds], verdict});
      if(verdict.stop) break;
    }
    if(!hardCase) events.push(tool('Full · final_validation', {development_accepted:acceptedIds, test:'One final evaluation after all choices are frozen', corrections:'Correct across the iterative candidate family'}, 'H1, H2 and H6 retain support in the final test; H3 does not pass. Final-test failures remain unvalidated and are not used for further tuning.',1300),
      {kind:'results'},
      {kind:'validation',rows:[['H1','Symptom-prediction gain; ΔR² 0.08; corrected q 0.03','Final validation passed'],['H2','Held-out-site generalization; ΔAUROC 0.05; corrected q 0.04','Final validation passed'],['H6','Association after motion-sensitive edge removal; standardized β 0.21; corrected q 0.02','Final validation passed'],['H3','Multimodal gain interval includes zero','Not validated']]},
      {kind:'scores',rows:finalHypotheses,pool:fullHypothesisPool(options),artifact:'hypotheses_final.csv',title:'Full · Final validated hypothesis ranking',summary:`The initial pool contains ${hypothesisPool.length} candidates. Adding H6 during iteration brings the total to **${hypothesisPool.length+1} candidates**. Only the **${finalHypotheses.length} candidates** passing final validation appear below. The complete table retains every candidate, score and validation status.`},
      message('Round 3 completes the stopping criterion: H1, H2 and H6 pass final validation. The deliverables include hypotheses, the data contract, model code, result tables and detailed round-by-round feedback.'));
    else events.push(message(`Round 20: no candidate meets the validation criteria, so the cycle stops at its limit. The initial ${hypothesisPool.length} candidates plus H6 give ${hypothesisPool.length+1} candidates in hypotheses_final.csv. Retain unsuccessful results and document the additional data needed; none is presented as validated.`));
    return events;
  }
  function buildScenario(id,options={}) {
    const definition=definitions.find(s=>s.id===id);
    if(!definition) throw new Error('Unknown demo scenario');
    const events=id==='chat' ? [message('I am NeuroDiscovery, your neuroimaging research assistant. The selected model is GPT-6-Astra, and AutoResearch is off.'),
      tool('Client · describe_capabilities',{model:'GPT-6-Astra',autoresearch:'off'},'Summarize NeuroOracle, NeuroRuntime and the AutoResearch modes.',700),
      message('NeuroOracle supports graph evidence and hypothesis discovery. NeuroRuntime handles data and model workflows. We can discuss a research question or work within a specific research stage.'),
      {kind:'capabilities'}, message('Choose an AutoResearch scope in the model menu at the bottom right of the composer: Idea, Data, Model or Full. Then send your research question. You can start a new conversation, stop a response, regenerate it or export the conversation.')]
      : id==='idea'?ideaEvents():id==='data'?dataEvents():id==='experiment'?experimentEvents():fullEvents(options);
    return {...definition,events};
  }
  const csv = rows => rows.map(row=>row.map(cell=>'"'+String(cell??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');
  function hypothesisCsv(rows) {
    return '\uFEFF'+csv([['rank','id','hypothesis','novelty','structure','gnn','statistical','clinical','methodological','review_mean','total','status','review_status','validation_status','data_kind'],
      ...rows.map((h,i)=>[i+1,h.id,h.text,h.novelty,h.structural,h.gnn,h.statistical,h.clinical,h.methodological,h.review?.toFixed(2),h.total?.toFixed(2),h.status,h.reviewStatus||'reviewed',h.validationStatus||'not_tested','simulated'])]);
  }
  function roundArtifacts(rounds) {
    const out={},headers=['round',...roundHeaders,'evaluation_split','data_kind'];
    const rows=rounds.flatMap(event=>event.comparisons.map(row=>[event.round,...row,'development','simulated_not_measured']));
    for(const event of rounds) out[`round_${String(event.round).padStart(2,'0')}.csv`]='\uFEFF'+csv([headers,...rows.filter(row=>row[0]===event.round)]);
    out['round_metrics.csv']='\uFEFF'+csv([headers,...rows]);
    out['loop_history.json']=JSON.stringify({demo:true,executed:false,scientific_validation:false,scope:'development_rounds',final_test_used_for_iteration:false,rounds},null,2);
    const table=event=>[roundHeaders,roundHeaders.map(()=>'---'),...event.comparisons].map(row=>'| '+row.map(value=>String(value).replaceAll('|','\\|').replaceAll('\n',' ')).join(' | ')+' |').join('\n');
    out['REPORT.md']='# Full research workflow\n\n'+disclosure+'\n\n'+rounds.map(event=>`## Round ${event.round} experiment · ${event.title}\n\n${event.plan}\n\nData review: ${event.dataReview}\n\nExperiment settings: ${event.experiment}\n\n${table(event)}\n\n${roundDecision(event)}`).join('\n\n---\n\n')+'\n\nThese are development-round records. Final-test results never provide tuning feedback.';
    return out;
  }
  function artifacts(id, options={}) {
    const out={};
    const ranked=id==='full'&&options.fullOutcome!=='limit'?finalHypotheses:hypotheses;
    if(['idea','experiment','full'].includes(id)) {
      out['IDEA.md'] = '# ADHD network hypotheses — DEMO\n\n'+disclosure+'\n\n'+ranked.filter(h=>h.status==='Selected'||h.status==='Validated (simulated)').map(h=>`## ${h.id}\n\n${h.text}\n\nFalsification criterion: no stable gain in the prespecified comparison with identical splits and matched capacity, or failure of the corrected statistical test.\n`).join('\n');
      out['hypotheses.csv']=hypothesisCsv(hypothesisPool);
    }
    if(['data','full'].includes(id)) out['DATA_LAYOUT.txt']=disclosure+'\n\n'+datasetTree;
    if(['experiment','full'].includes(id)) {
      out['model_example.py']='# '+disclosure+'\n'+modelCode;
      out['config.json']=JSON.stringify(config,null,2);
    }
    if(id==='experiment') out['results.csv']=csv([['data_kind',...resultHeaders],...results.map(r=>['simulated_not_measured',...r])]);
    if(id==='full') {
      out['hypotheses_final.csv']=hypothesisCsv(fullHypothesisPool(options));
      const rounds=fullEvents(options).filter(e=>e.kind==='round');
      const success=rounds.at(-1).verdict.reason==='target_reached';
      if(success)out['results.csv']=csv([['data_kind',...resultHeaders],...results.map(r=>['simulated_not_measured',...r])]);
      Object.assign(out,roundArtifacts(rounds));
      out['validation.csv']=csv([['data_kind','hypothesis','status'],...(success?[['H1','passed'],['H2','passed'],['H6','passed'],['H3','insufficient']]:[['all','insufficient_after_20_rounds']]).map(r=>['simulated_not_measured',...r])]);
      out['REPORT.md']+='\n\n## Final outcome\n\n'+(success?'3 rounds; H1/H2/H6 passed simulated final validation; H3 insufficient.\n\nOne final test after freezing: H1 ΔR² = 0.08, q = 0.03; H2 ΔAUROC = 0.05, q = 0.04; H6 standardized β = 0.21, q = 0.02. H3 remains insufficient.':'20 rounds; zero validated; stop at limit. The final test set was not used to search for positive results.');
    }
    return out;
  }
  return {disclosure,topic,datasets,hypotheses,hypothesisPool,fullHypothesisPool,hypothesisCsv,finalHypotheses,scoreNote,definitions,results,resultHeaders,config,modelCode,datasetTree,graph,roundHeaders,roundDecision,roundArtifacts,buildScenario,decideLoop,artifacts};
});
