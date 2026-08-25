from __future__ import annotations

"""Mass batch: model_training tasks T396-T446 (51 tasks).

Follows the T101/T134/T135 conventions: models/train_unified.py --model X,
5-fold deterministic split, HCP age regression + ABIDE dx classification,
artefacts under models/benchmark_results/<folder>/<setting>/.
"""

from neurobench_taskkit import body

CAT = "model_training"

MT_INS = [
    "ROI-level FC matrices per subject (NPZ), built from a chosen atlas "
    "(e.g. `schaefer_200_7net`, `aal_116`)",
    "Subject list file (`ready_subjects.txt`)",
    "Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: "
    "`data/abide_dx_labels.csv`)",
    "Atlas name and ROI count (must match FC dimension)",
]

MT_OUTS = [
    "Per-fold test metrics CSV (regression: MAE / RMSE / R^2; "
    "classification: accuracy / AUC / F1)",
    "Aggregated 5-fold mean +/- std",
    "`result_YYYYMMDD_HHMMSS.json` metadata file",
]

MT_EVAL = [
    "Results within a reasonable baseline range for the model class "
    "(document the reference used).",
    "Manually scored for reproducibility (seed + atlas + fold all logged).",
]


def _mt(num, slug, model, title, desc, extra_cons=None, extra_outs=None,
        extra_eval=None):
    folder = f"T{num}_{slug}"
    cons = [
        f"Use `models/train_unified.py --model {model}` for both settings.",
        "5-fold split with deterministic `--seed`; report fold 0..4 test "
        "metrics.",
    ] + (extra_cons or []) + [
        f"Save artefacts under `models/benchmark_results/{folder}/"
        "<setting>/`.",
        f"Save checkpoints under `models/checkpoints/{model}/<atlas>/"
        "fold{k}.pt`.",
    ]
    outs = MT_OUTS[:1] + MT_OUTS[1:2] + (extra_outs or []) + MT_OUTS[2:]
    return (num, slug, CAT, title,
            body(folder, desc, MT_INS, cons, outs,
                 evaluation=(extra_eval or []) + MT_EVAL))


def _pt(num, slug, title, desc, ins, cons, outs, ev=None):
    """Protocol task (not a plain train_unified run)."""
    folder = f"T{num}_{slug}"
    return (num, slug, CAT, title,
            body(folder, desc, ins, cons, outs, evaluation=ev))


TASKS = []

# --- A. New model architectures (T396-T411, 16) ------------------------------------
_MODELS = [
    ("graphsage", "graphsage", "GraphSAGE",
     "GraphSAGE (mean aggregator, 2 layers) on ROI graphs; node features "
     "from FC rows."),
    ("gin", "gin", "GIN",
     "Graph Isomorphism Network (2 layers, MLP epsilon-learnable) on "
     "thresholded FC graphs."),
    ("chebnet", "chebnet", "ChebNet",
     "Chebyshev spectral GCN (K=3) on FC graphs; document the Laplacian "
     "normalization used."),
    ("dgcnn", "dgcnn", "DGCNN (EdgeConv)",
     "DGCNN with EdgeConv layers on k-NN graphs built from FC (k=10); "
     "document graph construction."),
    ("transformer_ts", "transformer_ts", "ROI Time-Series Transformer",
     "Vanilla transformer over ROI time series (CLS token readout); "
     "positional encoding documented."),
    ("cnn1d_ts", "cnn1d_ts", "1D-CNN on ROI Time Series",
     "1D CNN over ROI time series with global average pooling readout."),
    ("cnn2d_fc", "cnn2d_fc", "2D-CNN on FC Matrix",
     "2D CNN treating the FC matrix as an image (symmetric input; document "
     "upper-triangle handling)."),
    ("gru_ts", "gru_ts", "GRU on ROI Time Series",
     "2-layer GRU over ROI time series, last-hidden readout."),
    ("tcn_ts", "tcn_ts", "Temporal Convolutional Network",
     "TCN over ROI time series with dilated causal convolutions; receptive "
     "field documented."),
    ("mlp_mixer_fc", "mlp_mixer_fc", "MLP-Mixer on FC",
     "MLP-Mixer over vectorized FC (ROI-mixing + feature-mixing blocks)."),
    ("autoencoder_pretrain", "ae_pretrain", "Autoencoder Pretraining + Probe",
     "Pretrain an FC autoencoder (reconstruction), then train a linear/ridge "
     "probe on the bottleneck for both settings."),
    ("simclr_fc", "simclr_fc", "SimCLR-Style Contrastive Pretraining",
     "Contrastive self-supervision on FC (augmentations: edge dropout, "
     "noise), then linear probe for both settings."),
    ("dann_site", "dann_site", "Domain-Adversarial Site Adaptation",
     "DANN with site-discriminator gradient reversal on top of a GCN "
     "backbone, targeting site-invariant ABIDE features."),
    ("mtl_age_dx", "mtl_age_dx", "Multi-Task Learning (age + dx)",
     "Joint multi-task model: shared GNN trunk with age-regression and "
     "dx-classification heads; loss weighting documented."),
    ("ensemble_gnn_voting", "ensemble_gnn_voting", "GNN Voting Ensemble",
     "Train 5 GNNs (braingnn/gcn/gat mix) and combine by majority vote "
     "(classification) / mean (regression); compare to best single."),
    ("distill_teacher_student", "distill_ts", "Knowledge Distillation",
     "Distill a large teacher (BNT) into a small student (GCN) on the same "
     "setting; report student vs. teacher gap."),
]
for i, (slug, model, name, detail) in enumerate(_MODELS):
    TASKS.append(_mt(
        396 + i, f"{slug}_train_eval", model, f"{name} Training and Evaluation",
        f"Train and evaluate {name} on preprocessed functional connectivity "
        "data for two settings: HCP age regression and ABIDE diagnosis "
        f"classification. {detail}",
        extra_cons=["Architecture hyperparameters documented in "
                    "`model_config.json`."]))

# --- B. Classical baselines (T412-T419, 8) -------------------------------------------
_BASELINES = [
    ("xgboost_baseline", "xgboost", "XGBoost Baseline",
     "Gradient-boosted trees on vectorized FC features (upper triangle)."),
    ("lightgbm_baseline", "lightgbm", "LightGBM Baseline",
     "LightGBM on vectorized FC features."),
    ("svm_rbf_baseline", "svm_rbf", "SVM-RBF Baseline",
     "SVM with RBF kernel; C/gamma via inner CV."),
    ("random_forest_baseline", "random_forest", "Random Forest Baseline",
     "Random forest (500 trees) with feature-importance export."),
    ("logistic_l2_baseline", "logistic_l2", "L2 Logistic Regression Baseline",
     "L2-regularized logistic regression (classification) / ridge "
     "(regression)."),
    ("elasticnet_baseline", "elasticnet", "ElasticNet Baseline",
     "ElasticNet with inner-CV alpha/l1_ratio selection."),
    ("kernel_ridge_baseline", "kernel_ridge", "Kernel Ridge Baseline",
     "Kernel ridge regression (RBF) for the regression setting."),
    ("gpr_baseline", "gpr", "Gaussian Process Regression Baseline",
     "GPR with RBF+WhiteKernel on a PCA-reduced FC feature set (document "
     "n_components)."),
]
for i, (slug, model, name, detail) in enumerate(_BASELINES):
    TASKS.append(_mt(
        412 + i, slug, model, name,
        f"Train and evaluate the {name} on vectorized FC features for the "
        f"standard settings. {detail}",
        extra_cons=["Feature scaling documented; inner CV protocol kept "
                    "nested (no test leakage)."]))

# --- C. Protocol / ablation variants (T420-T431, 12) -----------------------------------
TASKS.append(_pt(
    420, "optuna_hparam_sweep", "Optuna Hyperparameter Sweep (BrainGNN)",
    "Run an Optuna hyperparameter sweep for BrainGNN (lr, hidden dim, "
    "pooling ratio, weight decay): 30 trials, pruned, with the search space "
    "and best config recorded.",
    MT_INS + ["Base config to start from (optional)"],
    ["Nested protocol: sweep on train/val only; final eval on untouched "
     "test folds.",
     "Save artefacts under `models/benchmark_results/T420_optuna_hparam_sweep/`.",
     "Study persisted (sqlite) so it can be resumed."],
    ["`optuna_study.db`", "`best_config.json`", "`importance_plot.png`",
     "Final 5-fold metrics CSV with the best config"],
    ev=["Best-config final metrics must not reuse val-tuned information.",
        "This test case is manually evaluated."]))
TASKS.append(_pt(
    421, "learning_curve_scaling", "Learning Curve: Data Scaling",
    "Measure the learning curve of BrainGNN: train on 10/25/50/75/100% of "
    "the training data (stratified subsets, fixed test), plot performance "
    "vs. data fraction.",
    MT_INS,
    ["Subsets nested (10% is a subset of 25% etc.), seed fixed.",
     "Save artefacts under `models/benchmark_results/T421_learning_curve_scaling/`."],
    ["`learning_curve.csv`", "`learning_curve.png`",
     "`result_YYYYMMDD_HHMMSS.json`"],
    ev=["Curve monotonicity discussed; saturation point estimated.",
        "This test case is manually evaluated."]))
TASKS.append(_pt(
    422, "seed_stability_10seeds", "Seed Stability: 10-Seed Run",
    "Run the same model (GCN) with 10 different seeds on fold 0 only; "
    "report metric variance attributable to seed choice.",
    MT_INS,
    ["Everything else identical (config, fold, atlas).",
     "Save artefacts under `models/benchmark_results/T422_seed_stability_10seeds/`."],
    ["`seed_metrics.csv`", "`seed_variance_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    423, "ablation_no_topk", "Ablation: BrainGNN without TopK Pooling",
    "Ablate BrainGNN's TopK pooling (replace with global mean pooling); "
    "quantify the contribution of pooling to both settings.",
    MT_INS,
    ["Single-change principle: only the pooling differs.",
     "Save artefacts under `models/benchmark_results/T423_ablation_no_topk/`."],
    ["Ablation metrics CSV vs. full model",
     "`ablation_report.md`", "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    424, "ablation_no_consist_loss", "Ablation: BrainGNN without Consistency Loss",
    "Ablate the consistency loss term from BrainGNN training; quantify its "
    "contribution on both settings.",
    MT_INS,
    ["Only the loss term differs; weights of remaining terms unchanged.",
     "Save artefacts under `models/benchmark_results/T424_ablation_no_consist_loss/`."],
    ["Ablation metrics CSV", "`ablation_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    425, "fc_threshold_sensitivity", "Sensitivity: FC Edge Threshold",
    "Evaluate a GCN's sensitivity to FC binarization threshold (top "
    "5/10/20% edges): same model, same folds, three graphs.",
    MT_INS,
    ["Threshold applied identically across subjects.",
     "Save artefacts under `models/benchmark_results/T425_fc_threshold_sensitivity/`."],
    ["`threshold_metrics.csv`", "`threshold_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    426, "class_imbalance_strategies", "Class Imbalance: Weighting vs. Oversampling",
    "Compare class-imbalance handling for ABIDE dx: weighted loss vs. "
    "random oversampling vs. none, with the same GCN.",
    MT_INS,
    ["Report balanced accuracy + F1 in addition to accuracy.",
     "Save artefacts under `models/benchmark_results/T426_class_imbalance_strategies/`."],
    ["`imbalance_comparison.csv`", "`imbalance_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    427, "huber_vs_mse", "Loss Comparison: Huber vs. MSE (Regression)",
    "Compare Huber vs. MSE loss for HCP age regression with the same "
    "model; Huber delta documented.",
    MT_INS,
    ["Delta=1.0 unless justified.", "Report outlier sensitivity "
     "(worst-10-subject MAE).",
     "Save artefacts under `models/benchmark_results/T427_huber_vs_mse/`."],
    ["`loss_comparison.csv`", "`outlier_analysis.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    428, "optimizer_comparison", "Optimizer Comparison: Adam vs. AdamW vs. SGD",
    "Compare optimizers (Adam / AdamW / SGD+momentum with cosine schedule) "
    "for a GCN on both settings; learning rates re-tuned per optimizer on "
    "val only.",
    MT_INS,
    ["LR tuning protocol documented (val-only).",
     "Save artefacts under `models/benchmark_results/T428_optimizer_comparison/`."],
    ["`optimizer_metrics.csv`", "`optimizer_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    429, "early_stopping_sweep", "Early-Stopping Patience Sweep",
    "Sweep early-stopping patience (5/10/20 epochs) for a GNN; report "
    "effect on final metric and training time.",
    MT_INS,
    ["Patience is the only change.",
     "Save artefacts under `models/benchmark_results/T429_early_stopping_sweep/`."],
    ["`patience_metrics.csv`", "`patience_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    430, "batch_size_sweep", "Batch Size Sweep",
    "Sweep batch size (8/16/32/64) for a GNN; report metric + memory + "
    "epoch-time trade-offs.",
    MT_INS,
    ["Report peak GPU memory per batch size.",
     "Save artefacts under `models/benchmark_results/T430_batch_size_sweep/`."],
    ["`batch_metrics.csv`", "`batch_tradeoff.png`",
     "`result_YYYYMMDD_HHMMSS.json`"]))
TASKS.append(_pt(
    431, "edge_dropout_augmentation", "Augmentation: Edge Dropout",
    "Evaluate edge-dropout augmentation (drop 10% edges per epoch) for a "
    "GCN on both settings; compare against no augmentation.",
    MT_INS,
    ["Dropout applied to training graphs only.",
     "Save artefacts under `models/benchmark_results/T431_edge_dropout_augmentation/`."],
    ["`augmentation_metrics.csv`", "`augmentation_report.md`",
     "`result_YYYYMMDD_HHMMSS.json`"]))

# --- D. Transfer / pretraining (T432-T437, 6) -------------------------------------------------
_TRANSFER = [
    (432, "pretrain_hcp_finetune_abide", "Pretrain HCP -> Finetune ABIDE",
     "Pretrain a GNN on HCP (age regression), finetune on ABIDE dx; compare "
     "against training from scratch on ABIDE only.",
     ["Same atlas for both datasets.", "Finetune protocol (frozen layers, "
      "LR) documented."]),
    (433, "ssl_linear_probe", "SSL Pretrain + Linear Probe",
     "Use the T407 contrastive-pretrained encoder (or train one here), "
     "freeze it, and train linear probes for both settings; report vs. "
     "end-to-end.",
     ["Encoder frozen during probing.", "Probe protocol identical across "
      "settings."]),
    (434, "zeroshot_hcp_to_abide", "Zero-Shot Cross-Dataset Transfer",
     "Evaluate a model trained on HCP directly on ABIDE without any "
     "finetuning (common atlas); quantify the transfer gap vs. in-domain "
     "training.",
     ["Common atlas mandatory.", "Report both in-domain and transfer "
      "numbers."]),
    (435, "finetune_last_layer", "Finetune Last Layer Only",
     "Transfer with only the readout layer trainable (backbone frozen); "
     "compare against full finetuning from T432.",
     ["Only readout parameters update (verify via optimizer param "
      "groups)."]),
    (436, "fewshot_site_adaptation", "Few-Shot Site Adaptation",
     "Adapt an ABIDE-trained model to a held-out site with k=5 labeled "
     "subjects per site; report adaptation gain over zero-shot.",
     ["k fixed at 5; selection of the 5 documented (seeded)."]),
    (437, "curriculum_easy_to_hard", "Curriculum Learning",
     "Train with a curriculum: easy subjects (high tSNR / low motion) "
     "first, progressively adding harder ones; compare against shuffled "
     "training.",
     ["Difficulty score defined from QC metrics and documented."]),
]
for num, slug, title, desc, cons in _TRANSFER:
    TASKS.append(_pt(
        num, slug, title, desc,
        MT_INS,
        cons + [f"Save artefacts under `models/benchmark_results/T{num}_{slug}/`."],
        ["Transfer comparison CSV", "`transfer_report.md`",
         "`result_YYYYMMDD_HHMMSS.json`"]))

# --- E. Robustness / reliability training-side (T438-T442, 5) -----------------------------------------
_ROB = [
    (438, "test_retest_icc", "Test-Retest ICC of Predictions",
     "Train on HCP session-1 resting-state and predict on session-2 "
     "(retest): compute ICC of per-subject predictions across sessions.",
     ["Subject pairing verified from IDs.",
      "ICC(2,1) reported with 95% CI."]),
    (439, "noise_injection_training", "Noise-Robust Training",
     "Train with Gaussian noise augmentation on FC; evaluate clean vs. "
     "noisy test performance vs. a model trained without augmentation.",
     ["Noise sigma documented; test noise levels grid reported."]),
    (440, "roi_dropout_training", "ROI-Dropout Robust Training",
     "Train with random ROI dropout (simulate missing parcels); evaluate "
     "degradation vs. number of dropped ROIs.",
     ["Dropout rate grid documented."]),
    (441, "age_stratified_training", "Age-Stratified Training",
     "Train separate models per age band (HCP young vs. older bin) and "
     "compare with a pooled model; discuss age-effect confounding.",
     ["Bands defined a priori and documented."]),
    (442, "motion_matched_training", "Motion-Matched Training Cohort",
     "Build a motion-matched training subset (match mean-FD distribution "
     "across dx groups) and retrain; compare against the unmatched model.",
     ["Matching algorithm + post-match FD stats documented."]),
]
for num, slug, title, desc, cons in _ROB:
    TASKS.append(_pt(
        num, slug, title, desc,
        MT_INS,
        cons + [f"Save artefacts under `models/benchmark_results/T{num}_{slug}/`."],
        ["Metrics CSV per condition", "`robustness_report.md`",
         "`result_YYYYMMDD_HHMMSS.json`"]))

# --- F. Interpretability training-side (T443-T446, 4) -----------------------------------------------------
_INTERP = [
    (443, "gradcam_fc_maps", "Grad-CAM-style Saliency on FC-CNN",
     "Train the 2D-CNN-on-FC model and produce Grad-CAM-style saliency "
     "maps; aggregate per-ROI importance across subjects.",
     ["Saliency method equations documented.",
      "Per-ROI aggregation as mean |saliency|."]),
    (444, "integrated_gradients_roi", "Integrated Gradients ROI Importance",
     "Compute Integrated Gradients attributions for a trained GNN; export "
     "per-ROI importance rankings per fold.",
     ["Baseline (zero graph) documented.", "Attributions averaged per "
      "class for classification."]),
    (445, "attention_rollout_transformer", "Attention Rollout for TS-Transformer",
     "Apply attention rollout to the trained ROI time-series transformer; "
     "visualize ROI-to-ROI attention flow for example subjects.",
     ["Rollout implementation cited.", "Examples span both settings."]),
    (446, "shap_tabular_baselines", "SHAP for Tabular Baselines",
     "Compute SHAP values for the ridge and random-forest baselines; "
     "export global ROI importance and per-subject force summaries.",
     ["SHAP variant (KernelSHAP/TreeSHAP) per model documented.",
      "Global importance as mean |SHAP|."]),
]
for num, slug, title, desc, cons in _INTERP:
    TASKS.append(_pt(
        num, slug, title, desc,
        MT_INS,
        cons + [f"Save artefacts under `models/benchmark_results/T{num}_{slug}/`."],
        ["Per-ROI importance CSV", "Representative saliency/attribution PNG",
         "`result_YYYYMMDD_HHMMSS.json`"]))

assert len(TASKS) == 51, f"model_train batch must be 51 tasks, got {len(TASKS)}"
