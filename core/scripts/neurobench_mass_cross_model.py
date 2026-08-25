from __future__ import annotations

"""Mass batch: cross_model_evaluation tasks T447-T500 (54 tasks)."""

from neurobench_taskkit import body

CAT = "cross_model_evaluation"

CM_INS = [
    "Per-model 5-fold test predictions/metrics from the benchmark runs "
    "(required)",
    "Subject list + labels + site/sex/age covariates CSV (required)",
]


def _c(num, slug, title, desc, ins=None, cons=None, outs=None, ev=None):
    folder = f"T{num}_{slug}"
    cons = list(cons or []) + [
        f"Save artefacts under `models/benchmark_results/{folder}/`."]
    return (num, slug, CAT, title,
            body(folder, desc, ins or CM_INS, cons, outs, evaluation=ev))


TASKS = []

# --- A. Statistical comparison (T447-T454, 8) -----------------------------------
_STAT = [
    (447, "corrected_resampled_ttest", "Corrected Resampled t-Test Between Models",
     "Compare model pairs with Nadeau-Bengio corrected resampled t-test "
     "over the 5-fold results; produce a pairwise p-value matrix.",
     ["Correction factor applied (test/train ratio).",
      "Multiple-comparison note included."],
     ["`pairwise_pvalues.csv`", "`ttest_report.md`"]),
    (448, "bayesian_signed_rank", "Bayesian Signed-Rank Comparison",
     "Compare model pairs with the Bayesian signed-rank test; report "
     "posterior probabilities of practical equivalence (ROPE).",
     ["ROPE justified.", "Plot posterior distributions per pair."],
     ["`bayesian_pairwise.csv`", "`posterior_plots.png`"]),
    (449, "demsar_cd_diagram", "Demšar Critical-Difference Diagram",
     "Rank all models with the Friedman test and produce a "
     "critical-difference diagram (Nemenyi post-hoc) across evaluation "
     "settings.",
     ["Ranks computed per (dataset x fold).", "CD value shown on the "
      "diagram."],
     ["`cd_diagram.png`", "`friedman_ranks.csv`"]),
    (450, "bootstrap_ci_leaderboard", "Bootstrap CI Leaderboard",
     "Recompute the leaderboard with bootstrap 95% CIs (10k resamples over "
     "subject-level predictions) per model and setting.",
     ["Subject-level bootstrap, not fold-level.",
      "Leaderboard sorted with CI overlap discussion."],
     ["`leaderboard_ci.csv`", "`leaderboard_ci.png`"]),
    (451, "delong_roc_comparison", "DeLong ROC Comparison",
     "Compare classification AUCs between model pairs with the DeLong test "
     "on pooled test predictions.",
     ["Same test subjects across compared models (verify).",
      "Report AUC differences + CIs, not just p-values."],
     ["`delong_results.csv`", "`roc_overlays.png`"]),
    (452, "mcnemar_classifiers", "McNemar Test Between Classifiers",
     "Run McNemar's test on paired classification outcomes per model pair; "
     "build the discordant-pair tables.",
     ["Continuity correction documented.",
      "Discordant cases exported for error analysis."],
     ["`mcnemar_tables.csv`", "`discordant_subjects.csv`"]),
    (453, "atlas_model_anova", "Atlas x Model Interaction ANOVA",
     "Two-way ANOVA (model x atlas) on 5-fold metrics: quantify main "
     "effects and interaction; simple-effects follow-up where interaction "
     "is significant.",
     ["Assumptions checked (normality of residuals, sphericity note).",
      "Effect sizes (partial eta^2) reported."],
     ["`anova_table.csv`", "`interaction_plot.png`"]),
    (454, "friedman_rank_stability", "Friedman Rank Stability Across Settings",
     "Test whether model rankings are stable across all evaluation "
     "settings (datasets x atlases): Friedman test + Kendall's W per "
     "grouping.",
     ["Groupings defined a priori.",
      "W interpreted with conventional bands."],
     ["`rank_stability.csv`", "`rank_stability.md`"]),
]
for num, slug, title, desc, cons, outs in _STAT:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- B. Reliability / calibration (T455-T460, 6) -----------------------------------------
_REL = [
    (455, "calibration_reliability", "Classifier Calibration Analysis",
     "Assess probability calibration of classifiers: reliability diagrams, "
     "ECE, Brier score per model; Platt/isotonic recalibration comparison.",
     ["Same binning across models.", "Recalibration fitted on val only."],
     ["`calibration_metrics.csv`", "`reliability_diagrams.png`"]),
    (456, "test_retest_model_reliability", "Model Test-Retest Reliability",
     "Evaluate per-subject prediction reliability across HCP test-retest "
     "sessions for each model; rank models by ICC.",
     ["Paired sessions verified.", "ICC(2,1) + CI per model."],
     ["`retest_icc.csv`", "`retest_scatter.png`"]),
    (457, "prediction_consistency_icc", "Prediction Consistency Across Folds",
     "For models with out-of-fold predictions, compute prediction "
     "consistency across fold re-assignments (multi-seed re-split) as ICC.",
     ["3 re-splits minimum.", "Per-model ICC + ranking."],
     ["`fold_consistency.csv`", "`consistency_report.md`"]),
    (458, "seed_variance_report", "Seed Variance Report",
     "Aggregate 10-seed runs (from T422-style protocols) across models: "
     "report seed-driven variance per model as a robustness ranking.",
     ["Same seeds per model.", "Variance decomposition (seed vs. fold) "
      "where data allows."],
     ["`seed_variance.csv`", "`seed_variance.md`"]),
    (459, "fold_variance_report", "Fold Variance Report",
     "Characterize fold-to-fold variance for every model: per-fold metric "
     "distributions, worst-fold analysis, and implications for reporting.",
     ["Boxplots per model.", "Worst-fold subjects exported."],
     ["`fold_variance.csv`", "`fold_boxplots.png`"]),
    (460, "decision_curve_analysis", "Decision Curve Analysis",
     "Run decision curve analysis for the classifiers: net benefit across "
     "threshold probabilities, compared against treat-all/treat-none.",
     ["Threshold range justified clinically.",
      "Curves per model on one figure."],
     ["`decision_curves.png`", "`net_benefit_table.csv`"]),
]
for num, slug, title, desc, cons, outs in _REL:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- C. Fairness / subgroups (T461-T466, 6) ------------------------------------------------------
_FAIR = [
    (461, "sex_subgroup_metrics", "Sex-Subgroup Metrics",
     "Report per-sex performance for every model: metrics, CIs, and the "
     "male/female gap with significance.",
     ["Gaps tested with bootstrap CIs.",
      "Confounding by site/age discussed."],
     ["`sex_subgroup_metrics.csv`", "`sex_gap_report.md`"]),
    (462, "age_group_metrics", "Age-Group Metrics",
     "Report performance per age group (tertiles or study-defined bands) "
     "for every model; test for age-related performance gradients.",
     ["Bands identical across models.", "Gradient tested with trend "
      "test."],
     ["`age_group_metrics.csv`", "`age_gradient.png`"]),
    (463, "site_subgroup_metrics", "Site-Subgroup Metrics (ABIDE)",
     "Report per-site classification metrics for every model on ABIDE; "
     "identify sites where all models fail (systematic site effects).",
     ["Small-N sites flagged.", "Per-site N shown next to metrics."],
     ["`site_metrics.csv`", "`site_heatmap.png`"]),
    (464, "fairness_gap_summary", "Fairness Gap Summary",
     "Consolidate subgroup gaps (sex, age, site, motion) into one fairness "
     "scorecard per model with an overall fairness ranking.",
     ["Scorecard schema fixed.", "Ranking rule documented."],
     ["`fairness_scorecard.csv`", "`fairness_summary.md`"]),
    (465, "motion_stratified_eval", "Motion-Stratified Evaluation",
     "Stratify test subjects by mean FD (low/medium/high) and report "
     "per-stratum performance for every model.",
     ["Strata from FD tertiles of the full cohort.",
      "Discuss motion-driven performance drop."],
     ["`motion_strata_metrics.csv`", "`motion_robustness.png`"]),
    (466, "icv_confound_check", "ICV / Head-Size Confound Check",
     "Check whether model predictions correlate with intracranial volume "
     "(a proxy confound): partial correlations controlling for the label.",
     ["ICV from FreeSurfer eTIV.", "Report partial r + p per model."],
     ["`icv_confound.csv`", "`confound_report.md`"]),
]
for num, slug, title, desc, cons, outs in _FAIR:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- D. Interpretability agreement (T467-T472, 6) -----------------------------------------------------
_INTA = [
    (467, "roi_importance_topk_overlap", "ROI Importance Top-k Overlap",
     "Compare per-model ROI importance rankings (from attention/SHAP/IG "
     "tasks): top-20 overlap coefficients between model pairs.",
     ["Importance sources documented per model.",
      "Overlap = Szymkiewicz-Simpson coefficient."],
     ["`topk_overlap_matrix.csv`", "`overlap_heatmap.png`"]),
    (468, "shap_vs_attention_agreement", "SHAP vs. Attention Agreement",
     "Quantify agreement between SHAP-based and attention-based ROI "
     "importance for the same predictions: rank correlation per subject, "
     "aggregated.",
     ["Same subjects for both methods.", "Per-subject Spearman + "
      "distribution."],
     ["`method_agreement.csv`", "`agreement_report.md`"]),
    (469, "importance_fold_stability", "Importance Stability Across Folds",
     "Measure stability of ROI importance rankings across folds per model: "
     "pairwise rank correlation between fold-specific rankings.",
     ["Per-fold importance from matched checkpoints.",
      "Stability ranking across models."],
     ["`fold_stability.csv`", "`stability_heatmap.png`"]),
    (470, "saliency_method_comparison", "Saliency Method Comparison",
     "For one GNN, compare saliency methods (gradients, IG, attention, "
     "occlusion): agreement + a sanity check (model parameter "
     "randomization).",
     ["Cascading randomization sanity check included.",
      "Methods ranked by sanity-check pass."],
     ["`saliency_comparison.csv`", "`sanity_check.png`"]),
    (471, "counterfactual_edge_perturbation", "Counterfactual Edge Perturbation",
     "Perturbation analysis: remove top-k important edges (per importance "
     "method) and measure prediction change; validate that importance "
     "rankings are causal-ish.",
     ["k grid documented.", "Compared against random-edge removal "
      "baseline."],
     ["`perturbation_curves.csv`", "`perturbation_report.md`"]),
    (472, "disagreement_case_mining", "Model Disagreement Case Mining",
     "Mine subjects where top models disagree: cluster the disagreement "
     "cases, characterize them (site, motion, age), and write case "
     "summaries.",
     ["Disagreement defined on hard labels.",
      "At least 10 case summaries."],
     ["`disagreement_cases.csv`", "`case_summaries.md`"]),
]
for num, slug, title, desc, cons, outs in _INTA:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- E. Robustness evaluation (T473-T478, 6) -------------------------------------------------------------
_ROBE = [
    (473, "noise_robustness_comparison", "Noise Robustness Comparison",
     "Evaluate all models under FC noise injection (sigma grid); plot "
     "degradation curves and rank models by robustness.",
     ["Same noise realizations across models (seeded).",
      "AUC-of-degradation-curve as the robustness score."],
     ["`noise_robustness.csv`", "`degradation_curves.png`"]),
    (474, "missing_roi_comparison", "Missing-ROI Robustness Comparison",
     "Evaluate all models with randomly dropped ROIs (5/10/20%); compare "
     "degradation slopes.",
     ["Drop pattern shared across models per seed.",
      "Report graceful-degradation ranking."],
     ["`missing_roi_metrics.csv`", "`missing_roi_report.md`"]),
    (475, "site_shift_mmd", "Site Shift Quantification (MMD)",
     "Quantify distribution shift between ABIDE sites in FC space via MMD "
     "with a Gaussian kernel; relate shift magnitude to LOSO performance.",
     ["Bandwidth via median heuristic.",
      "Correlation of MMD vs. LOSO drop reported."],
     ["`site_mmd_matrix.csv`", "`mmd_vs_loso.png`"]),
    (476, "harmonization_method_comparison", "Harmonization Method Comparison",
     "Compare harmonization strategies (none / ComBat / CovBat) by "
     "downstream LOSO performance of a fixed model.",
     ["Harmonization fit on train only (no leakage).",
      "Same splits across methods."],
     ["`harmonization_comparison.csv`", "`harmonization_report.md`"]),
    (477, "fc_threshold_eval_comparison", "FC Threshold Sensitivity Comparison",
     "Compare model sensitivity to FC edge threshold (5/10/20%) across "
     "models; identify threshold-stable models.",
     ["Threshold protocol identical to T425.",
      "Stability score per model."],
     ["`threshold_sensitivity.csv`", "`threshold_stability_rank.md`"]),
    (478, "mc_dropout_uncertainty", "MC-Dropout Uncertainty Evaluation",
     "Estimate predictive uncertainty via MC dropout for each GNN; "
     "evaluate uncertainty quality (error-vs-uncertainty correlation, "
     "selective prediction AUC).",
     ["T=30 stochastic forward passes.",
      "Selective-prediction curves per model."],
     ["`uncertainty_metrics.csv`", "`selective_curves.png`"]),
]
for num, slug, title, desc, cons, outs in _ROBE:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- F. External validation (T479-T484, 6) ---------------------------------------------------------------------
_EXT = [
    (479, "abide2_external_validation", "External Validation on ABIDE-II",
     "Validate ABIDE-I-trained models on ABIDE-II (no finetuning): metrics, "
     "degradation vs. in-domain, and per-site breakdown.",
     ["Common atlas across ABIDE-I/II.",
      "No finetuning; document any preprocessing alignment."],
     ["`abide2_metrics.csv`", "`external_validity_report.md`"]),
    (480, "cobre_external_validation", "External Validation on COBRE",
     "Validate schizophrenia-relevant models (or dx-transfer probes) on "
     "COBRE data; report transfer metrics honestly.",
     ["Label mapping documented (ASD-trained vs. SCZ target).",
      "If not applicable, run as domain-shift probe and say so."],
     ["`cobre_metrics.csv`", "`cobre_report.md`"]),
    (481, "hcp_to_abide_transfer_eval", "HCP->ABIDE Transfer Evaluation",
     "Evaluate HCP-pretrained models on ABIDE (zero-shot and linear-probe): "
     "transfer curve vs. amount of ABIDE finetuning data.",
     ["Finetuning fractions: 0/10/25/50/100%.",
      "Same atlas and preprocessing."],
     ["`transfer_curve.csv`", "`transfer_curve.png`"]),
    (482, "pretrain_gain_analysis", "Pretraining Gain Analysis",
     "Quantify the gain of pretraining (SSL or HCP) across models: "
     "pretrained vs. scratch comparison table with significance.",
     ["Paired comparisons per model.",
      "Gain attributed to pretraining only (same budget)."],
     ["`pretrain_gain.csv`", "`pretrain_gain.md`"]),
    (483, "zeroshot_crossdataset_report", "Zero-Shot Cross-Dataset Report",
     "Consolidate zero-shot transfer results (HCP->ABIDE, ABIDE-I->II) "
     "into one cross-dataset generalization report with a shift-difficulty "
     "ranking.",
     ["Metrics pulled from the corresponding task outputs.",
      "Ranking justified by shift metrics (e.g. MMD)."],
     ["`crossdataset_report.md`", "`zeroshot_matrix.csv`"]),
    (484, "normative_modeling_eval", "Normative Modeling Evaluation",
     "Fit a normative model (PCN-style) on controls and evaluate deviation "
     "scores (z-scores) for patients; compare separation against direct "
     "classifiers.",
     ["Normative model fit on controls only.",
      "Report AUC of deviation-score classification."],
     ["`normative_metrics.csv`", "`deviation_maps.png`"]),
]
for num, slug, title, desc, cons, outs in _EXT:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- G. Ensembles / meta (T485-T489, 5) ----------------------------------------------------------------------------
_ENS = [
    (485, "voting_ensemble_eval", "Voting Ensemble Evaluation",
     "Build a hard/soft voting ensemble of the top-5 models; evaluate "
     "against the best single model with bootstrap CIs.",
     ["Members selected on val only.",
      "Both voting rules reported."],
     ["`voting_metrics.csv`", "`voting_report.md`"]),
    (486, "stacking_ensemble_eval", "Stacking Ensemble Evaluation",
     "Build a stacking ensemble (out-of-fold predictions -> logistic "
     "meta-learner); guard against leakage and evaluate honestly.",
     ["Meta-learner trained on OOF predictions only.",
      "Leakage checks documented."],
     ["`stacking_metrics.csv`", "`stacking_report.md`"]),
    (487, "ensemble_gain_analysis", "Ensemble Gain Analysis",
     "Analyze when ensembling helps: ensemble gain vs. member diversity "
     "(prediction disagreement), across settings.",
     ["Diversity via Q-statistic or disagreement measure.",
      "Scatter: diversity vs. gain."],
     ["`ensemble_gain.csv`", "`diversity_gain.png`"]),
    (488, "model_diversity_correlation", "Model Diversity Correlation",
     "Compute pairwise prediction-correlation between models; cluster "
     "models into families by prediction similarity.",
     ["Correlation on subject-level predictions.",
      "Hierarchical clustering dendrogram included."],
     ["`diversity_matrix.csv`", "`model_dendrogram.png`"]),
    (489, "checkpoint_soup_eval", "Checkpoint Soup Evaluation",
     "Evaluate model soups: average weights of the last-k checkpoints (or "
     "greedy soup) per model; compare against best-checkpoint selection.",
     ["Soup recipe documented.", "Same eval split as standard runs."],
     ["`soup_metrics.csv`", "`soup_report.md`"]),
]
for num, slug, title, desc, cons, outs in _ENS:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- H. Efficiency (T490-T494, 5) -----------------------------------------------------------------------------------
_EFF = [
    (490, "training_time_pareto", "Training-Time vs. Performance Pareto",
     "Plot the training-time vs. performance Pareto front across models "
     "per setting; identify Pareto-optimal models.",
     ["Wall time from training logs (same hardware).",
      "Pareto front marked on the plot."],
     ["`pareto.csv`", "`pareto_front.png`"]),
    (491, "model_size_vs_perf", "Model Size vs. Performance",
     "Compare parameter count vs. performance across models; compute "
     "parameters-per-point efficiency.",
     ["Parameter counts from model summaries (verified).",
      "Log-scale plot."],
     ["`size_vs_perf.csv`", "`size_vs_perf.png`"]),
    (492, "inference_latency_table", "Inference Latency Table",
     "Measure per-subject inference latency for every model (batch=1, "
     "CPU and GPU); produce a latency table for deployment discussion.",
     ["Warmup runs excluded; 100 repetitions.",
      "Hardware spec recorded."],
     ["`latency_table.csv`", "`latency_notes.md`"]),
    (493, "flops_estimation_report", "FLOPs Estimation Report",
     "Estimate FLOPs per forward pass per model (thop/fvcore or manual "
     "accounting); relate FLOPs to performance.",
     ["Method of FLOP counting documented.",
      "FLOPs-vs-performance scatter included."],
     ["`flops_table.csv`", "`flops_perf.png`"]),
    (494, "memory_peak_comparison", "Peak Memory Comparison",
     "Measure peak training GPU memory per model at the standard batch "
     "size; flag models exceeding a 12 GB budget.",
     ["Measured via torch.cuda.max_memory_allocated.",
      "Budget line marked on the plot."],
     ["`memory_table.csv`", "`memory_budget.png`"]),
]
for num, slug, title, desc, cons, outs in _EFF:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

# --- I. Reporting / leaderboard (T495-T500, 6) ------------------------------------------------------------------------
_REP = [
    (495, "final_leaderboard_generation", "Final Leaderboard Generation",
     "Generate the definitive benchmark leaderboard: all models x settings, "
     "mean +/- std, significance marks vs. the top model, in CSV + Markdown "
     "+ LaTeX.",
     ["Significance from the corrected tests (T447-T454).",
      "Formats: CSV, MD, LaTeX booktabs."],
     ["`leaderboard.csv`", "`leaderboard.md`", "`leaderboard.tex`"]),
    (496, "results_table_latex_export", "Results Tables LaTeX Export",
     "Export all evaluation tables (subgroups, robustness, efficiency) to "
     "publication-ready LaTeX with consistent formatting and booktabs "
     "style.",
     ["Bold-best rule documented and consistent.",
      "Every table compiles standalone."],
     ["`tables/*.tex`", "`tables_preview.pdf`"]),
    (497, "figure_reproduction_checklist", "Figure Reproduction Checklist",
     "Write the reproduction checklist: which script + config + seed "
     "regenerates every results figure/table, and verify one end-to-end "
     "re-run.",
     ["One row per artifact: script, config, seed, output hash.",
      "One artifact fully re-run as proof."],
     ["`REPRODUCE.md`", "`rerun_verification.json`"]),
    (498, "metric_definitions_audit", "Metric Definitions Audit",
     "Audit every reported metric: definition, averaging mode (macro/"
     "weighted/micro), implementation source, and edge-case behavior; "
     "publish the metric glossary.",
     ["Each metric linked to its code location.",
      "Inconsistencies found are listed with impact."],
     ["`metric_glossary.md`", "`metric_audit.csv`"]),
    (499, "benchmark_scorecard_summary", "Benchmark Scorecard Summary",
     "Produce the one-page benchmark scorecard: per-model radar/summary "
     "across accuracy, robustness, fairness, efficiency, and calibration "
     "dimensions.",
     ["Dimension scores normalized 0-1 with documented scaling.",
      "Radar chart per model."],
     ["`scorecard.md`", "`scorecard_radars.png`"]),
    (500, "evaluation_protocol_document", "Evaluation Protocol Document",
     "Write the evaluation protocol document: datasets, splits, seeds, "
     "metrics, statistical tests, and reporting rules, so a new model can "
     "be evaluated identically.",
     ["Versioned document; every number traceable to a task output.",
      "Includes the exact commands to run a new model."],
     ["`EVALUATION_PROTOCOL.md`"]),
]
for num, slug, title, desc, cons, outs in _REP:
    TASKS.append(_c(num, slug, title, desc, cons=cons, outs=outs))

assert len(TASKS) == 54, f"cross_model batch must be 54 tasks, got {len(TASKS)}"
