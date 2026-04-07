# USER.md - About My Human

- **Preferred name / how to address you**: Researcher (or you can use "you" directly in conversation)  
- **Pronouns**: they/them (neutral; adjust if told otherwise)  
- **Timezone**: Asia/Hong_Kong (HKT) — convert ALL times, logs, timestamps to HKT  
- **Location / Region**: Hong Kong  
- **Role / Focus**: Medical AI researcher working in neuroscience and clinical AI applications  
  - Main interests: brain imaging analysis, medical image processing, deep learning models for diagnosis/prognosis, public neuroimaging datasets, reproducible pipelines, clinical translation considerations  

## Communication Style
- Direct, technical, concise  
- Use precise scientific and medical terminology  
- Include code snippets, command lines, literature references, method comparisons when relevant  
- Default to English; occasional use of standard Chinese academic terms is acceptable if more precise  
- Minimal or no emojis unless emphasizing a light point  

## Important Preferences & Notes
- Deep learning framework preference: PyTorch (modular, well-documented code strongly preferred)  
- Frequently used datasets/domains: ADNI, UK Biobank, OpenNeuro, HCP, local clinical cohorts (when ethics-approved and de-identified)  
- High priority: ability to guide dependency installation, reproduce published models, handle real data processing pipelines  
- Output preferences:  
  - Tables for method/dataset/model comparisons  
  - Code + description for figures (matplotlib/seaborn/ plotly)  
  - Clear step-by-step plans for experiments and analyses  
- Pain points to remember: non-functional tools/skills, missing dependencies, agents failing to call tools, fabricated citations, ignoring reproducibility  

## Schedule / Work Pattern
- Often works evenings and late nights in HKT  
- Values clear progress tracking and structured weekly summaries  

## Harness Engineering Configuration (User Preferences)

### Self-Correction & Verification
- **Auto-correction enabled**: Yes (default)
- **Max auto-correction attempts**: 3 (halt and ask user after 3 failures)
- **Self-correction scope**: data integrity checks, environment verification, output validation
- **Behavior on verification failure**: Log error + suggest manual review (do not silently skip)

### Statistical Thresholds & Quality Gates
- **Data integrity checks** (enabled by default):
  - NaN/Inf tolerance: 0% (fail if any detected in critical outputs)
  - Value range deviation: flag if >10% of values outside expected bounds
  - Distribution shift (KL divergence): alert if >0.15 (warning), >0.25 (error)
  
- **Processing quality gates**:
  - Artifact removal success rate: target ≥95% (warn if <95%, fail if <85%)
  - Brain extraction Dice score: target ≥0.95 (warn if lower)
  - Motion parameter outliers: flag if >5% of frames exceed 0.5mm FD
  - Preprocessing convergence: require explicit success log before marking complete

- **Model inference thresholds**:
  - Prediction confidence: warn if <0.7 on binary classification
  - Latency drift: alert if inference time increases >20% vs. baseline
  - Output value bounds: error if predictions out of plausible range

### Logging & Persistence Strategy
- **Audit log format**: JSONL (JSON Lines) — one structured event per line
- **Audit log location**: `{workspace_root}/logs/audit_{YYYY-MM-DD}.jsonl` (rotated daily)
- **Metadata captured per event**:
  - Timestamp (ISO 8601 UTC + HKT offset)
  - Event type (phase_success, phase_failure, verification_pass, verification_fail, drift_detected, etc.)
  - Task/skill name and session ID
  - Relevant metrics (checksums, execution time, resource usage, error messages)
  - PII scrubbing: automatic redaction of patient IDs, file paths, email addresses

- **Checkpoint strategy**:
  - Auto-save checkpoint after each major phase (every 15–30 min for long tasks)
  - Checkpoint retention: keep last 5 checkpoints + current (auto-cleanup older files)
  - Compression: use LZ4 compression (fast + good ratio) for checkpoint storage
  - Checkpoint integrity: verify SHA256 hash before resuming

- **Log retention**:
  - Audit logs: keep for 90 days (exportable to archive before deletion)
  - Checkpoints: keep for 30 days (can be manually extended)
  - QC reports: keep indefinitely (stored alongside processed data)
  - Old logs movable to `logs/archive/` on request

- **Persistence triggers** (logs flushed to disk immediately):
  - End of phase execution (success or failure)
  - Completion of verification suite
  - Any error condition
  - Drift detection alert
  - Manual checkpoint request

### Experiment Reproducibility Preferences
- **Environment manifest capture**: Automatic (include Python version, packages, CUDA, OS)
- **Hash verification**: Enabled for all output artifacts (SHA256, stored alongside outputs)
- **Reproducibility report generation**: Automatic (after skill execution completes)
- **Re-run behavior**: Warn if re-running with different environment; prompt for explicit approval if environment mismatch detected

### Privacy & Security Defaults
- **PII redaction**: Enabled for all logs and reports (auto-detect and redact patient IDs, filenames, paths)
- **Docker sandboxing**: Preferred for containerized skills (enforce read-only root, capability dropping)
- **Permission model**: Principle of least privilege (restrict file access to explicit paths)
- **Data access logging**: Log all data file reads/writes with timestamp and brief reason

### Drift Detection & Monitoring
- **Continuous monitoring**: Enabled (monitor after every 50 inferences or per skill execution)
- **Supported detectors**: KL divergence (data distribution), latency shift (timing), failure rate (errors)
- **Alert thresholds** (can be overridden per-skill):
  - Data KL divergence: warn >0.1, error >0.2
  - Latency increase: warn if >20%, error if >50%
  - Failure rate: warn if >1%, error if >5%
- **Alert action**: Log to audit trail, generate drift report, ask user before continuing

---

## Environment Configuration

> This section is auto-populated by `installer/setup.py`. Update it whenever you re-run the installer or change your environment.

- **Setup type**: _(update after running installer — e.g. `conda`)_
- **Conda environment**: _(e.g. `neuroclaw`)_
- **Python path**: _(e.g. `/opt/conda/envs/neuroclaw/bin/python`)_
- **CUDA version**: _(e.g. `12.1`; or `cpu-only` if no GPU)_
- **PyTorch build**: _(e.g. `cu121`; or `cpu-only`)_
- **Default device**: _(e.g. `cuda:0`; or `cpu`)_
- **FSL home** (`FSLDIR`): _(e.g. `/usr/local/fsl`; or `null` if not installed)_
- **FreeSurfer home** (`FREESURFER_HOME`): _(e.g. `/usr/local/freesurfer`; or `null`)_
- **dcm2niix path**: _(e.g. `/usr/local/bin/dcm2niix`; or `null`)_
- **MATLAB path**: _(e.g. `/usr/local/MATLAB/R2024a/bin/matlab`; or `null`)_
- **LLM provider**: _(e.g. `openai` / `anthropic` / `local`)_
- **LLM model**: _(e.g. `gpt-4o`)_
- **LLM API key env var**: _(e.g. `OPENAI_API_KEY` — the variable name, not the value)_
- **Default BIDS root**: _(e.g. `~/data/bids`)_
- **Default output root**: _(e.g. `~/data/outputs`)_
- **Default n_jobs**: _(e.g. `4`)_

The authoritative values are stored in `neuroclaw_environment.json` at the workspace root.
Read that file (not this section) for programmatic access.

---

Update this file whenever new preferences or context are provided.