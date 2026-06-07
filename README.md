<div align="center">

<img src="materials/logo.png" alt="NeuroClaw Logo" width="200" />

# NeuroClaw: Closed-Loop Agentic AI for Executable and Reproducible Neuroimaging Research

<p align="center">
  <img src="docs/assets/logos/cuhk.png" alt="CUHK logo" height="50" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/logos/mgh.png" alt="Massachusetts General Hospital logo" height="50" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/logos/lehigh.png" alt="Lehigh University logo" height="50" />
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#-quick-start)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Skills](https://img.shields.io/badge/skills-86-purple)](skills)
[![arXiv](https://img.shields.io/badge/arXiv-2604.24696-b31b1b)](https://arxiv.org/abs/2604.24696)
[![Homepage](https://img.shields.io/badge/Project-Homepage-orange)](https://cuhk-aim-group.github.io/NeuroClaw/)
[![NeuroOracle](https://img.shields.io/badge/%F0%9F%A7%A0%20NeuroOracle-Live%20Demo-blue)](https://huggingface.co/spaces/zxcvb20001/NeuroOracle)

[中文版 README](README_zh.md)

<div align="center">

[Features](#-key-features) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Skills](#%EF%B8%8F-skill-quick-reference) • [Acknowledgments](#-acknowledgments)

</div>

</div>


## 📖 Overview

**NeuroClaw** is a research assistant for executable and reproducible neuroimaging research. Its core strength is **neuroimaging dataset and model adaptation**: turning raw scans into usable inputs quickly, and enabling medical practitioners to run deep learning models with minimal setup.

Neuroimaging datasets demand specialized preprocessing, and preprocessing quality directly determines model validity. Many workflows assume curated datasets, while MedicalClaw provides limited automation for open-source model execution (primarily large projects like TimesFM and AlphaFold), leaving users to spend significant time on environment configuration.

NeuroClaw prioritizes **data processing** and **model configuration/execution**. It ships with independent GUI and CLI interfaces for day-to-day use, and can also be installed as a reusable skill library inside agent projects such as OpenClaw, Hermes, and Claude Code.

**Notes**
- We constructed **NeuroBench** to benchmark multi-agent performance across neuroimaging workflows, especially raw data processing and model execution, and plan to refine and evaluate existing medical and general claw systems.
- Each SKILL.md ends with the author information; please open an issue to the corresponding author if you have questions.

---

## 🚀 Updates

- **[2026.05.23]**: NeuroBench now covers both data processing and model training/evaluation.
- **[2026.05.20]**: 7 atoms × 15 canonical tasks + 4 mediation chains in `neurooracle.atoms`.
- **[2026.05.15]**: NeuroOracle launched: knowledge-graph explorer plus hypothesis engine with live demo at https://huggingface.co/spaces/zxcvb20001/NeuroOracle.
- **[2026.05.06]**: Added 19 dataset and modality skills with companion scripts; all 86 skills enforce unified metadata (`layer`, `skill_type`, `dependencies`); skill_loader DAG validation ensures dependency graph correctness.
- **[2026.04.28]**: Our technical report is now available on arXiv: https://arxiv.org/abs/2604.24696
- **[2026.04.22]**: v1.0 released. Stable release with improvements and full documentation.
- **[2026.04.17]**: Our project homepage is now live. Welcome to visit: https://cuhk-aim-group.github.io/NeuroClaw/
- **[2026.04.08]**: NeuroBench released for multi-agent neuroimaging workflow evaluation.
- **[2026.04.02]**: v0.1 released with complete NeuroClaw framework and core functionality.

<a id="key-features"></a>
## ✨ Key Features

<div align="center">
  <img src="materials/framework.png" alt="NeuroClaw Framework Overview" style="width: 95%; max-width: 100%;" />
</div>

### 🔄 Data-Aware Orchestration
- **Dataset-Context Planning**: Organize capabilities around dataset structure, metadata, and workflow stage instead of simply "which tool to call"
- **Automatic Skill Recommendation**: Users specify the target dataset, and NeuroClaw recommends relevant skills and executable workflows
- **Preprocessing Constraint Awareness**: Dataset-specific modality availability and preprocessing requirements are considered during orchestration

#### Supported Dataset Overview

| Dataset | Supported Modalities | Additional Data | Cohort Scale | Official Link |
| :---: | --- | --- | --- | :--- |
| ABCD Study | T1w; T2w; dMRI; rs-fMRI; task-fMRI | Physical and mental health; substance use; culture/environment; neurocognition; biological data | Target cohort of ~11,500 children; full cohort releases through the NIMH Data Archive | https://abcdstudy.org/ |
| ABIDE | T1w; rs-fMRI | ASD/control phenotypic data | 1,112 datasets from 17 international sites | https://fcon_1000.projects.nitrc.org/indi/abide/ |
| ADHD-200 | T1w; rs-fMRI | Diagnostic status; ADHD symptom measures; demographics; medication history; QC measures | 776 participants/datasets across 8 imaging sites | https://fcon_1000.projects.nitrc.org/indi/adhd200/ |
| AIBL | T1w; PET (PiB, FDG, tau) | Cognitive assessments; blood biomarkers; lifestyle and demographic data; APOE genotype | ~1,100+ participants (healthy controls, MCI, AD) | https://aibl.csiro.au/ |
| AOMIC | T1w; rs-fMRI; task-fMRI | Personality traits (Big Five); fluid intelligence; demographic data | ~1,000+ participants | https://nilab-uva.github.io/AOMIC.github.io/ |
| ADNI | T1w; T2w; FLAIR; dMRI; rs-fMRI; PET | Genetics/omics data; clinical and cognitive assessments | ~2,000+ participants across ADNI phases | https://adni.loni.usc.edu/ |
| BOLD5000 | T1w; task-fMRI | Visual image stimuli; category and image metadata | 4 participants with 5,000-image visual fMRI sessions | https://bold5000-dataset.github.io/ |
| Cam-CAN | T1w; T2*w; rs-fMRI; task-fMRI; MEG | Cognitive, sensory, and health measures across the adult lifespan | ~700 participants ages 18-88 | https://www.cam-can.org/ |
| COBRE | T1w; rs-fMRI | Demographics; handedness; diagnostic information | 147 participants: 72 schizophrenia patients and 75 healthy controls | https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |
| DMT-HAR-MED | rs-fMRI | Psychedelic intervention conditions; behavioral and physiological measures | 40 participants in OpenNeuro ds006644 | https://openneuro.org/datasets/ds006644/versions/1.0.1 |
| HBN | T1w; T2w; dMRI; rs-fMRI; task-fMRI; EEG | Psychiatric, behavioral, cognitive, lifestyle, genetics, actigraphy | ~3,900+ released participants; target resource of at least 10,000 ages 5-21 | https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ |
| HCP Aging | T1w; T2w; dMRI; rs-fMRI; task-fMRI | Behavioral, cognitive, health, and demographic measures | ~700+ adults ages 36-100 | https://www.humanconnectome.org/study/hcp-lifespan-aging |
| HCP Development | T1w; T2w; dMRI; rs-fMRI; task-fMRI | Behavioral, cognitive, health, and demographic measures | ~600+ children and adolescents ages 5-21 | https://www.humanconnectome.org/study/hcp-lifespan-development |
| HCP Early Psychosis | T1w; T2w; dMRI; rs-fMRI; task-fMRI | Diagnostic, clinical, behavioral, and cognitive measures | ~250 early psychosis and control participants | https://www.humanconnectome.org/study/hcp-early-psychosis |
| HCP Young Adult | T1w; T2w; dMRI; rs-fMRI; task-fMRI | Behavioral and cognitive measures | ~1,200 young adult participants | https://www.humanconnectome.org/study/hcp-young-adult |
| IXI | T1w; T2w; MRA | Healthy brain MRI from three London hospitals | ~600 subjects | https://brain-development.org/ixi-dataset/ |
| MS Challenge | T1w; T2w; FLAIR; PD | Expert manual lesion segmentations for MS benchmarking | 5 MS patients with multiple longitudinal timepoints | https://smart-stats-tools.org/lesion-challenge |
| MND | rs-fMRI; task-fMRI | Motor neuron disease diagnosis and clinical measures | 59 participants in OpenNeuro ds005874 | https://openneuro.org/datasets/ds005874/versions/1.1.0 |
| Natural Scenes Dataset | T1w; task-fMRI | Natural image stimuli; behavioral responses; image annotations | 8 participants with dense repeated visual fMRI | https://naturalscenesdataset.org/ |
| NIFD | T1w; fMRI; DTI; PET | FTD clinical and cognitive data; UCSF Memory and Aging Center | Frontotemporal dementia and related disorders cohorts | https://ida.loni.usc.edu/ |
| OASIS | T1w; PET (PiB) | Clinical and cognitive assessments; dementia diagnosis; demographic data | Cross-sectional (400+) and longitudinal (150+) participants ages 18-96 | https://www.oasis-brains.org/ |
| PNC | T1w; dMRI; ASL; rs-fMRI; task-fMRI | Genotyping; clinical and neuropsychiatric assessment; Computerized Neurocognitive Battery | >9,500 youth cohort; 1,445 participants with neuroimaging | https://www.med.upenn.edu/bbl/philadelphianeurodevelopmentalcohort.html |
| PPMI | T1w; rs-fMRI; DAT-SPECT; PET | Clinical, genetic, biospecimen, and wearable sensor data for Parkinson's disease | ~2,000+ participants across 30+ clinical sites worldwide | https://www.ppmi-info.org/ |
| REST-meta-MDD | rs-fMRI | MDD diagnosis; clinical and demographic measures | 2,428 participants across 25 cohorts | http://rfmri.org/REST-meta-MDD |
| SEED-IV | EEG | Emotion labels across four affective categories; trial-level session metadata | 15 subjects across 3 sessions for emotion decoding benchmarks | https://bcmi.sjtu.edu.cn/home/seed/ |
| SEED-VIG | EEG | Vigilance/fatigue labels; continuous alertness annotations; behavioral metadata | 23 subjects in sustained-attention driving-style vigilance recordings | https://bcmi.sjtu.edu.cn/home/seed/ |
| TCP | rs-fMRI | Psychiatric diagnostic interviews; cognitive and clinical assessments | 245 transdiagnostic participants | https://openneuro.org/datasets/ds004215 |
| UCLA CNP | T1w; dMRI; rs-fMRI; task-fMRI | Diagnostic groups; neuropsychological and phenotypic assessments | 272 participants in OpenNeuro ds000030 | https://openneuro.org/datasets/ds000030 |
| UK Biobank | T1w; T2w; FLAIR; dMRI; rs-fMRI; task-fMRI | Genotype/genomic data; questionnaires; hospital records; environmental data; sociodemographic data; physical measures | ~50,000 participants with multimodal imaging data | https://www.ukbiobank.ac.uk/ |

### 🎯 Executability and Reproducibility
- **Automatic Dependency Management**: No manual installation needed; the system detects and resolves dependencies
- **True Model Execution**: Beyond sharing docs, it guides and executes model reproduction
- **Environment Isolation**: Virtual environments and containerization avoid system pollution
- **Verifiable Processes**: Complete logging and result tracking
- **Shadow Checkpoints**: Git-based filesystem snapshots for rollback and diff comparison without polluting the project repository
- **Subagent Orchestration**: Spawns specialized subagents (biostatistician, clinical neuroscientist, methodology expert) for multi-perspective task execution
- **Reflective Learning**: Automatic reflection on tool failures and task completion, with persistent memory for cross-session learning

### 🧠 End-to-End Research Coverage
- **Literature Review**: arXiv search, PubMed retrieval, academic resource integration
- **Experiment Design**: Scientific literature analysis, methodology evaluation, research proposal generation
- **Data Processing**: Multi-format conversion (DICOM ↔ NIfTI), automated preprocessing pipelines
- **Model Execution**: Run published research models, deep learning framework integration
- **Result Visualization**: Scientific data visualization, statistical chart generation
- **Paper Writing**: Auto-generated drafts, format standardization

### 🤝 Flexible Integration
- **NeuroClaw works as a standalone research assistant** with its own GUI and CLI, so researchers can use it directly without depending on another host project.
- `skills/`, `materials/`, `USER.md`, and `SOUL.md` can also be installed as a reusable skill library in existing agent systems such as OpenClaw, Hermes, and Claude Code.
- The bundled `core/` engine provides an integrated agent loop, skill loader, and tool runtime for standalone deployments.
- Non-neuroscience connectors (WhatsApp, Telegram, Slack, calendar, e-commerce, SaaS auth)
  are disabled by default via `core/config/features.json` and can be re-enabled if needed.

---

<a id="quick-start"></a>
## 🚀 Quick Start

### Prerequisites
- Python >= 3.10
- Git
- *(Optional)* Conda/Mamba for environment isolation
- *(Optional)* `nvidia-smi` / `nvcc` for GPU support
- *(Recommended for Web UI attachments)* `pypdf`, `python-docx`, `openpyxl`, `python-pptx`

> **NeuroClaw runs as a standalone research assistant** with its own GUI and CLI.
> The bundled installer configures everything, including your Python environment,
> CUDA version, neuroimaging toolchain, and LLM backend.

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/CUHK-AIM-Group/NeuroClaw.git
   cd NeuroClaw
   ```

2. **Run the Setup Wizard**
   ```bash
   python installer/setup.py
   ```
  This installs the standalone NeuroClaw environment for both the GUI and CLI workflows.
  The wizard will walk you through:
  - Python runtime (system / conda / Docker)
  - CUDA / GPU configuration and optional PyTorch install
  - Neuroscience toolchain paths (FSL, FreeSurfer, dcm2niix, etc.)
  - LLM backend selection (OpenAI, DeepSeek, MiniMax, Kimi/Moonshot, Qwen/DashScope, Baichuan, Zhipu GLM, Doubao/Ark, OpenRouter, Together, Groq, Fireworks, Ollama, llama.cpp, Anthropic, or local model)
  - Default BIDS and output directories
  - Web UI dependencies and attachment parsers (PDF/DOCX/XLSX/PPTX)

   Settings are saved to `neuroclaw_environment.json` and loaded automatically on every future session.
   Setup does not ask for an API key. Pass the key only at runtime with `--api-key`, or export the configured environment variable before startup.

   For a quick non-interactive setup with auto-detected defaults:
   ```bash
   python installer/setup.py --non-interactive
   ```

    If you skipped optional Web UI dependencies, install them manually:
    ```bash
    pip install "fastapi[standard]" uvicorn pypdf python-docx openpyxl python-pptx
    ```

3. **Start NeuroClaw**
   
   *Option A — Interactive REPL (terminal)*
   ```bash
   python core/agent/main.py --api-key "$OPENAI_API_KEY"
   ```

   *Option B — Browser Web UI (recommended)*
   ```bash
   python core/agent/main.py --web --api-key "$OPENAI_API_KEY"
   ```
   Then open **http://localhost:7080** in your browser. The Web UI features a chat interface, skills sidebar, markdown rendering, and code syntax highlighting.

  If you prefer environment variables, export the provider-specific key first and start NeuroClaw without `--api-key`.
  Built-in OpenAI-compatible provider profiles:
  - `deepseek`: `DEEPSEEK_API_KEY`, default endpoint `https://api.deepseek.com`
  - `minimax`: `MINIMAX_API_KEY`, default endpoint `https://api.minimaxi.com/v1`
  - `kimi` / `moonshot`: `MOONSHOT_API_KEY`, default endpoint `https://api.moonshot.cn/v1`
  - `qwen` / `dashscope`: `DASHSCOPE_API_KEY`, default endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `baichuan`: `BAICHUAN_API_KEY`, default endpoint `https://api.baichuan-ai.com/v1`
  - `zhipu` / `glm`: `ZHIPUAI_API_KEY`, default endpoint `https://open.bigmodel.cn/api/paas/v4`
  - `doubao` / `ark`: `ARK_API_KEY`, default endpoint `https://ark.cn-beijing.volces.com/api/v3`
  - `openrouter`: `OPENROUTER_API_KEY`, default endpoint `https://openrouter.ai/api/v1`
  - `together`: `TOGETHER_API_KEY`, default endpoint `https://api.together.xyz/v1`
  - `groq`: `GROQ_API_KEY`, default endpoint `https://api.groq.com/openai/v1`
  - `fireworks`: `FIREWORKS_API_KEY`, default endpoint `https://api.fireworks.ai/inference/v1`
  - `ollama`: no API key required, default endpoint `http://localhost:11434/v1`
  - `llamacpp`: no API key required, default endpoint `http://localhost:8080/v1`

    Web UI attachment parsing currently supports these file types:
    - Text/config/code: `.txt`, `.md`, `.markdown`, `.json`, `.yaml`, `.yml`, `.csv`, `.tsv`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.sh`, `.bash`, `.zsh`, `.sql`, `.html`, `.css`, `.xml`, `.log`, `.rst`, `.ini`, `.toml`, `.cfg`
    - Documents: `.pdf`, `.docx`, `.xlsx`, `.pptx`

    The file picker in the Web UI only allows these supported formats.

   To use a custom port or bind to all interfaces (e.g., for remote access):
   ```bash
  python core/agent/main.py --web --port 8080 --host 0.0.0.0 --api-key "$OPENAI_API_KEY"
   ```

4. **Install NeuroClaw into other agent systems**

   To make a host agent operate in NeuroClaw mode with NeuroClaw catalogs,
   NeuroOracle, NeuroBench, model workflows, and neuroimaging skills:

   | Host agent | Install command | Installed integration |
   |---|---|---|
   | Codex | `python installer/install_agent_integration.py --target codex` | `~/.codex/skills/neuroclaw/` |
   | Claude Code | `python installer/install_agent_integration.py --target claude-code` | `~/.claude/skills/neuroclaw/` |
   | Cursor | `python installer/install_agent_integration.py --target cursor --scope project` | `.cursor/rules/neuroclaw.mdc` plus generated references |
   | OpenClaw | `python installer/install_agent_integration.py --target openclaw` | `~/.openclaw/skills/neuroclaw/` |
   | Hermes | `python installer/install_agent_integration.py --target hermes` | `~/.hermes/skills/neuroclaw/` |
   | WorkBuddy | `python installer/install_agent_integration.py --target workbuddy` | `~/.workbuddy/skills/neuroclaw/` |
   | QClaw | `python installer/install_agent_integration.py --target qclaw` | `~/.qclaw/skills/neuroclaw/` |

   To install the Codex and Claude Code integrations together:
   ```bash
   python installer/install_agent_integration.py --target both
   ```

   To generate all supported integrations:
   ```bash
   python installer/install_agent_integration.py --target all
   ```

   If a host agent uses a different import location, export the generated skill
   pack and import or copy it manually:
   ```bash
   python installer/install_agent_integration.py --target all --export ./dist/agent-integrations
   ```

   After installation, ask the host agent to "use NeuroClaw" or "enter NeuroClaw
   mode" for neuroimaging and autoresearch work.

<div align="center">
  <img src="materials/index.png" alt="NeuroClaw Feature Overview" style="width: 80%; max-width: 100%;" />
</div>

> Note: We provide benchmark run results and per-model outputs under `materials/benchmark_results/`. These artifacts can be used as practical references when running NeuroClaw benchmarks or reproducing model outputs.

### Verify Installation
```bash
# Check that the environment file is valid
python installer/setup.py --check

# List registered neuroscience skills (Python)
python -c "
from core.skill_loader.loader import SkillLoader
from pathlib import Path
skills = SkillLoader(Path('skills')).load_all()
for s in skills:
    print(s['name'])
"
```

### Benchmark Evaluation

NeuroBench tasks live under `neurobench/`, and each task directory contains a `task.md` instruction file.

NeuroBench currently accepts these benchmark configurations:
- `with-skills`: the agent can use the skills loaded from `skills/`
- `no-skills`: the baseline run without skills
- `with-skills` + `no-skills` paired comparison: enable `--benchmark-compare-skills` to run both variants for the same task set

Benchmark scoring is handled separately with `--score-benchmark`: it reads reports in `output/`, applies a GPT-5.4 weighted rubric, and generates numeric scores for planning completeness, tool/skill reasonableness, and command/code correctness. For fairness, each task case is scored in one batch across all comparable models to reduce scoring-standard drift. Skill-call counts are recorded separately and used for efficiency analysis.

To score existing benchmark reports:
```bash
python core/agent/main.py --score-benchmark
```

To speed up scoring on larger runs:
```bash
python core/agent/main.py --score-benchmark --score-workers 8
```

**Web benchmark mode**
```bash
python core/agent/main.py --web --benchmark
```

**CLI benchmark batch runner**
```bash
python core/agent/main.py --benchmark
```

To run the paired skill comparison in CLI mode:
```bash
python core/agent/main.py --benchmark --benchmark-compare-skills
```

In CLI benchmark mode, NeuroClaw will ask for:
- the benchmark directory path
- the benchmark model name

Then it will:
- read all `task.md` files recursively from that directory
- sort tasks alphabetically by task folder name
- run tasks one by one without asking for intermediate confirmation
- print progress in the terminal only
- save reports under `output/<model_name>/`, with one markdown report per case and run

The benchmark reports include the solution thinking, skills used, skill-call counts, and the commands or code that were used or suggested.

---

<a id="project-structure"></a>
## 📁 Project Structure

```
NeuroClaw/
├── README.md                       # This file
├── USER.md                         # User-defined configurations and preferences
├── SOUL.md                         # System behavior guidelines and principles
│
├── core/                           # Self-contained NeuroClaw engine (no OpenClaw required)
│   ├── agent/                      # LLM conversation loop and tool-call dispatcher
│   │   └── main.py                 # Entry point; --web flag starts the Web UI
│   ├── web/                        # Browser-based Web UI (FastAPI + WebSocket)
│   │   ├── server.py               # FastAPI app: WebSocket chat, /api/skills, /api/env
│   │   └── static/
│   │       └── index.html          # Dark-theme chat interface (markdown + syntax highlight)
│   ├── skill_loader/               # Skill scanner: reads skills/*/SKILL.md and registers tools
│   │   └── loader.py
│   ├── tool-runtime/               # Executes handler.js / Python handlers
│   │   └── runtime.py
│   ├── session/                    # Session persistence and context-window compression
│   │   └── manager.py
│   ├── checkpoint/                 # Shadow-git filesystem checkpoint manager
│   │   ├── __init__.py
│   │   └── manager.py
│   └── config/
│       └── features.json           # Feature toggles (disable WhatsApp/Slack/etc.; enable web_ui)
│
├── installer/                      # Custom setup wizard (replaces OpenClaw's default installer)
│   ├── setup.py                    # Entry point: python installer/setup.py
│   ├── config_wizard.py            # Interactive 6-step configuration wizard (incl. Web UI deps)
│   └── neuro_defaults.json         # Neuroscience-specific default template
│
├── skills/                         # 86 skills: base (38) / subagent (42) / interface (6)
│   ├── abide-skill/
│   ├── aibl-skill/
│   ├── abcd-skill/
│   ├── academic-research-hub/
│   ├── adhd200-skill/
│   ├── adni-skill/
│   ├── aomic-skill/
│   ├── asl-skill/
│   ├── bids-organizer/
│   ├── beautiful-log/
│   ├── bnt/
│   ├── bold5000-skill/
│   ├── brain-visualization/
│   ├── brain_gnn/
│   ├── claw-shell/
│   ├── cobre-skill/
│   ├── camcan-skill/
│   ├── combraintf/
│   ├── conda-env-manager/
│   ├── conn-tool/
│   ├── dcm2nii/
│   ├── dependency-planner/
│   ├── detrending/
│   ├── dictlearning/
│   ├── dipy-tool/
│   ├── dmt-har-med-skill/
│   ├── docker-env-manager/
│   ├── dwi-skill/
│   ├── eeg-skill/
│   ├── experiment-controller/
│   ├── filtering/
│   ├── fm_app/
│   ├── fmri-skill/
│   ├── fmriprep-tool/
│   ├── freesurfer-tool/
│   ├── fsl-tool/
│   ├── git-essentials/
│   ├── git-workflows/
│   ├── glm/
│   ├── harmonization-tool/
│   ├── harness-core/
│   ├── hbn-skill/
│   ├── hcpa-skill/
│   ├── hcpd-skill/
│   ├── hcpep-skill/
│   ├── hcpya-skill/
│   ├── hcppipeline-tool/
│   ├── hierarchical/
│   ├── ibgnn/
│   ├── ica/
│   ├── ixi-skill/
│   ├── kmeans/
│   ├── knowledge-graph-builder/
│   ├── lggnn/
│   ├── method-design/
│   ├── mne-eeg-tool/
│   ├── meg-skill/
│   ├── mnd-skill/
│   ├── mschallenge-skill/
│   ├── multi-search-engine/
│   ├── neurostorm/
│   ├── nibabel-skill/
│   ├── nifd-skill/
│   ├── nsd-skill/
│   ├── nii2dcm/
│   ├── nilearn-tool/
│   ├── oasis-skill/
│   ├── overleaf-skill/
│   ├── paper-writing/
│   ├── pet-skill/
│   ├── pnc-skill/
│   ├── ppmi-skill/
│   ├── qsiprep-tool/
│   ├── research-idea/
│   ├── rest-mneta-mdd-skill/
│   ├── run_models/
│   ├── seed-iv-skill/
│   ├── seed-vig-skill/
│   ├── skill-updater/
│   ├── smri-skill/
│   ├── spacenet/
│   ├── svm/
│   ├── tcp-skill/
│   ├── ukb-skill/
│   ├── ucla-cnp-skill/
│   └── wmh-segmentation/
│
├── neurobench/                    # NeuroBench evaluation tasks (T01-T120)
│   ├── T00_installer_validation/   # Validates installer output
│   └── …
│
├── materials/                      # Research materials, benchmark run results, and model outputs
│   ├── CVPR_2026/
│   └── benchmark_results/
│
└── LICENSE                         # License

```

---

<a id="skill-quick-reference"></a>
## 🛠️ Skill Quick Reference

> **Tip**: Click the ℹ️ icon on any skill card in the Web UI to view expanded documentation, usage examples, and recent execution logs.

### Base Layer
| Skill | Function | Status |
|------|----------|--------|
| `dcm2nii` | DICOM → NIfTI conversion with metadata support | ✅ |
| `nii2dcm` | NIfTI → DICOM conversion for clinical interoperability | ✅ |
| `git-essentials` | Core Git commands for collaboration | ✅ |
| `git-workflows` | Advanced Git workflows (rebase/worktree/bisect) | ✅ |
| `multi-search-engine` | Multi-engine web search without API keys | ✅ |
| `conda-env-manager` | Conda environment lifecycle management | ✅ |
| `docker-env-manager` | Docker environment management | ✅ |
| `dependency-planner` | Dependency planning and safe installation workflow | ✅ |
| `claw-shell` | Safe shell execution gateway via dedicated session | ✅ |
| `overleaf-skill` | Overleaf sync and collaborative manuscript operations | ✅ |
| `academic-research-hub` | Multi-source academic search and paper retrieval | ✅ |
| `bids-organizer` | Base skill for organizing raw data into BIDS structure | ✅ |
| `beautiful-log` | Export clean User/NeuroClaw dialogue into beautiful HTML logs | ✅ |
| `knowledge-graph-builder` | Build domain knowledge graphs from literature and databases | ✅ |
| `skill-updater` | Skill updater and management utilities | ✅ |

### Interface Layer (Task Orchestration)
| Skill | Function | Status |
|------|----------|--------|
| `research-idea` | Brainstorms and generates research ideas from literature | ✅ |
| `method-design` | Formalizes network architecture and derives theoretical components | ✅ |
| `experiment-controller` | Finds and executes reproducible research experiments | ✅ |
| `paper-writing` | Generates hierarchical manuscript drafts from IDEA/METHOD/EXPERIMENT | ✅ |

### Subagent Layer
Subagent in NeuroClaw includes four categories: **tool**, **model**, **dataset**, and **modality**.

#### Tool
| Skill | Function | Status |
|------|----------|--------|
| `brain-visualization` | Publication-ready figures and 3D assets (connectomes, atlas summaries, FreeSurfer PLY) | ✅ |
| `harmonization-tool` | Cross-site / cross-scanner feature harmonization (ComBat, ComBat-GAM, CovBat, site-as-covariate) with site-stratified and leave-site-out splitters; required for honest mega-analysis across multi-site cohorts | ✅ |
| `harness-core` | Core harness SDK: verification, checkpointing, drift detection, audit logging | ✅ |
| `mne-eeg-tool` | Base-layer MNE-Python implementation for EEG | ✅ |
| `fsl-tool` | FSL-based sMRI/fMRI/DWI processing utilities | ✅ |
| `fmriprep-tool` | fMRIPrep pipeline wrapper and execution | ✅ |
| `qsiprep-tool` | qsiPrep pipeline wrapper for diffusion MRI | ✅ |
| `hcppipeline-tool` | HCP-style processing pipeline utilities | ✅ |
| `dipy-tool` | Diffusion MRI processing via DIPY | ✅ |
| `nibabel-skill` | Low-level neuroimaging I/O and geometry handling (NIfTI, affine, FreeSurfer I/O) | ✅ |
| `nilearn-tool` | Fast neuroimaging feature extraction and decoding prep | ✅ |
| `conn-tool` | Functional connectivity computation and analysis | ✅ |
| `freesurfer-tool` | FreeSurfer-based MRI processing and segmentation | ✅ |

#### Model
| Skill | Function | Status |
|------|----------|--------|
| `run_models` | Model registry and model execution orchestration | ✅ |
| `wmh-segmentation` | White matter hyperintensity segmentation (MARS-WMH nnU-Net) | ✅ |
| `brain_gnn` | BrainGNN: graph neural network for fMRI classification | ✅ |
| `bnt` | BrainNetworkTransformer: dense FC Transformer with DEC pooling for phenotype prediction | ✅ |
| `combraintf` | Com-BrainTF: community-aware two-level Transformer over dense FC matrices | ✅ |
| `ibgnn` | IBGNN: interpretable PyG-based GNN with MLP message function and edge-mask explainer | ✅ |
| `lggnn` | LG-GNN: PyG-based GNN with Self-Attention Brain Pooling and mutual-information regularization | ✅ |
| `fm_app` | FM-APP: multi-stage phenotype prediction with fMRI+sMRI | ✅ |
| `neurostorm` | NeuroStorm: neuroimaging foundation model | ✅ |
| `glm` | Classical first-level and second-level GLM for task-fMRI activation and group inference | ✅ |
| `ica` | Resting-state network decomposition via independent component analysis | ✅ |
| `dictlearning` | Sparse resting-state network decomposition via dictionary learning | ✅ |
| `svm` | Classical neuroimaging disease classification with ROI/tabular features | ✅ |
| `spacenet` | Voxel-wise neuroimaging disease classification with sparse coefficient maps | ✅ |
| `kmeans` | Brain parcellation via K-means clustering | ✅ |
| `hierarchical` | Multi-scale brain parcellation via hierarchical clustering | ✅ |
| `filtering` | Temporal filtering for neuroimaging signal denoising | ✅ |
| `detrending` | Temporal drift removal for neuroimaging signal denoising | ✅ |

#### Dataset
| Skill | Function | Status |
|------|----------|--------|
| `abide-skill` | ABIDE dataset download, BIDS staging, and sMRI/rs-fMRI processing | ✅ |
| `aibl-skill` | AIBL dataset access, BIDS staging, and sMRI/PET processing | ✅ |
| `abcd-skill` | ABCD Study dataset download, BIDS staging, and multimodal processing | ✅ |
| `adhd200-skill` | ADHD-200 dataset download, BIDS staging, and sMRI/rs-fMRI processing | ✅ |
| `adni-skill` | ADNI dataset automated processing workflow | ✅ |
| `aomic-skill` | AOMIC dataset validation, BIDS staging, and sMRI/rs-fMRI/task-fMRI processing | ✅ |
| `bold5000-skill` | BOLD5000 dataset BIDS validation and visual task-fMRI processing | ✅ |
| `camcan-skill` | Cam-CAN dataset BIDS validation, multimodal sMRI/rs-fMRI/task-fMRI/dMRI processing | ✅ |
| `cobre-skill` | COBRE dataset BIDS staging and schizophrenia-control fMRI processing | ✅ |
| `dmt-har-med-skill` | DMT-HAR-MED dataset BIDS validation and psychedelic rs-fMRI processing | ✅ |
| `hbn-skill` | HBN dataset download, BIDS staging, and multimodal sMRI/fMRI/dMRI/EEG processing | ✅ |
| `hcpa-skill` | HCP Aging dataset download, BIDS staging, and multimodal sMRI/fMRI/dMRI processing | ✅ |
| `hcpd-skill` | HCP Development dataset download, BIDS staging, and multimodal sMRI/fMRI/dMRI processing | ✅ |
| `hcpep-skill` | HCP Early Psychosis dataset download, BIDS staging, and multimodal sMRI/fMRI/dMRI processing | ✅ |
| `hcpya-skill` | HCP Young Adult (HCP1200) dataset download, BIDS staging, and multimodal sMRI/fMRI/dMRI processing | ✅ |
| `ixi-skill` | IXI dataset BIDS validation and multimodal sMRI/MRA/dMRI processing | ✅ |
| `mnd-skill` | MND dataset BIDS validation, rs-fMRI/task-fMRI processing, and phenotype extraction | ✅ |
| `mschallenge-skill` | MS Lesion Challenge BIDS validation, lesion analysis, and longitudinal tracking | ✅ |
| `nsd-skill` | Natural Scenes Dataset BIDS validation, task-fMRI processing, and COCO stimulus extraction | ✅ |
| `nifd-skill` | NIFD dataset BIDS validation, multimodal sMRI/rs-fMRI/dMRI processing for frontotemporal dementia | ✅ |
| `oasis-skill` | OASIS dataset BIDS validation, sMRI processing, and phenotype extraction for aging/AD research | ✅ |
| `pnc-skill` | PNC dataset BIDS validation, multimodal sMRI/rs-fMRI/task-fMRI/dMRI processing for developmental studies | ✅ |
| `ppmi-skill` | PPMI dataset BIDS validation, multimodal sMRI/rs-fMRI/dMRI processing for Parkinson's disease | ✅ |
| `rest-mneta-mdd-skill` | REST-meta-MDD multi-site rs-fMRI processing, site harmonization, and depression phenotype extraction | ✅ |
| `seed-iv-skill` | SEED-IV EEG emotion recognition (4 emotions), feature extraction, and classification | ✅ |
| `seed-vig-skill` | SEED-VIG EEG vigilance/fatigue detection, feature extraction, and drowsiness classification | ✅ |
| `tcp-skill` | Transdiagnostic Connectome Project BIDS validation, multimodal sMRI/rs-fMRI/dMRI processing | ✅ |
| `ucla-cnp-skill` | UCLA CNP BIDS validation, multimodal sMRI/task-fMRI/dMRI processing, multi-disorder phenotyping | ✅ |
| `ukb-skill` | UKB brain imaging automated processing workflow | ✅ |

#### Modality
| Skill | Function | Status |
|------|----------|--------|
| `eeg-skill` | EEG preprocessing and feature extraction workflows | ✅ |
| `fmri-skill` | Functional MRI preprocessing and analysis workflows | ✅ |
| `smri-skill` | Structural MRI preprocessing and analysis workflows | ✅ |
| `dwi-skill` | Diffusion MRI preprocessing and analysis workflows | ✅ |
| `pet-skill` | PET imaging workflows (SUVR computation, reference regions, PVC) | ✅ |
| `asl-skill` | ASL perfusion MRI workflows (CBF quantification, Buxton model) | ✅ |
| `meg-skill` | MEG processing workflows (source localization, time-frequency, connectivity) | ✅ |

**Legend**: ✅ Implemented | 🏗️ In Development | ⏳ Planned


---

<a id="acknowledgments"></a>
## 🙏 Acknowledgments

Thanks to:
- [OpenClaw](https://github.com/openclaw/openclaw)
- [Hermes](https://github.com/nousresearch/hermes-agent)
- [Claude Code](https://github.com/anthropics/claude-code)
- [Karcen/rs-fMRI-Pipeline-Tutorial](https://github.com/Karcen/rs-fMRI-Pipeline-Tutorial)
- [nature-skills](https://github.com/Yuan1z0825/nature-skills)
- Open-source neuroscience tools community (MNE-Python, FreeSurfer, FSL, etc.)
