<div align="center">

<img src="materials/logo.png" alt="NeuroClaw Logo" width="200" />

# NeuroClaw: From Raw Data to Reproducible Models

[中文版 README](README_zh.md)

<div align="center">

[Features](#-key-features) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Skills](#%EF%B8%8F-skill-quick-reference) • [Acknowledgments](#-acknowledgments)

</div>

</div>


## 📖 Overview

**NeuroClaw** is a neuroscience-first platform built on the [OpenClaw](https://github.com/openclaw/openclaw) framework. Its core strength is **neuroimaging dataset and model adaptation**: turning raw scans into usable inputs quickly, and enabling medical practitioners to run deep learning models with minimal setup.

Neuroimaging datasets demand specialized preprocessing, and preprocessing quality directly determines model validity. Many workflows assume curated datasets, while MedicalClaw provides limited automation for open-source model execution (primarily large projects like TimesFM and AlphaFold), leaving users to spend significant time on environment configuration.

NeuroClaw prioritizes **data processing** and **model configuration/execution**. It remains an end-to-end Claw system, but for neuroscience its center of gravity is data and models.

**Notes**
- We constructed **NeuroBench** to benchmark multi-agent performance across neuroimaging workflows, especially raw data processing and model execution, and plan to refine and evaluate existing medical and general claw systems.
- Each SKILL.md ends with the author information; please open an issue to the corresponding author if you have questions.


## 🚀 Updates

- **[2026.04.06]**: We begin constructing NeuroBench for multi-agent framework evaluation.
- **[2026.04.02]**: v0.1 released with complete NeuroClaw framework and core functionality.

<a id="key-features"></a>
## ✨ Key Features
### 🔄 Dataset-First Architecture
Organize capabilities around "which dataset to process" instead of "which tool to call":
- **ADNI Dataset** → Integrated ADNI standardized processing pipeline
- **UK Biobank** → Environment-aware deployment adaptation
- **Other Public Datasets** → Quick-start tool chains

Users simply specify the target dataset, and the system automatically recommends and orchestrates relevant skills.

### 🎯 Executability and Reproducibility
- **Automatic Dependency Management**: No manual installation needed; the system detects and resolves dependencies
- **True Model Execution**: Beyond sharing docs, it guides and executes model reproduction
- **Environment Isolation**: Virtual environments and containerization avoid system pollution
- **Verifiable Processes**: Complete logging and result tracking

### 🧠 End-to-End Research Coverage
- **Literature Review**: arXiv search, PubMed retrieval, academic resource integration
- **Experiment Design**: Scientific literature analysis, methodology evaluation, research proposal generation
- **Data Processing**: Multi-format conversion (DICOM ↔ NIfTI), automated preprocessing pipelines
- **Model Execution**: Run published research models, deep learning framework integration
- **Result Visualization**: Scientific data visualization, statistical chart generation
- **Paper Writing**: Auto-generated drafts, format standardization

### 🤝 OpenClaw Compatibility
- **NeuroClaw is now self-contained** — OpenClaw does not need to be installed separately.
  The bundled `core/` engine provides the same agent loop, skill loader, and tool runtime.
- `skills/`, `materials/`, `USER.md`, and `SOUL.md` remain fully compatible with existing
  OpenClaw workspaces if you still want to use NeuroClaw as an add-on.
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

> **NeuroClaw is now self-contained** — OpenClaw does not need to be installed separately.
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
   The wizard will walk you through:
   - Python runtime (system / conda / Docker)
   - CUDA / GPU configuration and optional PyTorch install
   - Neuroscience toolchain paths (FSL, FreeSurfer, dcm2niix, etc.)
   - LLM backend (OpenAI, Anthropic, or local model)
   - Default BIDS and output directories

   Settings are saved to `neuroclaw_environment.json` and loaded automatically on every future session.

   For a quick non-interactive setup with auto-detected defaults:
   ```bash
   python installer/setup.py --non-interactive
   ```

3. **Start NeuroClaw**

   **Option A — Interactive REPL (terminal)**
   ```bash
   python core/agent/main.py
   ```

   **Option B — Browser Web UI** *(recommended)*
   ```bash
   python core/agent/main.py --web
   ```
   Then open **http://localhost:7080** in your browser. The Web UI features a chat interface, skills sidebar, markdown rendering, and code syntax highlighting.

   To use a custom port or bind to all interfaces (e.g., for remote access):
   ```bash
   python core/agent/main.py --web --port 8080 --host 0.0.0.0
   ```

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

---

<a id="project-structure"></a>
## 📁 Project Structure

```
NeuroClaw/
├── README.md                       # This file
├── USER.md                         # User-defined configurations and preferences
├── SOUL.md                         # System behavior guidelines and principles
├── neuroclaw_environment.json      # Generated by installer — runtime config (Python, CUDA, toolchain, LLM)
│
├── core/                           # Self-contained NeuroClaw engine (no OpenClaw required)
│   ├── agent/                      # LLM conversation loop and tool-call dispatcher
│   │   └── main.py                 # Entry point; --web flag starts the Web UI
│   ├── web/                        # Browser-based Web UI (FastAPI + WebSocket)
│   │   ├── server.py               # FastAPI app: WebSocket chat, /api/skills, /api/env
│   │   └── static/
│   │       └── index.html          # Dark-theme chat interface (markdown + syntax highlight)
│   ├── skill-loader/               # Skill scanner: reads skills/*/SKILL.md and registers tools
│   │   └── loader.py
│   ├── tool-runtime/               # Executes handler.js / Python handlers
│   │   └── runtime.py
│   ├── session/                    # Session persistence and context-window compression
│   │   └── manager.py
│   └── config/
│       └── features.json           # Feature toggles (disable WhatsApp/Slack/etc.; enable web_ui)
│
├── installer/                      # Custom setup wizard (replaces OpenClaw's default installer)
│   ├── setup.py                    # Entry point: python installer/setup.py
│   ├── config_wizard.py            # Interactive 6-step configuration wizard (incl. Web UI deps)
│   └── neuro_defaults.json         # Neuroscience-specific default template
│
├── skills/                         # Flat skill directory
│   ├── academic-research-hub/
│   ├── adni-skill/
│   ├── bids-organizer/
│   ├── beautiful-log/
│   ├── claw-shell/
│   ├── conda-env-manager/
│   ├── conn-tool/
│   ├── dcm2nii/
│   ├── dependency-planner/
│   ├── dipy-tool/
│   ├── docker-env-manager/
│   ├── dti-skill/
│   ├── eeg-skill/
│   ├── experiment-controller/
│   ├── fmri-skill/
│   ├── fmriprep-tool/
│   ├── freesurfer-tool/
│   ├── fsl-tool/
│   ├── git-essentials/
│   ├── git-workflows/
│   ├── hcp-skill/
│   ├── hcppipeline-tool/
│   ├── method-design/
│   ├── mne-eeg-tool/
│   ├── multi-search-engine/
│   ├── nii2dcm/
│   ├── nilearn-tool/
│   ├── overleaf-skill/
│   ├── paper-writing/
│   ├── qsiprep-tool/
│   ├── research-idea/
│   ├── run_models/
│   ├── skill-updater/
│   ├── smri-skill/
│   └── wmh-segmentation/
│
├── neuro_bench/                    # NeuroBench evaluation tasks (T00–T100)
│   ├── T00_installer_validation/   # Validates installer output
│   └── …
│
├── materials/                      # Research materials and reference resources
│   ├── CVPR_2026/
│   └── examples/
│
└── LICENSE                         # License

```

---

<a id="skill-quick-reference"></a>
## 🛠️ Skill Quick Reference

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
| `harness-core` | Harness engineering SDK (verification, checkpointing, audit logging, drift detection) | ✅ |

### Interface Layer (Task Orchestration)
| Skill | Function | Status |
|------|----------|--------|
| `research-idea` | Brainstorms and generates research ideas from literature | ✅ |
| `method-design` | Formalizes network architecture and derives theoretical components | ✅ |
| `experiment-controller` | Finds and executes reproducible research experiments | ✅ |
| `paper-writing` | Generates hierarchical manuscript drafts from IDEA/METHOD/EXPERIMENT | ✅ |
| `run_models` | Model registry and model execution orchestration | ✅ |

### Subagent Layer
Subagent in NeuroClaw includes four categories: **tool**, **model**, **dataset**, and **modality**.

#### Tool
| Skill | Function | Status |
|------|----------|--------|
| `mne-eeg-tool` | Base-layer MNE-Python implementation for EEG | ✅ |
| `fsl-tool` | FSL-based sMRI/fMRI/DWI processing utilities | ✅ |
| `fmriprep-tool` | fMRIPrep pipeline wrapper and execution | ✅ |
| `qsiprep-tool` | qsiPrep pipeline wrapper for diffusion MRI | ✅ |
| `hcppipeline-tool` | HCP-style processing pipeline utilities | ✅ |
| `dipy-tool` | Diffusion MRI processing via DIPY | ✅ |
| `nilearn-tool` | Fast neuroimaging feature extraction and decoding prep | ✅ |
| `conn-tool` | Functional connectivity computation and analysis | ✅ |
| `freesurfer-tool` | FreeSurfer-based MRI processing and segmentation | ✅ |

#### Model
| Skill | Function | Status |
|------|----------|--------|
| `wmh-segmentation` | White matter hyperintensity segmentation (MARS-WMH nnU-Net) | ✅ |
| `brain_gnn` | BrainGNN: graph neural network for fMRI classification | ✅ |
| `fm_app` | FM-APP: multi-stage phenotype prediction with fMRI+sMRI | ✅ |
| `neurostorm` | NeuroStorm: neuroimaging foundation model | ✅ |

#### Dataset
| Skill | Function | Status |
|------|----------|--------|
| `adni-skill` | ADNI dataset automated processing workflow | ✅ |
| `hcp-skill` | HCP-YA dataset automated processing workflow | ✅ |
| `ukb-skill` | UKB brain imaging automated processing workflow | ⏳ |

#### Modality
| Skill | Function | Status |
|------|----------|--------|
| `eeg-skill` | EEG preprocessing and feature extraction workflows | ✅ |
| `fmri-skill` | Functional MRI preprocessing and analysis workflows | ✅ |
| `smri-skill` | Structural MRI preprocessing and analysis workflows | ✅ |
| `dti-skill` | Diffusion MRI preprocessing and analysis workflows | ✅ |

**Legend**: ✅ Implemented | 🏗️ In Development | ⏳ Planned


---

## TODO List

### Architecture & Foundation
- ✓ Hierarchical architecture design (Interface-Subagent-Base Tool)
- ✓ Complete Interface layer implementation
- ✓ Subagent coordination mechanisms

### Dataset Ecosystem
- ✓ Complete ADNI processing chain
- ✓ HCP dataset adaptation
- ☐ UK Biobank adaptation
- ☐ Multi-dataset workflow support

### Model Reproduction & Execution
- ✓ Automatic paper model retrieval
- ✓ Automatic environment configuration
- ✓ Full Harness Engineering for Reproducibility

### Community & Extensions
- ☐ Multi-institution collaboration capabilities
- ☐ Plugin ecosystem for third-party skills

---


<a id="acknowledgments"></a>
## 🙏 Acknowledgments

Thanks to:
- [OpenClaw](https://github.com/openclaw/openclaw) framework contributors
- All contributors and user feedback
- Open-source neuroscience tools community (MNE-Python, FreeSurfer, FSL, etc.)
