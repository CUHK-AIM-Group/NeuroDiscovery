"""AutoResearch component scopes and ``/help`` guidance for NeuroClaw clients."""

from __future__ import annotations

from dataclasses import dataclass
import re


AUTORESEARCH_MODE_OFF = "off"
AUTORESEARCH_MODE_DATA = "data"
AUTORESEARCH_MODE_MODEL = "model"
AUTORESEARCH_MODE_IDEA = "idea"
AUTORESEARCH_MODE_END_TO_END = "end-to-end"

SUPPORTED_AUTORESEARCH_MODES = (
    AUTORESEARCH_MODE_DATA,
    AUTORESEARCH_MODE_MODEL,
    AUTORESEARCH_MODE_IDEA,
    AUTORESEARCH_MODE_END_TO_END,
)


@dataclass(frozen=True)
class AutoResearchHelpRequest:
    """A parsed ``/help`` request and its optional component scope."""

    mode: str | None
    description: str
    language: str


_MODE_ALIASES = {
    AUTORESEARCH_MODE_DATA: {
        "data", "dataset", "preprocess", "preprocessing", "process data",
        "数据", "处理数据", "数据处理", "预处理", "只处理数据",
    },
    AUTORESEARCH_MODE_MODEL: {
        "model", "models", "train", "training", "run model", "model only",
        "模型", "建模", "训练", "运行模型", "编写模型", "只编写模型",
    },
    AUTORESEARCH_MODE_IDEA: {
        "idea", "ideas", "hypothesis", "brainstorm", "idea only",
        "想法", "研究想法", "假设", "生成idea", "生成 idea", "只生成idea", "只生成 idea",
    },
    AUTORESEARCH_MODE_END_TO_END: {
        "autoresearch", "end-to-end", "end to end", "full", "full pipeline",
        "端到端", "全流程", "完整流程", "完整autoresearch", "完整 autoresearch",
    },
}


def normalize_autoresearch_mode(value: object) -> str:
    """Return a supported mode name, defaulting to ``off``."""
    raw = str(value or "").strip().lower().replace("_", "-")
    legacy = {"on": AUTORESEARCH_MODE_END_TO_END, "high": AUTORESEARCH_MODE_END_TO_END}
    raw = legacy.get(raw, raw)
    return raw if raw in SUPPORTED_AUTORESEARCH_MODES else AUTORESEARCH_MODE_OFF


def _infer_mode(description: str) -> str | None:
    normalized = re.sub(r"\s+", " ", description.strip().lower())
    if not normalized:
        return None

    first_token = normalized.split(" ", 1)[0].strip(":：,，。")
    explicit_tokens = {
        "data": AUTORESEARCH_MODE_DATA,
        "model": AUTORESEARCH_MODE_MODEL,
        "idea": AUTORESEARCH_MODE_IDEA,
        "autoresearch": AUTORESEARCH_MODE_END_TO_END,
        "end-to-end": AUTORESEARCH_MODE_END_TO_END,
    }
    if first_token in explicit_tokens:
        return explicit_tokens[first_token]

    strong_patterns = (
        (AUTORESEARCH_MODE_END_TO_END, r"(?:端到端|全流程|完整流程|end[- ]to[- ]end|full pipeline)"),
        (AUTORESEARCH_MODE_MODEL, r"(?:只(?:想|需要|要)?\s*(?:编写|运行|训练).{0,8}模型|model only|only\s+(?:write|run|train).{0,8}model)"),
        (AUTORESEARCH_MODE_IDEA, r"(?:只(?:想|需要|要)?\s*(?:生成|提出)?\s*(?:研究\s*)?(?:idea|想法|假设)|idea only|only\s+(?:generate|create)?.{0,6}idea)"),
        (AUTORESEARCH_MODE_DATA, r"(?:只(?:想|需要|要)?\s*(?:处理|整理|预处理)\s*(?:数据)?|data only|only\s+(?:process|prepare).{0,6}data)"),
    )
    for mode, pattern in strong_patterns:
        if re.search(pattern, normalized):
            return mode

    # Prefer the complete workflow when the user explicitly asks for it, then
    # the narrower scopes. This prevents "full model + data workflow" from
    # accidentally becoming model-only.
    order = (
        AUTORESEARCH_MODE_END_TO_END,
        AUTORESEARCH_MODE_DATA,
        AUTORESEARCH_MODE_MODEL,
        AUTORESEARCH_MODE_IDEA,
    )
    for mode in order:
        if any(alias in normalized for alias in _MODE_ALIASES[mode]):
            return mode
    return None


def parse_help_command(text: object, language: object = None) -> AutoResearchHelpRequest | None:
    """Parse a Web-client ``/help`` command without treating normal text as one."""
    raw = str(text or "").strip()
    match = re.match(r"^/help(?:\s+|$)(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    description = match.group(1).strip()
    language_hint = str(language or "").strip().lower()
    if language_hint.startswith(("zh", "chinese", "中文", "simplified chinese", "traditional chinese")):
        response_language = "zh"
    elif language_hint.startswith(("en", "english")):
        response_language = "en"
    else:
        response_language = "zh" if re.search(r"[\u3400-\u9fff]", description) else "en"
    return AutoResearchHelpRequest(
        mode=_infer_mode(description),
        description=description,
        language=response_language,
    )


_MODE_CONTENT = {
    AUTORESEARCH_MODE_DATA: {
        "zh_title": "只处理数据",
        "en_title": "Data processing only",
        "zh_scope": "仅完成数据检查、整理、预处理、特征提取与质量控制；止于可供建模的数据产物，不生成 idea，也不编写或运行模型。",
        "en_scope": "Inspect, organize, preprocess, extract features, and run QC only. Stop at model-ready data; do not generate an idea or write/run a model.",
        "zh_items": (
            "研究目标，以及数据处理完成后希望得到的具体产物",
            "原始数据或 BIDS 数据集路径、模态、格式、受试者范围和大致规模",
            "标签/表型/协变量文件路径，以及受试者 ID 的对应方式",
            "已经完成的预处理步骤、希望使用或禁止使用的工具/流程",
            "输出目录、目标空间/atlas/特征格式与质量控制标准",
            "可用的 CPU、GPU、内存、存储空间和时间限制",
        ),
        "en_items": (
            "Research objective and the exact data artifact you want at the end",
            "Raw or BIDS dataset path, modality, format, subject scope, and approximate size",
            "Label/phenotype/covariate paths and the subject-ID mapping rule",
            "Completed preprocessing and required or prohibited tools/pipelines",
            "Output directory, target space/atlas/feature format, and QC criteria",
            "Available CPU, GPU, memory, storage, and time limits",
        ),
    },
    AUTORESEARCH_MODE_MODEL: {
        "zh_title": "只编写并运行模型",
        "en_title": "Model development and execution only",
        "zh_scope": "从已准备好的数据开始，完成模型选择/实现、训练、验证与结果记录；不重新处理原始数据，也不生成研究 idea。",
        "en_scope": "Start from model-ready data and perform model selection/implementation, training, validation, and result logging. Do not preprocess raw data or generate a research idea.",
        "zh_items": (
            "已准备数据的路径、文件格式、张量/表格形状和一个最小样例",
            "预测或建模任务、标签列、类别/回归目标和主要评价指标",
            "训练/验证/测试划分或受试者列表，以及防止数据泄漏的分组规则",
            "指定模型/基线，或允许 NeuroClaw 选择的模型范围",
            "现有代码、METHOD.md、配置、预训练权重和需要复现的仓库（如有）",
            "输出目录、随机种子、运行轮数/预算、CPU/GPU/内存与最长运行时间",
        ),
        "en_items": (
            "Model-ready data path, file format, tensor/table shape, and one minimal sample",
            "Prediction task, label column, class/regression target, and primary metrics",
            "Train/validation/test split or subject lists, including leakage-prevention grouping",
            "Required model/baselines or the model family NeuroClaw may choose from",
            "Existing code, METHOD.md, configs, pretrained weights, or repository to reproduce",
            "Output directory, random seeds, run budget, CPU/GPU/memory, and maximum runtime",
        ),
    },
    AUTORESEARCH_MODE_IDEA: {
        "zh_title": "只生成研究 idea",
        "en_title": "Research idea only",
        "zh_scope": "完成文献检索、研究空白分析和 idea 迭代，输出 IDEA.md；不处理数据、不设计/编写模型，也不运行实验。",
        "en_scope": "Search literature, analyze gaps, and refine an idea into IDEA.md. Do not process data, design/write a model, or run experiments.",
        "zh_items": (
            "研究领域、疾病/人群、成像模态和你关心的科学问题",
            "已有观察、初步假设或希望避开的方向（没有也可以）",
            "可获得的数据集、样本量或现实资源边界，用于判断可行性",
            "希望强调的新颖性类型：机制、方法、数据、临床转化或其他",
            "文献时间范围、必须纳入的论文/作者，以及允许使用的检索来源",
            "期望的 IDEA.md 深度、候选 idea 数量和筛选标准",
        ),
        "en_items": (
            "Research area, disease/population, imaging modality, and scientific question",
            "Existing observations, tentative hypotheses, or directions to avoid (optional)",
            "Available datasets, sample size, or practical resource limits for feasibility",
            "Preferred novelty: mechanism, method, data, clinical translation, or another type",
            "Literature date range, must-include papers/authors, and allowed search sources",
            "Desired IDEA.md depth, number of candidate ideas, and selection criteria",
        ),
    },
    AUTORESEARCH_MODE_END_TO_END: {
        "zh_title": "端到端 autoresearch",
        "en_title": "End-to-end autoresearch",
        "zh_scope": "依次覆盖 idea、数据、方法/模型、实验与验证；每一阶段保留可复现产物，并在长任务或高成本执行前确认方案。",
        "en_scope": "Cover idea, data, method/model, experiment, and validation in sequence. Preserve reproducible artifacts and confirm plans before long or costly execution.",
        "zh_items": (
            "总体科学问题、目标结论、预期输出和成功标准",
            "数据集路径/访问方式、模态、样本范围、标签/协变量和使用限制",
            "已有的 IDEA.md、METHOD.md、代码、论文、模型或中间结果（如有）",
            "允许检索的文献范围，以及对新颖性、可解释性和可复现性的要求",
            "数据处理、模型选择、基线、评价指标和统计检验方面的偏好",
            "输出根目录、阶段检查点、实验轮数/预算与停止条件",
            "CPU/GPU/内存/存储、软件许可、API key 和最长运行时间",
        ),
        "en_items": (
            "Overall scientific question, target conclusion, expected outputs, and success criteria",
            "Dataset path/access, modality, cohort scope, labels/covariates, and usage constraints",
            "Existing IDEA.md, METHOD.md, code, papers, models, or intermediate results (if any)",
            "Allowed literature scope and novelty, interpretability, and reproducibility requirements",
            "Preferences for preprocessing, model selection, baselines, metrics, and statistics",
            "Output root, stage checkpoints, experiment budget, and stopping conditions",
            "CPU/GPU/memory/storage, software licenses, API keys, and maximum runtime",
        ),
    },
}


def render_help_response(request: AutoResearchHelpRequest) -> str:
    """Render a concise, actionable checklist for a parsed help request."""
    if request.mode is None:
        if request.language == "zh":
            return (
                "## 选择 autoresearch 工作模式\n\n"
                "请在 `/help` 后描述你想做的工作。目前支持四种模式：\n\n"
                "- `/help data …`：只处理数据\n"
                "- `/help model …`：只编写并运行模型\n"
                "- `/help idea …`：只生成研究 idea\n"
                "- `/help autoresearch …`：端到端 autoresearch\n\n"
                "例如：`/help model 我有处理好的 ROI 特征，想训练分类模型`。"
            )
        return (
            "## Choose an autoresearch scope\n\n"
            "Describe what you want after `/help`. Four modes are supported:\n\n"
            "- `/help data …`: process data only\n"
            "- `/help model …`: write and run models only\n"
            "- `/help idea …`: generate a research idea only\n"
            "- `/help autoresearch …`: run end-to-end autoresearch\n\n"
            "Example: `/help model I have model-ready ROI features for classification`."
        )

    content = _MODE_CONTENT[request.mode]
    zh = request.language == "zh"
    title = content["zh_title" if zh else "en_title"]
    scope = content["zh_scope" if zh else "en_scope"]
    items = content["zh_items" if zh else "en_items"]
    heading = "请提供以下资料" if zh else "Please provide"
    next_step = (
        "资料不完整也可以先发；NeuroClaw 会标出缺失项，并在你确认范围后再执行。"
        if zh else
        "You can start with incomplete information; NeuroClaw will identify gaps and wait for scope confirmation before execution."
    )
    checklist = "\n".join(f"- [ ] {item}" for item in items)
    return f"## {title}\n\n**范围：** {scope}\n\n### {heading}\n\n{checklist}\n\n{next_step}"


def build_autoresearch_scope_prompt(mode: object) -> str:
    """Build hidden execution constraints for the selected component scope."""
    normalized = normalize_autoresearch_mode(mode)
    if normalized == AUTORESEARCH_MODE_OFF:
        return ""

    content = _MODE_CONTENT[normalized]
    required = "\n".join(f"- {item}" for item in content["en_items"])
    return (
        f"[NeuroClaw AutoResearch component scope: {normalized}]\n"
        f"Scope boundary: {content['en_scope']}\n"
        "Before substantive work, check whether the following inputs are available:\n"
        f"{required}\n"
        "If essential inputs are missing, ask only for those missing inputs. "
        "State the active component scope in the plan and do not execute components outside it. "
        "Require explicit confirmation before dependency installation, long-running jobs, or costly experiments."
    )
