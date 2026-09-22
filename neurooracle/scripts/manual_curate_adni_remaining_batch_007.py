from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_adni_relevance_batch as annotate


ROOT = Path(__file__).resolve().parents[2]
BATCH_ID = "batch18_manual_remaining_007"
OUTPUT = (
    ROOT
    / "neurooracle/data/user_study/"
    "adni_manual_relevance_annotations_v1_batch_18_manual_remaining_007.json"
)

EXPECTED_CATEGORIES = {
    "adni-ext-82df84f72489b9": "frontal_pole",
    "adni-ext-b0a7587d634dd3": "dmn_pcc",
    "adni-ext-b187a635667ca7": "dmn_temporal",
    "adni-ext-b2891c00ed0f51": "ifg_triangular",
    "adni-ext-b29b5f52ff7c12": "dmn_parietal",
    "adni-ext-b2ac7f0760c4a1": "dmn_temporal",
    "adni-ext-b334aad044f612": "dmn_prefrontal",
    "adni-ext-b647321881062c": "ifg_triangular",
    "adni-ext-b64803f11f9032": "insula",
    "adni-ext-b832deccf2e36a": "dmn_pcc",
    "adni-ext-b9a4b2bb8fcd36": "frontal_opercular",
    "adni-ext-bbc03f97a18748": "frontal_pole",
    "adni-ext-bf5481d56a2aec": "dmn_prefrontal",
    "adni-ext-c2552e9429d93f": "dmn_pcc",
    "adni-ext-c623a561a53cd0": "dmn_temporal",
    "adni-ext-c810d0192ae2c6": "frontal_opercular",
}


PAPER_STUDIES: dict[str, tuple[str, str]] = {
    "41836094": (
        "该研究将帕金森病患者按认知状态分为认知正常、轻度认知障碍和痴呆亚组，并与健康对照进行静息态 fMRI 比较。全体帕金森病患者右尾状核和左岛叶 ALFF 升高、左枕中回 ALFF 降低；PDMCI 相对 PDCN 的左额下回三角部和角回 ALFF 升高。研究还报告了岛叶—中扣带回、岛叶—颞上回等功能连接改变。",
        "This resting-state fMRI study stratified Parkinson disease into cognitively normal, mild-cognitive-impairment, and dementia subgroups and compared them with healthy controls. Across Parkinson disease, ALFF increased in right caudate and left insula and decreased in left middle occipital gyrus; PDMCI versus PDCN showed increased ALFF in left triangular inferior frontal gyrus and angular gyrus. Insula–mid-cingulate and insula–superior-temporal connectivity was also altered.",
    ),
    "37674874": (
        "该研究比较遗忘型轻度认知障碍与对照的静息态 fMRI，在颞叶、额叶、小脑、海马旁回、尾状核和丘脑报告异常 ALFF 与 ReHo，部分指标与 MoCA 或 MMSE 相关。",
        "This resting-state fMRI study compared amnestic mild cognitive impairment with controls and reported abnormal ALFF and ReHo in temporal, frontal, cerebellar, parahippocampal, caudate, and thalamic regions; several measures correlated with MoCA or MMSE.",
    ),
    "36340771": (
        "该研究在伴或不伴轻度认知障碍的阻塞性睡眠呼吸暂停患者中分析动态 ALFF 和小脑—前额叶通路。无 MCI 的 OSA 患者后部小脑 dALFF 升高；小脑—前额叶通路差异以及 dALFF 与呼吸暂停严重度和嗜睡的相关性区分了 OSA 认知表型。",
        "This study analyzed dynamic ALFF and cerebellar–prefrontal pathways in obstructive sleep apnea with and without mild cognitive impairment. Posterior-cerebellar dALFF was increased in OSA without MCI, while cerebellar–prefrontal pathway differences and dALFF correlations with apnea severity and sleepiness distinguished OSA cognitive phenotypes.",
    ),
    "35557608": (
        "该 ALE 元分析汇总轻度认知障碍的静息态功能研究，在背侧注意网络相关的颞叶、额叶、中央前回、顶叶和顶下小叶发现收敛异常；纳入证据混合了 ALFF、ReHo 和功能连接。",
        "This ALE meta-analysis synthesized resting-state functional studies of mild cognitive impairment and found convergent abnormalities in dorsal-attention-network-related temporal, frontal, precentral, parietal, and inferior-parietal regions; the evidence mixed ALFF, ReHo, and functional connectivity.",
    ),
    "35663572": (
        "该研究联合 VBM、ALFF、ReHo 和静息态功能连接评估遗忘型轻度认知障碍，报告额叶存在并行的结构与功能降低。",
        "This study combined VBM, ALFF, ReHo, and resting-state functional connectivity in amnestic mild cognitive impairment and reported concurrent structural and functional reductions in frontal regions.",
    ),
    "34861133": (
        "该研究在遗忘型轻度认知障碍中联合静息态 fMRI 与 AV45 PET，发现默认网络和视觉网络区域较低的标准化 ALFF 与较高的淀粉样蛋白 SUVR 相关；心血管信号回归会削弱部分相关。",
        "This study combined resting-state fMRI with AV45 PET in amnestic mild cognitive impairment and found that lower standardized ALFF in default-mode and visual-network regions correlated with higher amyloid SUVR; cardiovascular-signal regression attenuated some correlations.",
    ),
    "36003964": (
        "该研究从海马结构 MRI 和海马 ALFF 图像提取影像组学纹理；结构与 ALFF 联合模型最能区分 AD、aMCI 与对照，ALFF 纹理提高了早期 aMCI 的判别能力。",
        "This radiomics study extracted texture features from hippocampal structural MRI and hippocampal ALFF images. Combined structural-plus-ALFF models best distinguished AD, aMCI, and controls, and ALFF texture improved early-aMCI discrimination.",
    ),
    "34675797": (
        "该 ALE 元分析整合 ALFF/fALFF、ReHo 和功能连接研究，发现轻度认知障碍的默认模式网络存在跨研究收敛的功能异常。",
        "This ALE meta-analysis integrated ALFF/fALFF, ReHo, and functional-connectivity studies and identified cross-study convergent functional abnormalities of the default mode network in mild cognitive impairment.",
    ),
    "33453729": (
        "该研究使用动态分数 ALFF 与动态功能连接，所得时变特征可区分主观认知下降、轻度认知障碍和对照，并与 MMSE 相关。",
        "This study used dynamic fractional ALFF and dynamic functional connectivity; the time-varying features differentiated subjective cognitive decline, mild cognitive impairment, and controls and correlated with MMSE.",
    ),
    "23924467": (
        "该静息态 fMRI 研究直接比较遗忘型轻度认知障碍与对照，发现海马/海马旁回、外侧颞叶和腹内侧前额叶 ALFF 降低，而颞顶交界区和顶下小叶 ALFF 升高。",
        "This resting-state fMRI study directly compared amnestic mild cognitive impairment with controls and found decreased ALFF in hippocampal/parahippocampal, lateral-temporal, and ventromedial-prefrontal regions and increased ALFF in temporoparietal junction and inferior parietal lobule.",
    ),
    "40118165": (
        "该跨诊断结构 MRI 研究纳入 MCI、AD 和额颞叶痴呆，使用 FreeSurfer 与病灶—症状映射把认知表现与区域体积联系起来：记忆涉及海马和伏隔核，言语流畅性涉及左颞上沟/颞中回，顺背数字广度涉及左额上回和左顶下区，倒背数字广度涉及右楔前叶。",
        "This transdiagnostic structural-MRI study included MCI, AD, and frontotemporal dementia and used FreeSurfer lesion–symptom mapping to relate cognition to regional volume: memory involved hippocampus and nucleus accumbens, verbal fluency left superior-temporal sulcus/middle-temporal gyrus, forward digit span left superior-frontal and inferior-parietal regions, and backward digit span right precuneus.",
    ),
    "40167963": (
        "该纵向 PET/T1 MRI 研究分析认知正常和 MCI 的高淀粉样蛋白累积者。Aβ 主要累积于内侧眶额、扣带和楔前叶；灰质萎缩位于颞叶、枕叶、眶额和顶叶。APOE ε4 与记忆下降共同关联更快的病理进展，但 MCI 诊断本身未显示交互效应。",
        "This longitudinal PET/T1-MRI study analyzed cognitively normal and MCI participants with high amyloid accumulation. Amyloid predominated in medial orbitofrontal, cingulate, and precuneus cortex, with gray-matter atrophy in temporal, occipital, orbitofrontal, and parietal areas. APOE ε4 plus memory decline marked faster progression, whereas MCI diagnosis itself showed no interaction.",
    ),
    "40415136": (
        "该五年纵向队列比较前驱 DLB、前驱 AD、DLB+AD 与无相关症状组的认知、MRI 体积和 FDG-PET。前驱 AD 与 DLB+AD 的海马、内嗅、杏仁核和左岛叶体积下降更明显；DLB+AD 左眶额代谢下降更大。",
        "This five-year cohort compared prodromal DLB, prodromal AD, combined DLB+AD, and symptom-negative groups using cognition, MRI volume, and FDG-PET. Prodromal AD and DLB+AD showed greater hippocampal, entorhinal, amygdalar, and left-insular volume loss; DLB+AD also showed greater left-orbitofrontal metabolic decline.",
    ),
    "39920001": (
        "该研究以结构 MRI 深度学习预测认知正常或主观认知下降者向 MCI 进展。模型识别的关键区域包括海马、杏仁核、颞叶、岛叶和前部小脑，组合进展指数可预测转换。",
        "This structural-MRI deep-learning study predicted conversion from normal cognition or subjective cognitive decline to MCI. Key regions included hippocampus, amygdala, temporal lobe, insula, and anterior cerebellum, and the composite progression index predicted conversion.",
    ),
    "38107635": (
        "该结构 MRI 研究比较 SCD、MCI 和健康对照的皮层表面积、厚度、局部脑回指数及皮层下体积。MCI 的改变偏左侧，涉及颞横回、颞上回、岛叶和额下回盖部；这些形态改变与认知下降评分相关。",
        "This structural-MRI study compared cortical surface area, thickness, local gyrification, and subcortical volume across SCD, MCI, and healthy controls. MCI changes were predominantly left-sided and involved transverse-temporal gyrus, superior-temporal gyrus, insula, and pars opercularis; morphometry correlated with cognitive-decline ratings.",
    ),
    "34560530": (
        "该 VBM 元分析汇总 48 项 MDD、MCI 和对照研究，发现 MDD 与 MCI 共享岛叶、颞上回、额下回、杏仁核、海马和丘脑的体积减少。",
        "This VBM meta-analysis synthesized 48 MDD, MCI, and control studies and found shared volumetric reductions in insula, superior-temporal gyrus, inferior-frontal gyrus, amygdala, hippocampus, and thalamus in MDD and MCI.",
    ),
}


def rel(
    support_zh: str,
    support_en: str,
    strength_zh: str,
    strength_en: str,
    boundary_zh: str,
    boundary_en: str,
) -> dict[str, str]:
    return {
        "support_zh": support_zh,
        "support_en": support_en,
        "strength_zh": strength_zh,
        "strength_en": strength_en,
        "boundary_zh": boundary_zh,
        "boundary_en": boundary_en,
    }


def relation(category: str, paper_id: str, region: str) -> dict[str, str]:
    if paper_id == "41836094":
        if category == "insula":
            return rel(
                "全体帕金森病患者相对健康对照左岛叶 ALFF 升高，直接表明岛叶局部低频振幅可随疾病状态改变；岛叶 ALFF 还中介认知损害。目标区域同为岛叶，因此脑区和指标均直接对应。",
                "Left-insular ALFF was increased in Parkinson disease versus healthy controls and mediated cognitive impairment, directly showing disease-related alteration of local low-frequency amplitude in the same anatomical region and metric targeted by this hypothesis.",
                "中度间接支持",
                "moderate indirect support",
                "MCI 亚组比较报告的是额下回三角部和角回，而不是岛叶；岛叶结果来自全体帕金森病与健康对照，不能证明通常的 MCI 相对认知正常对照在岛叶有相同改变。",
                "The MCI-subgroup contrast concerned triangular IFG and angular gyrus, not insula. The insular result was Parkinson disease versus controls and therefore does not establish the same change in general MCI versus cognitively normal controls.",
            )
        if category == "ifg_triangular":
            return rel(
                "PDMCI 相对 PDCN 的左额下回三角部 ALFF 升高，与目标 {region} 属于同一额下回三角部结构，并使用相同的 ALFF 指标；这是本批文献中对目标解剖位置最接近的功能证据。",
                "PDMCI versus PDCN showed increased ALFF in left triangular inferior frontal gyrus, matching the same triangular-IFG structure and ALFF metric as target {region}; this is the closest functional-anatomical evidence in the supplied literature.",
                "中度间接支持",
                "moderate indirect support",
                "论文结果为左侧，而目标分区为 {region}；研究对象是帕金森病相关 MCI，且比较组为 PDCN，不是通常的 MCI 与健康认知正常对照。",
                "The reported effect was left-sided whereas the target is {region}; the population was Parkinson-associated MCI and the comparator PDCN rather than general MCI versus healthy cognitively normal controls.",
            )
        if category == "frontal_opercular":
            return rel(
                "PDMCI 相对 PDCN 的左额下回三角部 ALFF 升高，说明紧邻额下回盖部/额盖的额下回亚区可出现认知状态相关 ALFF 改变。三角部与目标 {region} 同属外侧额下回，但并非同一细分区。",
                "Increased ALFF in left triangular IFG for PDMCI versus PDCN shows cognition-related ALFF alteration in an inferior-frontal subdivision adjacent to opercular IFG/frontal operculum. It is anatomically related to target {region} but not the same subdivision.",
                "低至中度间接支持",
                "weak-to-moderate indirect support",
                "未报告目标 {region}，且半球、MCI 病因和对照组均不完全匹配；不能把三角部升高直接外推到盖部。",
                "Target {region} was not reported, and hemisphere, MCI etiology, and comparator differ; an increase in pars triangularis cannot be directly extrapolated to pars opercularis.",
            )
        if category == "frontal_pole":
            return rel(
                "PDMCI 相对 PDCN 的左额下回三角部 ALFF 升高，证明额叶注意网络的局部低频振幅可随认知状态改变；这为目标额极 {region} 提供额叶系统层面的邻近证据。",
                "Increased left triangular-IFG ALFF in PDMCI versus PDCN demonstrates cognition-related low-frequency-amplitude alteration within frontal attention circuitry, providing neighboring frontal-system evidence for target frontal-pole region {region}.",
                "低度间接支持",
                "weak indirect support",
                "额下回三角部位于腹外侧额叶，并非更前端的额极；研究也没有报告 {region} 的数值，且 MCI 为帕金森病相关。",
                "Triangular IFG is ventrolateral frontal cortex rather than the more anterior frontal pole; no value was reported for {region}, and the MCI was Parkinson-associated.",
            )
        if category == "dmn_parietal":
            return rel(
                "PDMCI 相对 PDCN 的角回 ALFF 升高。角回是顶下小叶的重要组成，并常属于默认网络顶叶节点，因此与目标 {region} 在脑区和网络归属上接近，且指标同为 ALFF。",
                "PDMCI versus PDCN showed increased angular-gyrus ALFF. Angular gyrus is part of inferior parietal cortex and commonly a default-mode parietal node, making it anatomically and network-related to target {region} with the same ALFF metric.",
                "中度间接支持",
                "moderate indirect support",
                "未给出 Schaefer 目标分区的映射，且 PDMCI 与 PDCN 的比较不能等同于一般 MCI 与健康对照；目标假设也未预设升高方向。",
                "The paper did not map the result to the target Schaefer parcel, and PDMCI versus PDCN is not general MCI versus healthy controls; the hypothesis also does not prespecify an increase.",
            )
        if category == "dmn_temporal":
            return rel(
                "论文发现左岛叶—右颞上回连接下降，说明颞叶网络节点与其他认知网络的耦合可异常；目标 {region} 同属默认网络颞叶，但论文并未报告该节点的局部 ALFF。",
                "Reduced left-insula–right-superior-temporal connectivity shows abnormal coupling of a temporal-network node. Target {region} is also a default-mode temporal parcel, but the paper did not measure local ALFF in that parcel.",
                "低度间接支持",
                "weak indirect support",
                "支持来自功能连接而非 ROI ALFF，颞上回也不能精确对应 {region}；疾病为帕金森病且没有一般 MCI–健康对照比较。",
                "The evidence is functional connectivity rather than ROI ALFF, superior-temporal gyrus is not an exact map to {region}, and the population was Parkinson disease rather than general MCI versus controls.",
            )
        if category == "dmn_pcc":
            return rel(
                "岛叶—中扣带回连接下降表明扣带网络耦合可随帕金森病认知损害改变；目标 {region} 位于楔前叶/后扣带默认网络，因此仅在扣带系统层面相关。",
                "Reduced insula–mid-cingulate connectivity shows that cingulate-network coupling can change with cognitive impairment in Parkinson disease. Target {region} is a precuneus/posterior-cingulate default-mode parcel, so the relation is limited to the broader cingulate system.",
                "低度间接支持",
                "weak indirect support",
                "中扣带回不是后扣带/楔前叶，测量是连接而不是目标区域 ALFF；论文未检验一般 MCI 与认知正常对照的 {region}。",
                "Mid-cingulate is not posterior cingulate/precuneus, the metric was connectivity rather than target-region ALFF, and the paper did not test {region} in general MCI versus cognitively normal controls.",
            )
        return rel(
            "PDMCI 的额下回三角部和角回 ALFF 改变说明额叶与顶叶认知网络可出现 MCI 状态相关局部振幅异常，为默认网络前额叶目标 {region} 提供跨网络的额叶背景。",
            "ALFF changes in triangular IFG and angular gyrus in PDMCI show cognition-related local-amplitude abnormalities in frontal and parietal networks, providing cross-network frontal context for default-mode prefrontal target {region}.",
            "低度间接支持",
            "weak indirect support",
            "目标 {region} 是背侧/内侧默认网络前额叶分区，不等于额下回三角部；疾病背景和比较组也不匹配。",
            "Target {region} is a dorsal/medial default-mode prefrontal parcel, not triangular IFG; disease context and comparator also differ.",
        )

    if category in {"frontal_pole", "ifg_triangular", "frontal_opercular"}:
        if paper_id == "37674874":
            return rel(
                "aMCI 相对对照在额叶报告异常 ALFF/ReHo，直接支持轻度认知障碍可伴随额叶静息态局部活动异常；目标 {region} 属于额叶，ALFF 指标也在论文范围内。",
                "Abnormal frontal ALFF/ReHo in aMCI versus controls directly supports frontal resting-state local-activity alteration in mild cognitive impairment; target {region} is frontal and ALFF was among the measured metrics.",
                "中度间接支持",
                "moderate indirect support",
                "现有摘要没有列出具体额叶亚区、半球和方向，无法确认异常是否落在 {region}，也不能区分 ALFF 与 ReHo 各自贡献。",
                "The available evidence does not specify frontal subdivision, hemisphere, or direction, so localization to {region} and the separate ALFF contribution cannot be confirmed.",
            )
        if paper_id == "36340771":
            return rel(
                "伴与不伴 MCI 的 OSA 表型存在小脑—前额叶通路差异，说明前额叶相关回路可与认知受损状态相关；这为目标 {region} 提供前额叶网络层面的背景。",
                "Cerebellar–prefrontal pathway differences between OSA cognitive phenotypes show that prefrontal circuitry can track cognitive impairment, providing frontal-network context for target {region}.",
                "仅背景相关",
                "background only",
                "局部 dALFF 结果主要位于后部小脑，论文没有报告 {region} 的 ALFF；人群为 OSA，通路差异也不能等同于一般 MCI 的局部振幅改变。",
                "The local dALFF result was mainly posterior cerebellar, with no ALFF reported in {region}; the population was OSA and pathway differences do not equal local-amplitude change in general MCI.",
            )
        if paper_id == "35557608":
            return rel(
                "MCI 的 ALE 元分析在背侧注意网络相关额叶区域发现跨研究收敛功能异常，说明额叶注意系统是可重复受影响的网络；目标 {region} 位于额叶并可能参与相邻控制/注意回路。",
                "The MCI ALE meta-analysis found cross-study convergent abnormalities in frontal regions related to the dorsal attention network, showing reproducible frontal-system involvement relevant to neighboring control/attention functions of target {region}.",
                "中度间接支持",
                "moderate indirect support",
                "元分析混合 ALFF、ReHo 和功能连接，现有材料没有给出与 {region} 对应的坐标或单独 ALFF 方向。",
                "The meta-analysis mixed ALFF, ReHo, and connectivity, and the available evidence gives neither coordinates mapping to {region} nor an ALFF-specific direction.",
            )
        if paper_id == "35663572":
            return rel(
                "aMCI 在额叶出现并行结构与功能降低，且功能分析包含 ALFF，支持额叶局部活动降低可与 aMCI 同时出现；目标 {region} 属于该广义额叶范围。",
                "Concurrent frontal structural and functional reductions in aMCI, with ALFF included in the functional analyses, support frontal local-activity reduction as part of aMCI and place target {region} within the relevant broad lobe.",
                "中度间接支持",
                "moderate indirect support",
                "材料未说明具体额叶分区、左右侧以及哪一项功能指标驱动结果，不能据此断言 {region} 的 ROI ALFF 降低。",
                "The evidence does not specify frontal subdivision, hemisphere, or which functional metric drove the result, so it cannot establish decreased ROI ALFF in {region}.",
            )

    if category.startswith("dmn_"):
        if paper_id == "34861133":
            return rel(
                "aMCI 中默认网络区域较低 sALFF 与更高淀粉样蛋白负荷相关，直接把默认网络低频振幅与 AD 相关病理联系起来；目标 {region} 是默认网络分区，因此网络和指标概念高度相关。",
                "In aMCI, lower sALFF in default-mode regions correlated with higher amyloid burden, directly linking default-mode low-frequency amplitude to AD-related pathology. Target {region} is a default-mode parcel, so network and metric concepts align closely.",
                "中度间接支持",
                "moderate indirect support",
                "这是 sALFF–淀粉样蛋白的组内相关，而非 MCI–对照组间差异；现有材料未列出 {region}，且 sALFF 不完全等同于普通 ROI ALFF。",
                "This was a within-aMCI sALFF–amyloid correlation rather than an MCI–control group difference; {region} was not identified, and sALFF is not identical to ordinary ROI ALFF.",
            )
        if paper_id == "36003964":
            strength = "中度间接支持" if category == "dmn_temporal" else "低度间接支持"
            strength_en = "moderate indirect support" if category == "dmn_temporal" else "weak indirect support"
            return rel(
                "海马 ALFF 纹理提高 aMCI 与对照的区分，证明内侧颞叶局部低频活动图像包含早期认知受损信息。"
                + (
                    "目标 {region} 同属默认网络颞叶，与海马在记忆网络中紧密连接。"
                    if category == "dmn_temporal"
                    else "目标 {region} 与海马同属默认网络但位于不同解剖节点。"
                ),
                "Hippocampal ALFF texture improved discrimination of aMCI from controls, showing that medial-temporal low-frequency-activity images contain early cognitive-impairment information. "
                + (
                    "Target {region} is also a default-mode temporal parcel and is closely connected to hippocampal memory circuitry."
                    if category == "dmn_temporal"
                    else "Target {region} shares default-mode membership with hippocampal circuitry but is a different anatomical node."
                ),
                strength,
                strength_en,
                "论文分析海马纹理而非 {region} 的平均 ROI ALFF；纹理可在平均振幅不变时改变，因此不能推出目标分区的方向或组间效应。",
                "The study analyzed hippocampal texture rather than mean ROI ALFF in {region}; texture can change without a mean-amplitude change, so target direction and group effect remain untested.",
            )
        if paper_id == "34675797":
            return rel(
                "ALE 元分析确认 MCI 的默认网络存在跨研究收敛功能异常，目标 {region} 正是默认网络分区，因此为该节点可能异常提供直接的网络级依据。",
                "The ALE meta-analysis confirmed cross-study convergent default-mode dysfunction in MCI. Because target {region} is a default-mode parcel, it provides direct network-level evidence that this class of node may be abnormal.",
                "中度间接支持",
                "moderate indirect support",
                "证据混合 ALFF/fALFF、ReHo 和连接，现有摘要未给出收敛簇是否覆盖 {region}，也没有该 ROI 的 ALFF 方向。",
                "Evidence mixed ALFF/fALFF, ReHo, and connectivity; the available abstract does not show whether a convergence cluster covers {region} or provide an ROI-ALFF direction.",
            )
        if paper_id == "33453729":
            return rel(
                "动态 dfALFF/dFC 特征可区分 MCI 与对照并与 MMSE 相关，说明低频振幅的时变属性具有认知状态信息；这支持在目标 {region} 进一步检验局部振幅。",
                "Dynamic dfALFF/dFC features differentiated MCI from controls and correlated with MMSE, showing that time-varying low-frequency amplitude carries cognitive-state information and motivating local-amplitude testing in target {region}.",
                "低度间接支持",
                "weak indirect support",
                "现有材料没有报告具体脑区，动态分数 ALFF 也不同于静态 ROI ALFF，因此不能定位到 {region} 或提供方向。",
                "No specific region was reported, and dynamic fractional ALFF differs from static ROI ALFF, so the evidence cannot localize to {region} or supply a direction.",
            )

    if category == "ifg_triangular":
        if paper_id == "23924467":
            return rel(
                "aMCI 相对对照的腹内侧前额叶 ALFF 降低，同时颞顶交界和顶下小叶升高，直接证明 aMCI 可出现方向不一的额叶—顶叶 ALFF 重组；目标 {region} 位于邻近但更外侧的额下回。",
                "aMCI versus controls showed decreased ventromedial-prefrontal ALFF and increased temporoparietal/inferior-parietal ALFF, directly demonstrating bidirectional frontal–parietal ALFF reorganization; target {region} is a neighboring but more lateral inferior-frontal region.",
                "中度间接支持",
                "moderate indirect support",
                "论文没有报告额下回三角部，腹内侧前额叶与 {region} 不同，因而不能把降低方向外推到目标区域。",
                "Triangular IFG was not reported; ventromedial prefrontal cortex differs from {region}, so the decreased direction cannot be extrapolated to the target.",
            )
        if paper_id == "40118165":
            return rel(
                "MCI/AD/bvFTD 的顺背数字广度与左额上回体积相关，说明额叶结构与工作记忆表现相连；该认知域与额下回控制功能相关，为目标 {region} 提供功能系统背景。",
                "Forward digit span in MCI/AD/bvFTD was associated with left superior-frontal volume, linking frontal structure to working-memory performance. That cognitive domain relates to inferior-frontal control functions and provides system-level context for target {region}.",
                "仅背景相关",
                "background only",
                "研究的是体积—认知相关，脑区为额上回而非 {region}，没有 ALFF 或 MCI–对照组间结果。",
                "The study examined volume–cognition associations in superior-frontal gyrus rather than {region}, with no ALFF or MCI–control group comparison.",
            )
        if paper_id == "40167963":
            return rel(
                "高淀粉样蛋白累积者的内侧眶额皮层出现 Aβ 病理，灰质萎缩也涉及眶额区，说明前额叶在前驱 AD 病理中受累；目标 {region} 属于另一外侧额下回分区。",
                "High-amyloid accumulators showed medial-orbitofrontal amyloid pathology and orbitofrontal gray-matter atrophy, demonstrating prefrontal involvement in prodromal AD pathology; target {region} is a different lateral inferior-frontal subdivision.",
                "低度间接支持",
                "weak indirect support",
                "论文未测量 ALFF，MCI 诊断本身没有交互效应，且眶额区不能精确代表 {region}，因此只支持广义前额叶易感性。",
                "ALFF was not measured, MCI diagnosis itself showed no interaction, and orbitofrontal cortex does not map to {region}; the evidence only supports broad prefrontal vulnerability.",
            )

    if category == "insula":
        if paper_id == "40415136":
            return rel(
                "前驱 AD 与 DLB+AD 的左岛叶体积下降更明显，直接显示认知障碍前驱阶段岛叶可受累；目标同为岛叶，因此解剖位置一致。",
                "Greater left-insular volume loss in prodromal AD and DLB+AD directly shows insular involvement during prodromal cognitive disease; the anatomical target is the same insular region.",
                "中度间接支持",
                "moderate indirect support",
                "指标是 MRI 体积而非 ALFF，队列混合 SCI/MCI 与不同病因，也没有一般 MCI–健康对照的岛叶 ALFF 结果。",
                "The metric was MRI volume rather than ALFF, the cohort mixed SCI/MCI and etiologies, and no general MCI–healthy-control insular ALFF result was reported.",
            )
        if paper_id == "39920001":
            return rel(
                "结构 MRI 深度学习将岛叶识别为预测认知正常/SCD 向 MCI 进展的关键 ROI，说明岛叶在临床 MCI 出现之前已经携带进展信息；目标脑区与之直接一致。",
                "Structural-MRI deep learning identified insula as a key ROI predicting conversion from normal cognition/SCD to MCI, showing that the same target region carries progression information before clinical MCI.",
                "中度间接支持",
                "moderate indirect support",
                "模型使用结构图像和组合进展指数，未报告岛叶 ALFF、单区效应方向或 MCI 与对照的直接比较。",
                "The model used structural images and a composite progression index and did not report insular ALFF, a single-region direction, or a direct MCI–control comparison.",
            )
        if paper_id == "38107635":
            return rel(
                "MCI 相对健康对照在岛叶出现显著皮层表面形态改变，并与认知下降评分相关；这是针对同一疾病比较和同一脑区的直接解剖证据。",
                "MCI versus healthy controls showed significant insular cortical-surface morphometric changes associated with cognitive-decline ratings, providing direct disease-comparison evidence in the same anatomical region.",
                "中度间接支持",
                "moderate indirect support",
                "论文测量表面积、厚度或脑回指数而非 ALFF，改变以左侧为主；结构异常不能确定岛叶局部振幅是否或如何改变。",
                "The paper measured surface area, thickness, or gyrification rather than ALFF and changes were predominantly left-sided; structural abnormality cannot determine whether or how insular local amplitude changes.",
            )
        if paper_id == "34560530":
            return rel(
                "48 项 VBM 研究的元分析发现 MCI 与 MDD 均有岛叶体积减少，提供跨研究重复的 MCI 岛叶受累证据；目标脑区完全一致。",
                "A 48-study VBM meta-analysis found insular volume reduction shared by MCI and MDD, providing cross-study reproducible evidence of insular involvement in MCI in the exact target region.",
                "中度间接支持",
                "moderate indirect support",
                "证据是灰质体积而非静息态 ALFF，并且是跨诊断共享模式；不能推断岛叶 ROI ALFF 的方向。",
                "The evidence concerns gray-matter volume rather than resting-state ALFF and is a transdiagnostic shared pattern; it cannot determine the direction of insular ROI ALFF.",
            )

    raise KeyError(f"No manual relation for {category=} {paper_id=}")


def main() -> None:
    bank = annotate._load_json(annotate.DEFAULT_BANK)
    truth = annotate._load_json(annotate.DEFAULT_TRUTH)
    processed = annotate._processed_ids(annotate.DEFAULT_ANNOTATIONS)
    candidates = annotate._select_candidates(bank, truth, processed, 16, "any")
    actual_ids = [str(candidate["id"]) for candidate in candidates]
    if actual_ids != list(EXPECTED_CATEGORIES):
        raise RuntimeError(f"Manual batch selection changed: {actual_ids}")

    paper_lookup = {
        annotate._paper_id(paper): paper
        for candidate in candidates
        for paper in candidate["literature"]
    }
    if set(paper_lookup) != set(PAPER_STUDIES):
        raise RuntimeError("Manual paper set changed")

    paper_profiles: dict[str, dict[str, Any]] = {}
    for paper_id, paper in paper_lookup.items():
        study_zh, study_en = PAPER_STUDIES[paper_id]
        has_abstract = bool(str(paper.get("abstract") or "").strip())
        paper_profiles[paper_id] = {
            "study_zh": study_zh,
            "study_en": study_en,
            "abstract_verified": has_abstract,
            "abstract_source": "embedded_pubmed_abstract" if has_abstract else "embedded_evidence_excerpt",
        }

    assignments: dict[str, str] = {}
    relation_profiles: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        category = EXPECTED_CATEGORIES[candidate_id]
        region = str(candidate["metadata"]["candidate_tuple"]["region_name"])
        group_id = f"{BATCH_ID}_{candidate_id}"
        assignments[candidate_id] = group_id
        relation_profiles[group_id] = {
            annotate._paper_id(paper): relation(category, annotate._paper_id(paper), region)
            for paper in candidate["literature"]
        }

    payload = {
        "schema_version": "1.0",
        "current_batch": BATCH_ID,
        "batches": [
            {
                "id": BATCH_ID,
                "method": "manual_abstract_and_excerpt_review_in_current_session",
                "hypothesis_count": len(candidates),
                "paper_assessment_count": sum(len(group) for group in relation_profiles.values()),
                "unique_papers": len(paper_profiles),
                "api_used": False,
            }
        ],
        "assignments": assignments,
        "paper_profiles": paper_profiles,
        "relation_profiles": relation_profiles,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "hypotheses": len(assignments),
                "assessments": sum(len(group) for group in relation_profiles.values()),
                "unique_papers": len(paper_profiles),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
