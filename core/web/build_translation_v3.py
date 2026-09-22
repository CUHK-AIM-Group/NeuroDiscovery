"""Build the v6 English display catalog (cs1_discovery_en_v3.json).

Template strings (title, machine hypothesis, plain hypothesis, definition) are
translated programmatically from the same frozen records that built the cards,
and every generated Chinese source is asserted byte-identical to the card text
before its translation is accepted. Free-text model answers (rationale,
transfer limits, feedback action, discussion) use the hand-written translations
below. The catalog is bound to the exact v6 pack hash; it also keeps every v2
string so historical v5 sessions stay fully translated.
"""
import hashlib
import json
import re
from pathlib import Path

from core.web.build_discovery_pilot_v6 import (
    SOURCE_HASHES, SOURCES, MATERIALS, describe, plain_hypothesis, roi_info,
)

PACK = MATERIALS / "cs1_discovery_pilot_v6.json"
PREVIOUS = MATERIALS / "cs1_discovery_en_v2.json"
TARGET = MATERIALS / "cs1_discovery_en_v3.json"

TITLE_EN = {"Vis": "visual", "SomMot": "somatomotor", "DorsAttn": "dorsal attention",
            "SalVentAttn": "salience/ventral attention", "Limbic": "limbic", "Cont": "control", "Default": "default"}
PLAIN_EN = {"Vis": "visual network", "SomMot": "sensorimotor network", "DorsAttn": "dorsal attention network",
            "SalVentAttn": "salience/ventral attention network", "Limbic": "limbic network",
            "Cont": "control network", "Default": "default network"}
GROUP_MACHINE_EN = {"ADHD": "ADHD", "bipolar": "bipolar disorder",
                    "psychosis_SZ_SZA": "schizophrenia/schizoaffective disorder"}
GROUP_PLAIN_EN = {"ADHD": "ADHD", "bipolar": "bipolar disorder",
                  "psychosis_SZ_SZA": "schizophrenia/schizoaffective"}

CJK = re.compile(r"[㐀-鿿]")


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


def en_title(record):
    spec = record["canonical_record"]["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        if a == b:
            return f"Within-network {TITLE_EN[a]} connectivity"
        return f"{TITLE_EN[a].capitalize()} and {TITLE_EN[b]} networks"
    roi = roi_info(record)
    net = record["endpoint"].split(":")[-1]
    return f"{'Left' if '_LH_' in roi['name'] else 'Right'} ROI{roi['roi']:03} and the {TITLE_EN[net]} network"


def en_hypothesis(record):
    canonical = record["canonical_record"]
    groups = " and ".join(GROUP_MACHINE_EN[g] for g in canonical["named"])
    return (f"{record['endpoint']}; mean Pearson r is lower in each complete predefined TCP case group "
            f"({groups}) relative to predefined controls, without selection by disease state or other subgroups.")


def en_plain(record):
    canonical = record["canonical_record"]
    named = canonical["named"]
    if len(named) != 2 or any(g not in GROUP_PLAIN_EN for g in named):
        raise ValueError(f"Plain template covers two named groups, got {named}")
    spec = canonical["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        subject = (f"Functional connectivity between the {PLAIN_EN[a]} ({a}) and the {PLAIN_EN[b]} ({b})"
                   if a != b else f"Functional connectivity among parcels within the {PLAIN_EN[a]} ({a})")
    else:
        roi = roi_info(record)
        net = record["endpoint"].split(":")[-1]
        subject = (f"Functional connectivity between the {'left' if '_LH_' in roi['name'] else 'right'} "
                   f"ROI{roi['roi']:03} parcel and the {PLAIN_EN[net]} ({net})")
    return (f"{subject} is expected to be lower in both {GROUP_PLAIN_EN[named[0]]} and {GROUP_PLAIN_EN[named[1]]} "
            f"patients than in their respective control groups — a decrease in the same direction in both groups.")


def en_definition(record, counts):
    spec = record["canonical_record"]["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        edges = counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) // 2
        if a == b:
            return (f"Schaefer-400 / 7-network; the arithmetic mean of {edges:,} unique off-diagonal Pearson r edges "
                    f"within {a} ({counts[a]} parcels), without Fisher transformation or outcome-driven edge selection.")
        return (f"Schaefer-400 / 7-network; the arithmetic mean of {edges:,} unique off-diagonal Pearson r edges "
                f"between {a} ({counts[a]} parcels) and {b} ({counts[b]} parcels), "
                f"without Fisher transformation or outcome-driven edge selection.")
    roi = roi_info(record)
    net = record["endpoint"].split(":")[-1]
    edges = counts[net] - (1 if roi["network"] == net else 0)
    return (f"Schaefer-400 / 7-network; ROI{roi['roi']:03} = {roi['name']}. The arithmetic mean of {edges} "
            f"off-diagonal Pearson r edges from this parcel to {net}; it is not equivalent to the entire anatomical "
            f"region of the same name. No Fisher transformation or outcome-driven edge selection.")


# Hand-written translations for free-text model answers and shared v6 strings.
FREETEXT = {
    # packet-01
    "跨诊断动态FC示突显网络降低(P1)，突显网络同伦降低(P2)，患者短程连接降低(P3)，故该突显网络内边降低。":
        "Transdiagnostic dynamic FC shows reduced salience-network connectivity (P1), reduced salience-network homotopic connectivity (P2), and reduced short-range connectivity in patients (P3); hence this within-salience-network edge is hypothesized to decrease.",
    "P1为动态FC、P2为同伦、P3为距离依赖，均非静态边级r；P3同时报告长程连接增高，方向异质；父节点5accc8b6联合稳定性0.51、成分beta近零。":
        "P1 is dynamic FC, P2 is homotopic connectivity and P3 is distance-dependent connectivity; none is a static edge-level r. P3 also reports increased long-range connectivity, so directions are heterogeneous. Parent node 5accc8b6 has joint stability 0.51 and near-zero component betas.",
    "采纳三方评审：补记[已移除引文]→P3替换（P3为距离依赖指标），转移限制补注父节点0.51稳定性与近零beta，说明-1先验支持有限。":
        "Adopted the three reviewers' comments: recorded the [citation removed]→P3 replacement (P3 is a distance-dependent measure), and added the parent's 0.51 stability and near-zero betas to the transfer limits, noting limited prior support for the -1 direction.",
    "采纳三方评审：修订转移限制与反馈动作，补注父节点0.51稳定性及[已移除引文]→P3替换。":
        "Adopted the three reviewers' comments: revised the transfer limits and feedback action, adding the parent's 0.51 stability and the [citation removed]→P3 replacement.",
    # packet-04
    "注意网络同伦连接降低（P1），精神分裂整体连接强度减弱（P2），跨诊断动态FC示额顶网络连接降低（P3），故该感觉运动-注意边在病例中降低。":
        "Homotopic connectivity in attention networks is reduced (P1), global connectivity strength is reduced in schizophrenia (P2), and transdiagnostic dynamic FC shows reduced frontoparietal connectivity (P3); hence this somatomotor–attention edge is hypothesized to decrease in patients.",
    "P1为同伦指标、P2为n=27单研究且仅精神分裂、P3为动态FC且不含背侧注意网络，均非边级静态r；[已移除引文]示SZ额顶连接增高，方向异质；转移不确定。":
        "P1 is a homotopic measure, P2 a single schizophrenia-only study (n=27), and P3 dynamic FC that does not cover the dorsal attention network; none is a static edge-level r. [citation removed] reports increased frontoparietal connectivity in SZ, so directions are heterogeneous; transfer is uncertain.",
    "采纳方法学与临床评审：理由改为'该感觉运动-注意边'，转移限制补注P3不含背侧注意网络与P2单研究局限。":
        "Adopted the methodology and clinical reviews: the rationale now says 'this somatomotor–attention edge', and the transfer limits note that P3 does not cover the dorsal attention network and that P2 is a single study.",
    "采纳临床与方法学评审：端点036的ROI属感觉运动网络，理由须改为感觉运动-注意边，并披露P3不含背侧注意网络。":
        "Adopted the clinical and methodology reviews: the ROI of endpoint 036 belongs to the somatomotor network, so the rationale had to describe a somatomotor–attention edge, and it had to be disclosed that P3 does not cover the dorsal attention network.",
    # packet-05
    "P1来源分析报告双相与精神病性障碍躯体运动网络内连接下降，故聚焦该网络内均值作探索。":
        "The source analysis behind P1 reports reduced within-somatomotor-network connectivity in bipolar and psychotic disorders, so the within-network mean is explored.",
    "P1的FDR/q值属于来源分析，不是TCP结果；网络内与TCP聚合均值不等价，且无TCP效应量、P值或校准CI。":
        "The FDR/q values of P1 belong to the source analysis, not to TCP results; within-network values are not equivalent to the TCP aggregate mean, and there is no TCP effect size, P value or calibrated CI.",
    "无既往TCP反馈；本次按审阅收窄诊断与证据，仍仅作初始探索。":
        "No prior TCP feedback; diagnoses and evidence were narrowed as reviewed, and the card remains an initial exploration.",
    "回应生统、临床对ADHD外推和证据异质性的意见，按审阅聚焦P1并收窄诊断。":
        "In response to the biostatistics and clinical comments on ADHD extrapolation and evidence heterogeneity, the card focuses on P1 and narrows the diagnoses as reviewed.",
    # packet-06
    "跨诊断动态FC研究示感觉运动与额顶网络连接降低（P1），患者短程连接降低（P2），精神分裂整体连接减弱（P3），故该注意-感觉运动边降低。":
        "Transdiagnostic dynamic FC studies show reduced somatomotor and frontoparietal connectivity (P1), patients show reduced short-range connectivity (P2), and global connectivity is reduced in schizophrenia (P3); hence this attention–somatomotor edge is hypothesized to decrease.",
    "P1为动态FC、P2为距离依赖、P3为全局拓扑，均非静态ROI-网络均值r；转移至混合TCP不确定，存在用药混杂。":
        "P1 is dynamic FC, P2 is distance-dependent and P3 is global topology; none is a static ROI-to-network mean r. Transfer to the mixed TCP is uncertain, and medication confounding exists.",
    "采纳方法学评审：保留SomMot-1方向，端点改为271:SomMot，证据改用P1/P2/P3。":
        "Adopted the methodology review: kept the SomMot -1 direction, changed the endpoint to 271:SomMot, and switched the evidence to P1/P2/P3.",
    "三方评审一致保留：端点271:SomMot未重复，P1/P2/P3度量错配已披露，父节点0.947可排序，故原样保留。":
        "All three reviewers agreed to keep: endpoint 271:SomMot is not duplicated, the P1/P2/P3 metric mismatches are disclosed, and the parent node's 0.947 stability allows ranking, so the card is kept unchanged.",
    # packet-07
    "默认-突显跨网络边：SZ/BD/MDD突显网络内连接降低(P1)，SZ任务正-负网络降低(P2)，DMN-突显连接与症状相关(P3，无方向)。":
        "A default–salience cross-network edge: within-salience connectivity is reduced in SZ/BD/MDD (P1), task positive–negative network connectivity is reduced in SZ (P2), and DMN–salience connectivity is symptom-related (P3, no direction).",
    "P1为突显网络内动态FC且含MDD，P2仅SZ小样本，P3无方向；非该特定跨网络边，方向不确定。":
        "P1 is within-salience dynamic FC and includes MDD, P2 is an SZ-only small sample, and P3 has no direction; none measures this specific cross-network edge, so the direction is uncertain.",
    "父假设371...为突显网络边；本轮换用新突显端点155，保持双相+精神病与-1，并注明P1为网络内证据。":
        "Parent hypothesis 371... is a salience-network edge; this round switches to the new salience endpoint 155, keeps bipolar+psychosis and the -1 direction, and notes that P1 is within-network evidence.",
    "三位评审均判keep：端点155为默认ROI至突显网络，跨网络标注正确，P1网络内动态FC与小样本限制已注明。":
        "All three reviewers judged keep: endpoint 155 is a default-network ROI to the salience network, the cross-network label is correct, and P1's within-network dynamic FC and small-sample limits are noted.",
    # packet-08
    "KG证据显示精神分裂症视觉加工通路半球间连接降低，精神分裂症与精神病性双相视觉加工区连接存在差异，精神病视觉网络复杂性破坏；据此假设该视觉网络内连接降低。":
        "KG evidence shows reduced interhemispheric connectivity in visual processing pathways in schizophrenia, altered connectivity of visual processing regions in schizophrenia and psychotic bipolar disorder, and disrupted visual-network complexity in psychosis; on this basis, reduced connectivity within this visual network is hypothesized.",
    "P1仅测半球间同源边，端点平均含全部视觉网络内边；P2为组间比较，双相方向无直接证据；P3无方向。":
        "P1 measures only interhemispheric homologous edges while the endpoint averages all within-visual-network edges; P2 is a between-group comparison with no direct evidence for the bipolar direction; P3 has no direction.",
    "父假设[父假设]方向稳定性较高；依审查保留-1并换用未测试的视觉网络内端点以扩展主题。":
        "The parent hypothesis [parent hypothesis] has relatively high directional stability; as reviewed, the -1 direction is kept and an untested within-visual-network endpoint is used to broaden coverage.",
    "依生物统计师审查修订转移限制：P1仅覆盖半球间同源边，端点平均含全部视觉网络内边，P2为组间比较且双相方向无直接证据。":
        "Transfer limits revised as the biostatistician required: P1 covers only interhemispheric homologous edges while the endpoint averages all within-network edges, and P2 is a between-group comparison with no direct evidence for the bipolar direction.",
    # packet-09
    "初始探索：P2报告视觉网络内连接下降；P1支持精神分裂症视觉相关连接降低，P3提示双相视觉网络异常。":
        "Initial exploration: P2 reports reduced within-visual-network connectivity; P1 supports reduced vision-related connectivity in schizophrenia, and P3 suggests visual-network abnormalities in bipolar disorder.",
    "KG证据多未经独立复核且来自异质任务；与TCP聚合端点不完全同构，方向仅作探索性假设。":
        "Most KG evidence has not been independently re-checked and comes from heterogeneous tasks; it is not fully isomorphic to the TCP aggregate endpoint, so the direction is only an exploratory hypothesis.",
    "无既往TCP反馈，本卡用于初始探索，不作反馈驱动调整。":
        "No prior TCP feedback; this card is an initial exploration with no feedback-driven adjustment.",
    "采纳三方pass1共识，保留原卡；以P2为主，不把来源FDR结果外推为TCP显著性。":
        "Adopted the three reviewers' pass1 consensus: keep the original card, rely mainly on P2, and do not extrapolate source FDR results into TCP significance.",
    # packet-10
    "P1来源分析在双相与精神病性障碍中报告显著网络内连接下降，支持SalVentAttn内均值作探索。":
        "The source analysis behind P1 reports significant within-network decreases in bipolar and psychotic disorders, supporting exploration of the within-SalVentAttn mean.",
    "来源q<.05/FDR校正不等于TCP显著；网络内结果不保证TCP全病例均值方向或精度，无TCP效应量、P值和校准CI。":
        "Source-level q<.05/FDR correction does not equal TCP significance; within-network results do not guarantee the direction or precision of the TCP all-cases mean, and there is no TCP effect size, P value or calibrated CI.",
    "无既往TCP反馈；本次按审阅收窄诊断并限定来源显著性，仍仅作初始探索。":
        "No prior TCP feedback; diagnoses were narrowed and source-level significance was qualified as reviewed, and the card remains an initial exploration.",
    "回应三方指出的ADHD外推、方向不足及来源FDR误读，删ADHD并仅用P1。":
        "In response to the three reviewers' comments on ADHD extrapolation, insufficient direction and misread source FDR, ADHD was removed and only P1 is used.",
    # packet-11 / packet-15 / packet-19 (shared rationale)
    "注意网络同伦连接降低(P1)，精神分裂整体连接减弱(P2)，跨诊断动态FC降低(P3)，故该感觉运动-注意边降低。":
        "Homotopic connectivity in attention networks is reduced (P1), global connectivity is reduced in schizophrenia (P2), and transdiagnostic dynamic FC is reduced (P3); hence this somatomotor–attention edge is hypothesized to decrease.",
    "P1为同伦指标、P2为单研究且仅精神分裂、P3为动态FC，均非边级静态r；转移至混合TCP不确定。":
        "P1 is a homotopic measure, P2 a single schizophrenia-only study and P3 dynamic FC; none is a static edge-level r, so transfer to the mixed TCP is uncertain.",
    "沿用DorsAttn-1父节点a93e510d，端点改为未覆盖的255以拓宽主题。":
        "Followed the DorsAttn -1 parent node a93e510d and changed the endpoint to the uncovered 255 to broaden coverage.",
    "三方两轮均判keep：ROI 255属SomMot，边为感觉运动-注意，理由与指标一致；P1同伦、P2单研究、P3动态FC的度量错配已在转移限制披露。":
        "All three reviewers judged keep in both passes: ROI 255 belongs to SomMot, the edge is somatomotor–attention, and the rationale matches the measure; the metric mismatches (P1 homotopic, P2 single study, P3 dynamic FC) are disclosed in the transfer limits.",
    # packet-12
    "仅P1提供双相与精神病多网络内下降线索；共同降低仅作低先验可证伪探索。":
        "Only P1 provides clues of within-network decreases across multiple networks in bipolar disorder and psychosis; a joint decrease is explored only as a low-prior, falsifiable hypothesis.",
    "P1为动态网络内结果，不等同TCP两网络全边均值；方向、效应量、P值和校准CI均需独立估计。":
        "P1 is a dynamic within-network result and is not equivalent to the TCP mean over all edges between the two networks; direction, effect size, P value and calibrated CI all require independent estimation.",
    "依据父卡[父假设]并回应评审，删除方向混合且未明确覆盖双相的[已移除引文]、[已移除引文]。":
        "Following the parent card [parent hypothesis] and responding to the reviews, [citation removed] and [citation removed] — direction-mixed and not clearly covering bipolar disorder — were removed.",
    "回应评审关于[已移除引文]未覆盖双相、[已移除引文]方向混合的意见，删除二者，仅保留P1并收紧转移边界。":
        "In response to reviewers' comments that [citation removed] does not cover bipolar disorder and that [citation removed] is direction-mixed, both were removed; only P1 is kept and the transfer boundary is tightened.",
    # packet-13
    "突显-默认跨网络边：P1为ADHD症状维度关联、P2为DMN-突显连接与症状相关、P3为跨图谱额叶缺陷汇总，三项均无方向，-1为探索性。":
        "A salience–default cross-network edge: P1 is an ADHD symptom-dimension association, P2 links DMN–salience connectivity to symptoms, and P3 summarizes cross-atlas frontal deficits; none gives a direction, so -1 is exploratory.",
    "P1为ADHD儿童维度关联非病例-对照，P2/P3无方向且P3非默认网络特异；该边为突显-默认跨网络，外推有限，方向不确定。":
        "P1 is a dimensional association in children with ADHD, not case–control; P2/P3 have no direction and P3 is not default-network-specific. This is a salience–default cross-network edge, so extrapolation is limited and the direction uncertain.",
    "依methodology_expert/pass1，更正动作：命名集合由父假设的双相+精神病改为ADHD+精神病，并将端点312改述为突显-默认跨网络边。":
        "Per methodology_expert/pass1, the action was corrected: the named set changed from the parent's bipolar+psychosis to ADHD+psychosis, and endpoint 312 was re-described as a salience–default cross-network edge.",
    "采纳methodology_expert/pass1：更正命名集合变更描述，改述为突显-默认跨网络边并注明三项证据均无方向。":
        "Adopted methodology_expert/pass1: corrected the description of the named-set change, re-described the edge as salience–default cross-network, and noted that all three pieces of evidence lack direction.",
    # packet-14
    "跨诊断动态FC示突显网络连接降低(P1)，突显网络同伦降低(P2)，患者短程连接降低(P3)，故该突显网络内边降低。":
        "Transdiagnostic dynamic FC shows reduced salience-network connectivity (P1), reduced salience-network homotopic connectivity (P2), and reduced short-range connectivity in patients (P3); hence this within-salience-network edge is hypothesized to decrease.",
    "P1为动态FC、P2为同伦、P3为距离依赖，均非静态边级r；P3同时报告长程连接增高，方向异质；转移至混合TCP不确定。":
        "P1 is dynamic FC, P2 is homotopic and P3 is distance-dependent; none is a static edge-level r. P3 also reports increased long-range connectivity, so directions are heterogeneous; transfer to the mixed TCP is uncertain.",
    "采纳方法学评审：在反馈动作补记父节点[已移除引文]→P3替换（P3为静态距离依赖指标），端点106与-1方向不变。":
        "Adopted the methodology review: the feedback action now records the parent node's [citation removed]→P3 replacement (P3 is a static distance-dependent measure); endpoint 106 and the -1 direction are unchanged.",
    "采纳方法学评审：父节点5accc8b6用[已移除引文]，本卡改用P3，反馈动作未记录该证据替换，须补记。":
        "Adopted the methodology review: parent node 5accc8b6 used [citation removed] while this card uses P3, and the feedback action had not recorded that evidence replacement, so it had to be added.",
    # packet-15
    "P1为同伦指标、P2为单研究且仅精神分裂、P3为动态FC且不含背侧注意网络，均非边级静态r；[已移除引文]示SZ额顶连接增高，方向异质；转移至混合TCP不确定。":
        "P1 is a homotopic measure, P2 a single schizophrenia-only study, and P3 dynamic FC that does not cover the dorsal attention network; none is a static edge-level r. [citation removed] reports increased frontoparietal connectivity in SZ, so directions are heterogeneous; transfer to the mixed TCP is uncertain.",
    "采纳方法学与临床评审：在转移限制补注P3不含背侧注意网络、属跨网络外推；保留-1方向与父节点a93e510d。":
        "Adopted the methodology and clinical reviews: the transfer limits now note that P3 does not cover the dorsal attention network and that this is cross-network extrapolation; the -1 direction and parent node a93e510d are kept.",
    "采纳方法学与临床评审：P3仅覆盖视觉/感觉运动/突显/额顶，不含背侧注意网络，须在转移限制补注该跨网络外推。":
        "Adopted the methodology and clinical reviews: P3 covers only visual/somatomotor/salience/frontoparietal networks, not the dorsal attention network, so this cross-network extrapolation had to be added to the transfer limits.",
    # packet-16
    "注意网络同伦连接降低（P1）且SZ整体连接减弱（P2），跨诊断感觉运动降低（P3），故该感觉运动-注意边降低。":
        "Homotopic connectivity in attention networks is reduced (P1), global connectivity is reduced in SZ (P2), and transdiagnostic somatomotor connectivity is reduced (P3); hence this somatomotor–attention edge is hypothesized to decrease.",
    "P1为同伦指标，P2为全局连接，非该具体边；[已移除引文]示SZ额顶连接增高，方向异质；转移不确定。":
        "P1 is a homotopic measure and P2 a global-connectivity result, not this specific edge; [citation removed] reports increased frontoparietal connectivity in SZ, so directions are heterogeneous and transfer is uncertain.",
    "采纳反馈：保留SomMot主题但改为DorsAttn端点，使用P1/P2/P3。":
        "Adopted the feedback: kept the SomMot theme but changed to a DorsAttn endpoint, using P1/P2/P3.",
    "三位评审均判keep：端点237:DorsAttn在池内，-1方向与P3一致，转移限制已披露P1同伦、P2全局拓扑与[已移除引文]方向异质。":
        "All three reviewers judged keep: endpoint 237:DorsAttn is in the pool, the -1 direction matches P3, and the transfer limits disclose P1's homotopic metric, P2's global topology and [citation removed]'s heterogeneous direction.",
    # packet-17
    "突显-默认跨网络边：双相抑郁DMN连接降低(P1/P3，P3方向混合)，精神分裂症DMN异常(P2，无方向)，DMN-突显连接与症状相关(P4，无方向)；提示该跨网络边可能降低。":
        "A salience–default cross-network edge: DMN connectivity is reduced in bipolar depression (P1/P3, P3 direction-mixed), DMN abnormalities exist in schizophrenia (P2, no direction), and DMN–salience connectivity is symptom-related (P4, no direction); this suggests the cross-network edge may decrease.",
    "P1/P3为双相抑郁亚组且P3方向混合，P2为综述无方向，P4无方向；三项为DMN网络内证据，跨网络外推有限，方向不确定。":
        "P1/P3 are bipolar-depression subgroups and P3 is direction-mixed, P2 is a review without direction, and P4 has no direction; all three are within-DMN evidence, so cross-network extrapolation is limited and the direction uncertain.",
    "依methodology_expert/pass1与biostatistician/pass1，将端点316改述为突显-默认跨网络边，加入P4直接跨网络证据并注明DMN内证据外推有限。":
        "Per methodology_expert/pass1 and biostatistician/pass1, endpoint 316 was re-described as a salience–default cross-network edge, P4 was added as direct cross-network evidence, and the limited extrapolation of within-DMN evidence was noted.",
    "采纳methodology_expert/pass1与biostatistician/pass1：端点316为突显网络ROI至默认网络，属跨网络边，理由与限制须改述并补P4直接跨网络证据。":
        "Adopted methodology_expert/pass1 and biostatistician/pass1: endpoint 316 is a salience-network ROI to the default network — a cross-network edge — so the rationale and limits had to be re-described and P4 added as direct cross-network evidence.",
    # packet-18
    "沿用SalVentAttn-1父节点5accc8b6，端点改为未覆盖的107以拓宽主题。":
        "Followed the SalVentAttn -1 parent node 5accc8b6 and changed the endpoint to the uncovered 107 to broaden coverage.",
    "三方pass1均判keep：ROI 107属突显网络，理由与端点一致；动态FC、同伦与距离依赖错配及P3方向异质已披露。":
        "All three reviewers judged keep at pass1: ROI 107 belongs to the salience network and the rationale matches the endpoint; the dynamic-FC, homotopic and distance-dependent mismatches and P3's heterogeneous direction are disclosed.",
    # packet-19
    "P1为同伦、P2为单研究且仅精神分裂、P3为动态FC且所报网络不含背侧注意，均非边级静态r；[已移除引文]示SZ额顶连接增高，方向异质。":
        "P1 is homotopic, P2 a single schizophrenia-only study, and P3 dynamic FC whose reported networks do not include dorsal attention; none is a static edge-level r. [citation removed] reports increased frontoparietal connectivity in SZ, so directions are heterogeneous.",
    "采纳三方评审：在转移限制补注P3所报网络不含背侧注意、该边注意侧属范围外推，并注明[已移除引文]示SZ额顶连接增高、方向异质。":
        "Adopted the three reviewers' comments: the transfer limits now note that P3's reported networks exclude dorsal attention and that the attention side of this edge is out-of-range extrapolation, and that [citation removed] reports increased frontoparietal connectivity in SZ with heterogeneous direction.",
    "采纳生物统计与方法学评审：仅修订转移限制，补注P3网络范围与[已移除引文]方向异质；端点、命名、方向与证据集不变。":
        "Adopted the biostatistics and methodology reviews: only the transfer limits were revised, adding P3's network coverage and [citation removed]'s heterogeneous direction; endpoint, naming, direction and evidence set are unchanged.",
    # packet-20
    "跨诊断动态FC示感觉运动降低(P1)，患者短程连接降低(P2)，精神分裂整体连接减弱(P3)，故该注意-感觉运动边降低。":
        "Transdiagnostic dynamic FC shows reduced somatomotor connectivity (P1), patients show reduced short-range connectivity (P2), and global connectivity is reduced in schizophrenia (P3); hence this attention–somatomotor edge is hypothesized to decrease.",
    "P1为动态FC且属感觉运动网络内证据、P2为距离依赖、P3为单研究且仅精神分裂，均非边级静态r；该边背侧注意侧属跨网络外推。":
        "P1 is dynamic FC and within-somatomotor evidence, P2 is distance-dependent, and P3 a single schizophrenia-only study; none is a static edge-level r, and the dorsal-attention side of this edge is cross-network extrapolation.",
    "沿用SomMot-1父节点6d6f7619，端点改为未覆盖的289以拓宽主题。":
        "Followed the SomMot -1 parent node 6d6f7619 and changed the endpoint to the uncovered 289 to broaden coverage.",
    "三方两轮均判keep：端点289:SomMot在池内，理由'注意-感觉运动边'与端点结构一致，P1网络内证据与P3单研究局限已披露。":
        "All three reviewers judged keep in both passes: endpoint 289:SomMot is in the pool, the rationale 'attention–somatomotor edge' matches the endpoint structure, and P1's within-network evidence and P3's single-study limit are disclosed.",
    # packet-21
    "跨诊断动态FC示视觉网络降低(P1)，视觉同伦降低(P2)，患者短程连接降低(P3)，故该突显-视觉边降低。":
        "Transdiagnostic dynamic FC shows reduced visual-network connectivity (P1), reduced visual homotopic connectivity (P2), and reduced short-range connectivity in patients (P3); hence this salience–visual edge is hypothesized to decrease.",
    "P1为动态FC、P2为同伦、P3为距离依赖，均为视觉网络内部证据，非该突显-视觉边；P3同时报告长程连接增高，方向异质；跨网络外推不确定。":
        "P1 is dynamic FC, P2 is homotopic and P3 is distance-dependent — all within-visual-network evidence, not this salience–visual edge. P3 also reports increased long-range connectivity, so directions are heterogeneous; cross-network extrapolation is uncertain.",
    "采纳方法学pass1：理由与转移限制改为'该突显-视觉边'，并补注P3长程连接增高的方向异质。":
        "Adopted methodology pass1: the rationale and transfer limits now say 'this salience–visual edge', with a note on P3's heterogeneous direction of increased long-range connectivity.",
    "方法学pass0/pass1与生统pass1一致：ROI 315属突显网络，该边为突显-视觉，理由与转移限制误称注意-视觉边。":
        "Methodology pass0/pass1 and biostatistics pass1 agreed: ROI 315 belongs to the salience network and the edge is salience–visual; the rationale and transfer limits had mislabeled it as an attention–visual edge.",
    # packet-22
    "P1、P2、P3提供运动相关或精神病连接指标降低线索；仅作双诊断ROI低方向探索。":
        "P1, P2 and P3 provide clues of reduced motor-related or psychosis connectivity measures; this is only a two-diagnosis ROI decrease-direction exploration.",
    "证据来自局部回路、疾病阶段及异质指标，未直接验证该ROI—网络边；不外推共同效应、特异性或因果关系。":
        "The evidence comes from local circuits, illness stages and heterogeneous measures and does not directly test this ROI–network edge; no common effect, specificity or causality is extrapolated.",
    "承接[父假设]的ADHD与精神病低方向可排名反馈，细化至新ROI 109。":
        "Following the rankable ADHD-and-psychosis decrease feedback of [parent hypothesis], refined to the new ROI 109.",
    "三位专家均认为端点与范围可执行；保留原卡，不将间接方向或父反馈稳定性解释为本ROI的显著性、共同效应或因果结果。":
        "All three experts considered the endpoint and scope executable; the original card is kept, and indirect direction or parent-feedback stability is not interpreted as significance, a common effect or a causal result for this ROI.",
    # packet-23
    "跨诊断动态FC研究示感觉运动与额顶网络连接降低（P1），患者短程连接降低（P2），注意网络同伦连接降低（P3），故该感觉运动-注意边降低。":
        "Transdiagnostic dynamic FC studies show reduced somatomotor and frontoparietal connectivity (P1), patients show reduced short-range connectivity (P2), and homotopic connectivity in attention networks is reduced (P3); hence this somatomotor–attention edge is hypothesized to decrease.",
    "P1为动态FC、P2为距离依赖、P3为同伦指标，均非静态ROI-网络均值r；已删去皮层下范围错配的[已移除引文]；转移至混合TCP不确定，存在用药混杂。":
        "P1 is dynamic FC, P2 is distance-dependent and P3 is a homotopic measure; none is a static ROI-to-network mean r. [citation removed], whose subcortical scope mismatched, has been removed. Transfer to the mixed TCP is uncertain, and medication confounding exists.",
    "采纳三方评审：删去皮层下范围错配的[已移除引文]，改用皮层证据P3，并同步改写理由与转移限制。":
        "Adopted the three reviewers' comments: removed [citation removed] with its subcortical scope mismatch, switched to cortical evidence P3, and rewrote the rationale and transfer limits accordingly.",
    "采纳三方评审：删去皮层下范围错配的[已移除引文]，改用皮层证据P3，同步改写理由与转移限制，纠正父节点已承诺的删除。":
        "Adopted the three reviewers' comments: removed [citation removed] with its subcortical scope mismatch, switched to cortical evidence P3, rewrote the rationale and transfer limits accordingly, and implemented the deletion already promised at the parent node.",
    # packet-24
    "P1报告双相与精神病视觉、躯体运动网络内下降，P2与P3补充精神病视觉及躯体感觉/枕叶连接降低，检验SomMot-Vis负向。":
        "P1 reports within-network decreases in visual and somatomotor networks in bipolar disorder and psychosis, and P2 and P3 add reduced visual and somatosensory/occipital connectivity in psychosis; a negative SomMot–Vis direction is tested.",
    "P2为任务局部、P3为正则化区域、P1为动态网络内结果；均非SomMot-Vis全边均值，TCP统计量需独立估计。":
        "P2 is task-local, P3 covers regularized regions and P1 is a dynamic within-network result; none is the SomMot–Vis all-edge mean, so TCP statistics must be estimated independently.",
    "依据父卡[父假设]的三诊断SomMot-Vis高方向稳定，聚焦双相与精神病，避免把稳定性直接当作效应证据。":
        "Following the parent card [parent hypothesis] with high three-diagnosis SomMot–Vis directional stability, the focus is narrowed to bipolar disorder and psychosis, avoiding treating stability directly as effect evidence.",
    "六份审阅均认可端点与限制；保留原卡，来源方向和显著性不直接迁移到TCP跨网均值。":
        "All six reviews accepted the endpoint and limits; the original card is kept, and source direction and significance do not transfer directly to the TCP cross-network mean.",
    # packet-25
    "突显-默认跨网络边：双相抑郁DMN连接降低(P1/P2，P2方向混合)，精神分裂症DMN异常(P3，无方向)，DMN-突显连接与症状相关(P4，无方向)。":
        "A salience–default cross-network edge: DMN connectivity is reduced in bipolar depression (P1/P2, P2 direction-mixed), DMN abnormalities exist in schizophrenia (P3, no direction), and DMN–salience connectivity is symptom-related (P4, no direction).",
    "P1/P2为双相抑郁亚组且P2方向混合，P3为综述无方向，P4无方向；DMN内证据跨网络外推有限。":
        "P1/P2 are bipolar-depression subgroups and P2 is direction-mixed, P3 is a review without direction, and P4 has no direction; within-DMN evidence has limited cross-network extrapolation.",
    "父假设83e...为默认网络边；本轮换用新默认端点317，保持双相+精神病与-1，并注明DMN内证据外推限制。":
        "Parent hypothesis 83e... is a default-network edge; this round switches to the new default endpoint 317, keeps bipolar+psychosis and the -1 direction, and notes the extrapolation limits of within-DMN evidence.",
    "三位评审均判keep：端点317为突显ROI至默认网络，跨网络标注与端点结构一致，方向证据未定已注明，保持原稿。":
        "All three reviewers judged keep: endpoint 317 is a salience ROI to the default network, the cross-network label matches the endpoint structure, and the undetermined directional evidence is noted, so the draft is kept.",
    # packet-26
    "KG证据显示双相抑郁皮层区域连接降低、双相默认网络内低连接；精神病网络复杂性破坏但无方向，据此弱化地假设该默认颞区到背侧注意网络连接降低。":
        "KG evidence shows reduced cortical connectivity in bipolar depression and within-default-network hypoconnectivity in bipolar disorder; psychosis shows disrupted network complexity without a direction. On this basis, a weakened hypothesis of reduced connectivity from this default temporal region to the dorsal attention network is made.",
    "P1为双相抑郁对单相抑郁的组间比较，无健康对照方向；P3为无方向综述；未直接检验该边。":
        "P1 is a between-group comparison of bipolar versus unipolar depression without a healthy-control direction; P3 is a review without direction; this edge has not been directly tested.",
    "沿用父假设[父假设]的默认到背侧注意降低方向，换用未测试的背侧注意端点155。":
        "Followed the parent hypothesis [parent hypothesis]'s default-to-dorsal-attention decrease direction and switched to the untested dorsal-attention endpoint 155.",
    "三位审查者均判keep：端点155为默认颞区到背侧注意网络的跨网络边，标签与理由一致，P1组间比较无对照方向与P3无方向均已披露。":
        "All three reviewers judged keep: endpoint 155 is a cross-network edge from a default temporal region to the dorsal attention network, the label matches the rationale, and both P1's between-group comparison without a control direction and P3's lack of direction are disclosed.",
    # packet-27
    "P1/P2仅为小脑或FEF/丘脑局部线索，P3无方向；DorsAttn–SomMot低向是反馈后待检编码。":
        "P1/P2 are only local clues from the cerebellum or FEF/thalamus, and P3 has no direction; the DorsAttn–SomMot decrease is a post-feedback candidate encoding to be tested.",
    "局部结构与跨网均值不等价；P3无方向，本端点无统一N、效应量、P值、校准CI；诊断缩减属适应性选择，未校正多重性。":
        "Local structures are not equivalent to a cross-network mean; P3 has no direction, and this endpoint has no pooled N, effect size, P value or calibrated CI; the diagnosis reduction is an adaptive choice without multiplicity correction.",
    "无新TCP结果；承接父项[父假设]低向反馈，将范围缩为双诊断并仅作反馈后探索；不作确证并披露多重性。":
        "No new TCP results; following the parent item [parent hypothesis]'s decrease feedback, the scope was reduced to two diagnoses as post-feedback exploration only; no confirmation is claimed and multiplicity is disclosed.",
    "采纳三方pass1：补端点映射与统计不确定性，说明诊断缩减为反馈后探索，去除预注册确认意味。":
        "Adopted the three reviewers' pass1 comments: added endpoint mapping and statistical uncertainty, stated that the diagnosis reduction is post-feedback exploration, and removed any preregistered-confirmation implication.",
    # packet-28
    "P1、P2和P3提供默认或全局连接降低/受损线索；检验两诊断默认网络内较低方向。":
        "P1, P2 and P3 provide clues of reduced or impaired default/global connectivity; a lower within-default-network direction in the two diagnoses is tested.",
    "KG证据均为间接、异质且未独立复核的外部材料，未直接验证本TCP边；不能断言诊断特异性或因果关系。":
        "The KG evidence is indirect, heterogeneous external material that has not been independently re-checked and does not directly test this TCP edge; no diagnosis specificity or causality can be asserted.",
    "无既往反馈；本轮为初始探索，按固定全体病例—对照范围检验该方向。":
        "No prior feedback; this round is an initial exploration, testing the direction under the fixed all-cases-versus-controls scope.",
    "三位专家最终一致支持保留；应用其提醒，将P1和P3视为跨结构及全局的间接线索。":
        "All three experts ultimately agreed to keep the card; applying their cautions, P1 and P3 are treated as indirect cross-structure and global clues.",
    # packet-29
    "KG证据显示双相静息态默认网络内低连接，精神病存在网络时空复杂性破坏但无方向；P1为双相抑郁对单相抑郁的组间比较，不提供相对对照方向；据此弱化地假设该左侧默认颞区到背侧注意网络连接降低。":
        "KG evidence shows resting-state within-default-network hypoconnectivity in bipolar disorder and disrupted spatiotemporal network complexity in psychosis without direction; P1 is a between-group comparison of bipolar versus unipolar depression and gives no control-referenced direction. On this basis, a weakened hypothesis of reduced connectivity from this left default temporal region to the dorsal attention network is made.",
    "P1为双相抑郁对单相抑郁的组间比较，无健康对照方向；P3为无方向复杂性综述；均未直接检验该左侧默认颞区到背侧注意网络的边。":
        "P1 is a between-group comparison of bipolar versus unipolar depression without a healthy-control direction, and P3 is a complexity review without direction; neither directly tests the edge from this left default temporal region to the dorsal attention network.",
    "依methodology_expert/pass1要求，将转移限制改为披露P1为BD对UD组间比较、无健康对照方向，并相应弱化理由表述。":
        "As required by methodology_expert/pass1, the transfer limits were changed to disclose that P1 is a BD-versus-UD between-group comparison without a healthy-control direction, and the rationale was weakened accordingly.",
    "采纳methodology_expert/pass1的required_changes：P1为双相抑郁对单相抑郁的组间比较，无健康对照方向，原转移限制误标为亚组；改写两字段披露该设计限制并弱化方向。":
        "Adopted methodology_expert/pass1's required_changes: P1 is a between-group comparison of bipolar versus unipolar depression without a healthy-control direction, and the original transfer limits mislabeled it as a subgroup; both fields were rewritten to disclose this design limitation and weaken the direction.",
    # packet-31
    "跨诊断动态FC示感觉运动降低(P1)，患者短程连接降低(P2)，精神分裂整体连接减弱(P3)，故该感觉运动网络内边降低。":
        "Transdiagnostic dynamic FC shows reduced somatomotor connectivity (P1), patients show reduced short-range connectivity (P2), and global connectivity is reduced in schizophrenia (P3); hence this within-somatomotor-network edge is hypothesized to decrease.",
    "P1为动态FC、P2为距离依赖、P3为单研究且仅精神分裂，均非边级静态r；转移至混合TCP不确定。":
        "P1 is dynamic FC, P2 is distance-dependent and P3 a single schizophrenia-only study; none is a static edge-level r, so transfer to the mixed TCP is uncertain.",
    "采纳三方评审：补记[已移除引文]→P3替换（P3为精神分裂全局拓扑证据、非边级静态r），端点与-1方向不变。":
        "Adopted the three reviewers' comments: recorded the [citation removed]→P3 replacement (P3 is schizophrenia global-topology evidence, not a static edge-level r); the endpoint and -1 direction are unchanged.",
    "采纳三方评审：仅修订反馈动作，补记[已移除引文]→P3替换；端点、命名、方向与转移限制不变。":
        "Adopted the three reviewers' comments: only the feedback action was revised to record the [citation removed]→P3 replacement; endpoint, naming, direction and transfer limits are unchanged.",
    # packet-32
    "双相II抑郁默认网络连接降低（P1），精神分裂症默认网络异常（P2，方向未定），双相抑郁DMN连接降低（P3），提示该默认网络边在病例中可能降低。":
        "Default-network connectivity is reduced in bipolar II depression (P1), default-network abnormalities exist in schizophrenia (P2, direction undetermined), and DMN connectivity is reduced in bipolar depression (P3), suggesting this default-network edge may decrease in patients.",
    "P1为双相II抑郁单研究，P2为综述未给方向，P3为双相抑郁meta分析；均非该特定ROI-网络边，方向不确定。":
        "P1 is a single bipolar-II-depression study, P2 a review without direction, and P3 a bipolar-depression meta-analysis; none is this specific ROI-to-network edge, so the direction is uncertain.",
    "依methodology_expert/pass1，[已移除引文]不含默认网络，已移除并替换为默认网络证据P3，同步改写理由与限制。":
        "Per methodology_expert/pass1, [citation removed] does not cover the default network; it was removed and replaced with default-network evidence P3, and the rationale and limits were rewritten accordingly.",
    "三方一致指出[已移除引文]不含默认网络，属证据-端点错配；采纳methodology_expert/pass1要求，移除[已移除引文]并替换为默认网络证据P3，同步改写理由与转移限制。":
        "All three reviewers noted that [citation removed] does not cover the default network — an evidence–endpoint mismatch. As required by methodology_expert/pass1, [citation removed] was removed and replaced with default-network evidence P3, and the rationale and transfer limits were rewritten accordingly.",
    # packet-33
    "注意网络同伦连接降低（P1），精神分裂整体连接强度减弱（P2），跨诊断感觉运动连接降低（P3），故该感觉运动-注意边在病例中降低。":
        "Homotopic connectivity in attention networks is reduced (P1), global connectivity strength is reduced in schizophrenia (P2), and transdiagnostic somatomotor connectivity is reduced (P3); hence this somatomotor–attention edge is hypothesized to decrease in patients.",
    "P1为同伦指标、P2为全局拓扑，非该具体边；[已移除引文]示SZ额顶连接增高，方向异质；转移不确定。":
        "P1 is a homotopic measure and P2 global topology, not this specific edge; [citation removed] reports increased frontoparietal connectivity in SZ, so directions are heterogeneous and transfer is uncertain.",
    "父节点237:DorsAttn联合稳定性0.98，保留DorsAttn-1方向，改换SomMot-ROI端点。":
        "Parent node 237:DorsAttn has joint stability 0.98; the DorsAttn -1 direction is kept and the endpoint switched to a SomMot ROI.",
    "三位评审pass0与pass1均判keep：端点239:DorsAttn在池内且未重复，-1由P1/P2/P3支持，度量错配已披露，父节点稳定性0.98。":
        "All three reviewers judged keep at pass0 and pass1: endpoint 239:DorsAttn is in the pool and not duplicated, the -1 direction is supported by P1/P2/P3, the metric mismatches are disclosed, and the parent node's stability is 0.98.",
    # packet-34
    "KG证据显示感觉运动失连接横跨多维度，成人ADHD与精神分裂症感觉运动连接降低；据此假设该感觉运动网络内连接降低。":
        "KG evidence shows somatomotor disconnection across multiple dimensions, with reduced somatomotor connectivity in adult ADHD and schizophrenia; on this basis, reduced connectivity within this somatomotor network is hypothesized.",
    "P1无方向，P4报告ADHD感觉运动连接增高与本边-1相反；P2仅成人ADHD；未直接检验该网络内边。":
        "P1 has no direction; P4 reports increased ADHD somatomotor connectivity, opposite to this edge's -1; P2 covers only adult ADHD; the within-network edge has not been directly tested.",
    "依methodology_expert/pass1与biostatistician/pass1要求，改换为感觉运动网络对端点以扩展主题，并披露P1无方向、P4相反方向与P2成人样本。":
        "As required by methodology_expert/pass1 and biostatistician/pass1, the endpoint was switched to a somatomotor-network target to broaden coverage, disclosing that P1 has no direction, P4 shows the opposite direction, and P2 is an adult sample.",
    "采纳methodology_expert/pass1与clinical_neuroscientist/pass1意见：改换为感觉运动网络对端点以扩展主题，并披露P1无方向、P4相反方向与P2成人样本。":
        "Adopted the comments of methodology_expert/pass1 and clinical_neuroscientist/pass1: the endpoint was switched to a somatomotor-network target to broaden coverage, disclosing that P1 has no direction, P4 shows the opposite direction, and P2 is an adult sample.",
    # shared v6 strings
    "完整原始提议、六次模型评议、预检验锁定和外部行均保留来源绑定；该假设来自首轮提议，生成时同 seed 尚无已完成 TCP 反馈，无父假设。此页是汇总展示，不是重新运行实验。":
        "The complete original proposal, six model reviews, the pre-test lock and the external rows all retain source bindings; this hypothesis came from a first-round proposal, when no completed TCP feedback existed for the same seed, so there is no parent hypothesis. This page is a summary display, not a re-run of the experiment.",
    "34 份已完成且外部验证通过的研究材料能力评分；按 10 名专家 × 每人 10 份 × 每份 3 人设计；尚未完成正式伦理与招募安排。":
        "Capability ratings for 34 completed and externally validated research materials; designed as 10 experts × 10 materials each × 3 reviewers per material; formal ethics and recruitment arrangements are not yet complete.",
    "34 份材料全部来自已完成的发现臂与扩充臂及其两阶段外部分析。两批中外部整理条件全部通过的假设共 38 条，本面板按最小方向标准化效应降序取前 34 条（并列按联合方向频率），最弱 4 条未纳入；其中 3 份沿用上一版材料，31 份按相同版式新建。面板按外部效应强弱排序，不是代表性抽样，不能用来估计自然发现成功率。":
        "All 34 materials come from the completed discovery and expansion arms and their two-stage external analyses. Across the two batches, 38 hypotheses passed all external triage conditions; this panel takes the top 34 by minimum directional standardized effect (ties broken by joint direction frequency), leaving out the weakest 4. Three materials are carried over from the previous version and 31 were newly built in the same format. The panel is ordered by external effect strength; it is not a representative sample and cannot be used to estimate the natural discovery success rate.",
    "两批已完成发现活动外部整理条件全部通过的假设共 38 条，按最小方向标准化效应降序取前 34 条（并列按联合方向频率），最弱 4 条未纳入；面板按外部效应强弱排序，不能用来估计自然发现成功率。":
        "Across the two completed discovery campaigns, 38 hypotheses passed all external triage conditions; the top 34 were taken by minimum directional standardized effect (ties broken by joint direction frequency), leaving out the weakest 4. The panel is ordered by external effect strength and cannot be used to estimate the natural discovery success rate.",
    "每份 6 题；按分配表每位专家评审 10 份、共 60 题；可暂停，实际耗时待试评测定":
        "6 questions per material; each expert reviews 10 assigned materials (60 questions total); pausing is allowed, actual time to be measured in piloting.",
    "v6 · 2026-09-18 · 单轮": "v6 · 2026-09-18 · Single round",
}


def load_checked(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {Path(path).name}")
    return json.loads(raw)


def records_of(doc):
    return doc if isinstance(doc, list) else doc["records"]


def build():
    pack_raw = PACK.read_bytes()
    pack = json.loads(pack_raw)
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    strings = dict(previous["strings"])

    counts = load_checked(SOURCES / "old_r1/candidate_index.json",
                          SOURCE_HASHES["old_r1/candidate_index.json"])["counts"]["network_parcel_counts"]
    records = {}
    for prefix in ("old_r1", "r7"):
        for record in records_of(load_checked(SOURCES / f"{prefix}/candidate_index.json", SOURCE_HASHES[f"{prefix}/candidate_index.json"])):
            records[record["canonical_hypothesis_id"]] = record

    generated = {}
    mapping = {entry["id"]: entry["hypothesis_id"] for entry in pack["organizer"]["mapping"]}
    for card in pack["cards"]:
        hypothesis_id = mapping[card["id"]]
        record = records[hypothesis_id]
        template_zh = [card["pre"]["title"], card["pre"]["hypothesis"],
                       card["pre"]["hypothesis_plain"], card["pre"]["definition"]]
        if all(zh in strings for zh in template_zh):
            continue  # Reused v5 cards are already fully translated; leave them untouched.
        title, definition = describe(record, counts)
        plain = plain_hypothesis(record)
        assert title == card["pre"]["title"], card["id"]
        assert definition == card["pre"]["definition"], card["id"]
        assert plain == card["pre"]["hypothesis_plain"], card["id"]
        assert record["canonical_record"]["hypothesis_zh"] == card["pre"]["hypothesis"], card["id"]
        for zh, en in ((title, en_title(record)), (card["pre"]["hypothesis"], en_hypothesis(record)),
                       (plain, en_plain(record)), (definition, en_definition(record, counts))):
            if zh not in strings:
                generated.setdefault(zh, en)

    additions = {**generated}
    projection = {"meta": pack["public_meta"], "questions": pack["questions"], "issues": pack["issues"],
                  "common_pre": pack["common_pre"], "common_post": pack["common_post"], "cards": pack["cards"]}
    needed = {text for text in strings_in(projection) if CJK.search(text)}
    for zh, en in FREETEXT.items():
        if zh in needed and zh not in strings:
            additions[zh] = en
    strings.update(additions)

    missing = sorted(text for text in needed if text not in strings)
    if missing:
        raise ValueError("Untranslated strings remain: " + json.dumps(missing[:5], ensure_ascii=False))
    for key, value in strings.items():
        if CJK.search(value):
            raise ValueError(f"Translation still contains CJK: {key[:40]}")
    unused = sorted(zh for zh in FREETEXT if zh not in needed and zh not in previous["strings"])
    if unused:
        raise ValueError("FREETEXT entries not present in the pack: " + json.dumps(unused[:5], ensure_ascii=False))

    return {
        "version": "cs1-discovery-display-en-v3",
        "source_pack": PACK.name,
        "source_sha256": hashlib.sha256(pack_raw).hexdigest(),
        "covers_packs": [PREVIOUS.name.replace("en_v2", "pilot_v5"), PACK.name],
        "strings": strings,
    }


def main():
    catalog = build()
    if TARGET.exists():
        if json.loads(TARGET.read_text(encoding="utf-8")) != catalog:
            raise ValueError("en_v3 already exists with different content; create a new version.")
    else:
        with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(catalog, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"version": catalog["version"], "strings": len(catalog["strings"]),
                      "source_sha256": catalog["source_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
