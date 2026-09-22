"""Add plain-language, dataset-free hypothesis statements in a new source-preserving version.

The owner asked for two things (2026-09-18):
1. The "what is this discovery" statement must not be tied to a dataset — the
   machine string mentioned the TCP sample and preregistration details.
2. Every hypothesis must be readable in plain language instead of the raw
   program-rendered `type:a:b` title.

Nothing else changes: the original `hypothesis` strings, evidence, questions,
anchors and scoring rules stay exactly as in v4. The dataset and analysis
details remain documented in `common_pre` ("人群与分析").
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PARENT = HERE / "cs1_discovery_pilot_v4.json"
TARGET = HERE / "cs1_discovery_pilot_v5.json"
PARENT_HASH = "c99f1765154ad2eec1e0a82464ff8c2c1d261bfd681c4275eaabf728beb288f4"

# Plain-language rewrite per card id, keyed to the immutable card order in v4.
# Each statement keeps the exact scientific claim (edge set, diagnosis groups,
# separate-vs-own-control comparison, same-direction decrease) and drops the
# dataset/preregistration framing, which stays in common_pre.
PLAIN_HYPOTHESES = {
    "packet-01": "左侧 ROI108 分区与躯体运动网络（SomMot）之间的功能连接强度，在双相障碍患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-02": "左侧 ROI098 分区与躯体运动网络（SomMot）之间的功能连接强度，在双相障碍患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-03": "边缘网络（Limbic）与躯体运动网络（SomMot）之间的功能连接强度，在 ADHD 患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-04": "背侧注意网络（DorsAttn）与视觉网络（Vis）之间的功能连接强度，在 ADHD、双相障碍、精神分裂症/分裂情感性障碍三组患者中预计均低于各自的对照组——三组下降方向一致。",
    "packet-05": "右侧 ROI314 分区与躯体运动网络（SomMot）之间的功能连接强度，在 ADHD、双相障碍、精神分裂症/分裂情感性障碍三组患者中预计均低于各自的对照组——三组下降方向一致。",
    "packet-06": "左侧 ROI104 分区与躯体运动网络（SomMot）之间的功能连接强度，在 ADHD 患者和双相障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-07": "右侧 ROI254 分区与显著性/腹侧注意网络（SalVentAttn）之间的功能连接强度，在双相障碍患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-08": "背侧注意网络（DorsAttn）与显著性/腹侧注意网络（SalVentAttn）之间的功能连接强度，在双相障碍患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
    "packet-09": "显著性/腹侧注意网络（SalVentAttn）内部各分区之间的功能连接强度，在 ADHD、双相障碍、精神分裂症/分裂情感性障碍三组患者中预计均低于各自的对照组——三组下降方向一致。",
    "packet-10": "边缘网络（Limbic）与视觉网络（Vis）之间的功能连接强度，在双相障碍患者和精神分裂症/分裂情感性障碍患者中预计均低于各自的对照组——两组下降方向一致。",
}


def build(parent=PARENT):
    raw = Path(parent).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_HASH:
        raise ValueError("Unexpected v4 pack; do not silently rebase.")
    pack = copy.deepcopy(json.loads(raw))
    pack["pack_id"] = "cs1-discovery-capabilities-20260918-v5"
    pack["protocol_version"] = "cs1-complete-output-single-round-v5"
    pack["public_meta"]["version_label"] = "v5 · 2026-09-18 · 单轮"
    card_ids = [card["id"] for card in pack["cards"]]
    if card_ids != list(PLAIN_HYPOTHESES):
        raise ValueError("Card set or order changed; rewrite plain statements deliberately.")
    for card in pack["cards"]:
        card["pre"]["hypothesis_plain"] = PLAIN_HYPOTHESES[card["id"]]
    pack["organizer"]["plain_language_amendment"] = {
        "parent_v4_sha256": PARENT_HASH,
        "change": "Added pre.hypothesis_plain: plain-language, dataset-free restatement of every hypothesis; the original program-rendered hypothesis string is kept unchanged.",
        "unchanged": "All case evidence, question wording, options, anchors, scoring rules, references and the original hypothesis strings.",
        "dataset_boundary": "Dataset and analysis details stay in common_pre; the finding statement no longer names the sample.",
    }
    return pack


if __name__ == "__main__":
    pack = build()
    if TARGET.exists():
        if json.loads(TARGET.read_text(encoding="utf-8")) != pack:
            raise ValueError("v5 already exists with different content; create a new version.")
    else:
        with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(pack, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(TARGET.resolve())
