"""Judge CLM_CONCEPT hypothesis endpoints via GPT-5.4 (5 rounds, majority vote).

Determines which CLM_CONCEPT names are specific enough to serve as
hypothesis endpoints for neuroimaging experiments. Vague/generic names
are flagged for removal.

Usage:
    python -m core.knowledge_graph.src.judge_clm_endpoints \
        --hyp-dir core/knowledge_graph/data/quick \
        --out core/knowledge_graph/data/quick/clm_endpoint_judgments.json \
        [--rounds 5] [--batch-size 40] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx
from openai import OpenAI

log = logging.getLogger("judge_clm_endpoints")

SYSTEM_PROMPT = """\
You are a neuroscience research assistant. Your task is to judge whether
concept names are SPECIFIC enough to serve as endpoints (source or target)
in testable neuroimaging hypotheses.

A GOOD endpoint is a concept that:
- Refers to a specific, measurable biological entity, brain mechanism,
  disease subtype, cognitive construct, or behavioral phenotype
- Can be operationalized in a neuroimaging experiment (fMRI, sMRI, EEG, PET)
- Examples: "mTOR signaling", "BDNF levels", "inhibitory engrams",
  "neuropsychiatric lupus", "social-communicative problems",
  "cortical thickness of visual perceptual areas"

A BAD endpoint is a concept that:
- Is too vague/generic to drive a specific experiment
- Is just an adjective + generic noun (e.g., "motor deficits", "cognitive impairments")
- Is an umbrella term covering many unrelated things
- Cannot be mapped to a specific measurement or label in a dataset
- Examples: "neurocognitive deficits", "clinically significant impairments",
  "functional disability", "balance", "focus", "integration"

For each concept, respond with KEEP or REMOVE and a brief reason (< 15 words).
"""

USER_PROMPT_TEMPLATE = """\
Judge each concept below. For each line, respond with the EXACT format:
<number>. <KEEP|REMOVE> — <reason>

Concepts:
{concepts_block}
"""


def collect_clm_endpoints(hyp_dir: Path) -> list[dict]:
    """Collect unique CLM_CONCEPT endpoint names from hypothesis files."""
    main_files = sorted(
        f for f in hyp_dir.glob("hypotheses_*.json")
        if "pre_" not in f.name and "post_" not in f.name
    )

    endpoints: dict[str, dict] = {}
    for fpath in main_files:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        items = data.get("hypotheses", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for h in items:
            for id_key, name_key in [("source_id", "source_name"), ("target_id", "target_name")]:
                nid = h.get(id_key, "")
                name = h.get(name_key, "")
                if nid.startswith("CLM_CONCEPT"):
                    key = name.lower().strip()
                    if key not in endpoints:
                        endpoints[key] = {"id": nid, "name": name, "count": 0}
                    endpoints[key]["count"] += 1

    return sorted(endpoints.values(), key=lambda x: -x["count"])


def judge_batch(
    client: OpenAI,
    model: str,
    concepts: list[dict],
    round_id: int,
) -> dict[str, str]:
    """Send one batch to GPT and parse results. Returns {name_lower: 'KEEP'|'REMOVE'}."""
    concepts_block = "\n".join(
        f"{i+1}. {c['name']}" for i, c in enumerate(concepts)
    )
    prompt = USER_PROMPT_TEMPLATE.format(concepts_block=concepts_block)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content or ""
            break
        except Exception as e:
            log.warning(f"round {round_id} attempt {attempt+1} failed: {e}")
            time.sleep(5)
            text = ""

    results: dict[str, str] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Parse "1. KEEP — reason" or "1. REMOVE — reason"
        import re
        m = re.match(r"(\d+)\.\s*(KEEP|REMOVE)", line, re.I)
        if m:
            idx = int(m.group(1)) - 1
            verdict = m.group(2).upper()
            if 0 <= idx < len(concepts):
                key = concepts[idx]["name"].lower().strip()
                results[key] = verdict

    return results


def run_judging(
    hyp_dir: Path,
    out_path: Path,
    rounds: int = 5,
    batch_size: int = 40,
    dry_run: bool = False,
):
    endpoints = collect_clm_endpoints(hyp_dir)
    log.info(f"collected {len(endpoints)} unique CLM_CONCEPT endpoints")

    if dry_run:
        log.info("dry-run: would judge these endpoints")
        for e in endpoints[:20]:
            log.info(f"  x{e['count']:3d}: {e['name']}")
        return

    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    keys_raw = os.environ.get("OPENAI_API_KEYS", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            keys = [key]
    if not keys:
        log.error("no API keys found (OPENAI_API_KEYS or OPENAI_API_KEY)")
        return

    model = os.environ.get("OPENAI_MODEL", "gpt-5.5")

    # Round-robin clients
    clients = [
        OpenAI(base_url=base_url, api_key=k, http_client=httpx.Client(verify=False))
        for k in keys
    ]

    # Split into batches
    batches = []
    for i in range(0, len(endpoints), batch_size):
        batches.append(endpoints[i:i + batch_size])

    log.info(f"split into {len(batches)} batches of up to {batch_size}")

    # Run N rounds
    all_votes: dict[str, list[str]] = {
        e["name"].lower().strip(): [] for e in endpoints
    }

    for round_id in range(rounds):
        log.info(f"--- Round {round_id + 1}/{rounds} ---")
        for batch_idx, batch in enumerate(batches):
            client = clients[(round_id * len(batches) + batch_idx) % len(clients)]
            results = judge_batch(client, model, batch, round_id)
            for key, verdict in results.items():
                if key in all_votes:
                    all_votes[key].append(verdict)
            log.info(f"  batch {batch_idx+1}/{len(batches)}: got {len(results)} judgments")
            time.sleep(1)

    # Majority vote
    final: dict[str, dict] = {}
    for e in endpoints:
        key = e["name"].lower().strip()
        votes = all_votes.get(key, [])
        keep_count = sum(1 for v in votes if v == "KEEP")
        remove_count = sum(1 for v in votes if v == "REMOVE")
        verdict = "KEEP" if keep_count >= remove_count else "REMOVE"
        final[key] = {
            "name": e["name"],
            "id": e["id"],
            "count": e["count"],
            "votes_keep": keep_count,
            "votes_remove": remove_count,
            "verdict": verdict,
        }

    keep_list = [v for v in final.values() if v["verdict"] == "KEEP"]
    remove_list = [v for v in final.values() if v["verdict"] == "REMOVE"]

    log.info(f"Final: KEEP={len(keep_list)}, REMOVE={len(remove_list)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "rounds": rounds,
        "total_endpoints": len(endpoints),
        "keep_count": len(keep_list),
        "remove_count": len(remove_list),
        "judgments": final,
        "remove_ids": [v["id"] for v in remove_list],
        "remove_names": [v["name"] for v in remove_list],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_judging(
        hyp_dir=args.hyp_dir,
        out_path=args.out,
        rounds=args.rounds,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
