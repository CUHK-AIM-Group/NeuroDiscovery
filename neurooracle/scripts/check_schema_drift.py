"""Read-only schema/prompt drift report. Sends nothing, changes nothing.

Reports, for every prompt string that is reachable in the codebase, which schema
fields it omits and which required keys it never states. This is the evidence
required before any embedded shape text is replaced by rendered schema text.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neurooracle/scripts"))
from neurooracle.src import prompt_schema

CODE_FILES = (
    "neurooracle/schemas/candidate.schema.json",
    "neurooracle/src/prompt_schema.py",
    "neurooracle/scripts/check_schema_drift.py",
    "neurooracle/tests/test_prompt_schema.py",
)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_prompts():
    """Every prompt string that code can actually reach, with its status."""
    import paper_batch_model
    import paper_extract_v2
    import paper_pilot_batch
    from neurooracle.src.paper_extraction_contract_v3 import EXTRACTION_AMENDMENT_V3, build_prompt

    return [
        {"name": "paper_batch_model.SYSTEM", "status": "live_legacy_route",
         "used_by": "paper_batch_model.send_one / run",
         "text": paper_batch_model.SYSTEM},
        {"name": "paper_pilot_batch.SYSTEM", "status": "retired_superseded",
         "used_by": "nothing, once paper_extract_v2.apply_prompt runs",
         "text": paper_pilot_batch.SYSTEM},
        {"name": "paper_extract_v2.EXTRACTION_SYSTEM_V2", "status": "live_base",
         "used_by": "paper_extract_v2.apply_prompt",
         "text": paper_extract_v2.EXTRACTION_SYSTEM_V2},
        {"name": "paper_extraction_contract_v3.EXTRACTION_AMENDMENT_V3", "status": "live_amendment",
         "used_by": "appended by build_prompt", "text": EXTRACTION_AMENDMENT_V3},
        {"name": "effective extraction prompt (base + amendment)", "status": "live_effective",
         "used_by": "what a model actually receives today",
         "text": build_prompt(paper_extract_v2.EXTRACTION_SYSTEM_V2)},
    ]


def replacement_delta():
    """What would change if the amendment's hand-written block were rendered.

    The plan requires this to be reported and accepted before any embedded shape
    text is replaced. Two effects are measured here, both read-only:

    * the rendered block restates definitions the base prompt already states, so
      a naive ``replace`` duplicates them rather than removing anything;
    * the effective prompt's sha therefore changes, which invalidates the frozen
      prompt binding recorded for the completed cohorts.
    """
    import paper_extract_v2
    from neurooracle.src.paper_extraction_contract_v3 import EXTRACTION_AMENDMENT_V3, build_prompt

    schema = prompt_schema.load_schema()
    base = paper_extract_v2.EXTRACTION_SYSTEM_V2
    old_effective = build_prompt(base)
    base_block = prompt_schema.shape_block(base, "JSON shape") or {}
    rendered = prompt_schema.render_shape(schema)
    rendered_definitions = [line.split(":")[0] for line in rendered.splitlines()[2:]]
    stated_in_base = [name for name in schema["definitions"] if name in base_block]
    new_effective = old_effective.replace(
        EXTRACTION_AMENDMENT_V3[EXTRACTION_AMENDMENT_V3.index(prompt_schema.SHAPE_MARKER):],
        prompt_schema.SHAPE_MARKER + "\n" + rendered + "\n")
    return {
        "hand_written_block_sha256": sha(
            EXTRACTION_AMENDMENT_V3[EXTRACTION_AMENDMENT_V3.index(prompt_schema.SHAPE_MARKER):]),
        "rendered_shape_sha256": prompt_schema.shape_sha(schema),
        "rendered_states_definitions": rendered_definitions,
        "definitions_already_stated_by_base_prompt": sorted(stated_in_base),
        "definitions_the_render_would_restate": sorted(set(rendered_definitions) & set(stated_in_base)),
        "old_effective_sha256": sha(old_effective),
        "new_effective_sha256_if_replaced": sha(new_effective),
        "behaviour_change": True,
        "blocked_by": "gate.embedded_shape_text_replaced; replacement changes the text the model receives"
                     " and the frozen prompt sha the completed cohorts are bound to",
    }


def report():
    schema = prompt_schema.load_schema()
    declared = prompt_schema.schema_fields(schema)
    rows = []
    for prompt in collect_prompts():
        rows.append({
            "name": prompt["name"], "status": prompt["status"], "used_by": prompt["used_by"],
            "chars": len(prompt["text"]), "sha256": sha(prompt["text"]),
            "fields_without_json_key": prompt_schema.fields_without_json_key(prompt["text"], schema),
            "quoted_tokens_undefined": prompt_schema.undefined_field_mentions(prompt["text"], schema),
            "drift": prompt_schema.prompt_drift(prompt["text"], schema),
            "embedded_shape_delta": prompt_schema.shape_delta(prompt["text"], schema),
            "envelope_shape_delta": prompt_schema.shape_delta_full(prompt["text"], schema),
        })
    effective = next(row for row in rows if row["status"] == "live_effective")
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "schema_path": str(prompt_schema.SCHEMA_PATH.relative_to(ROOT)),
        "schema_sha256": hashlib.sha256(prompt_schema.SCHEMA_PATH.read_bytes()).hexdigest(),
        "declared_field_count": len(declared),
        "rendered_shape_sha256": prompt_schema.shape_sha(schema),
        "prompts": rows,
        "live_effective_fields_without_json_key": effective["fields_without_json_key"],
        "gate": {
            "embedded_shape_text_replaced": False,
            "reason": "plan requires the delta list to be reported and accepted before any embedded shape text is replaced",
        },
        "replacement_delta": replacement_delta(),
        "note": "fields_without_json_key is strict: a field named only in prose counts as missing, because the model must emit the quoted key.",
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in CODE_FILES},
        "model_calls": 0, "production_writes": 0, "prompt_files_modified": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    for row in result["prompts"]:
        delta = row["embedded_shape_delta"]
        print("%-52s %-22s nokey=%d qundef=%d drift=%d shp=%s" % (
            row["name"], row["status"], len(row["fields_without_json_key"]),
            len(row["quoted_tokens_undefined"]), len(row["drift"]),
            "none" if delta is None else len(delta)))
    print(json.dumps({"live_effective_fields_without_json_key": result["live_effective_fields_without_json_key"],
                      "schema_sha256": result["schema_sha256"][:16]}, indent=2))


if __name__ == "__main__":
    main()
