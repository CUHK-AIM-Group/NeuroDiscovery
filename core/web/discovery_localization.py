"""Read-only, versioned display translations; never translate scientific data.

Only strings already present in the public projection are sent to the browser.
In particular, loading config or a historical stage A cannot reveal later results.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

CATALOG = Path(__file__).with_name("study_materials") / "cs1_discovery_en_v15.json"


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


@lru_cache(maxsize=1)
def catalog():
    raw = CATALOG.read_bytes()
    data = json.loads(raw)
    return data, hashlib.sha256(raw).hexdigest()


def presentation(public):
    data, checksum = catalog()
    # Session answers and free-text notes must remain exactly as the author wrote
    # them and must never be treated as translation keys or sent for translation.
    fields = {key: value for key, value in public.items()
              if key in {"meta", "questions", "issues", "issue_options", "common_pre", "common_post", "cards"}}
    visible = set(strings_in(fields))
    return {"version": data["version"], "sha256": checksum, "languages": ["zh", "en"],
            "strings": {key: value for key, value in data["strings"].items() if key in visible}}
