"""Durable KG request capture, row isolation and offline resume.

No network or graph writer is imported. A transport records its start before
sending, then saves the entire response before interpreting model content.
Reopening or replaying this store never sends a request or retries a timeout.
"""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json
import sqlite3

from neurooracle.src import kg_claim_compact_protocol as compact
from neurooracle.src import kg_claim_support_contract as contract

VERSION = "kg.claim_batch_store.v2"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def _usage(envelope, spec):
    """Count provider usage independently of parsing or semantic validity."""
    unknown = {"input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
               "usage_known": False, "cost_lower_usd": 0.0, "cost_upper_usd": spec["reserved_usd"]}
    if not isinstance(envelope, dict):
        return unknown
    p, o = envelope.get("prompt_eval_count"), envelope.get("eval_count")
    if not all(type(n) is int and n >= 0 for n in (p, o)):
        return unknown
    cached = envelope.get("cached_prompt_eval_count")
    if cached is not None and not (type(cached) is int and 0 <= cached <= p):
        return unknown
    rates = {key: Decimal(str(v)) for key, v in spec["rates_per_million"].items()}
    million = Decimal(1000000)
    if cached is None:
        low = (p * rates["cached_input"] + o * rates["output"]) / million
        high = (p * rates["input"] + o * rates["output"]) / million
    else:
        low = high = ((p-cached) * rates["input"] + cached * rates["cached_input"] + o * rates["output"]) / million
    return {"input_tokens": p, "output_tokens": o, "cached_input_tokens": cached,
            "usage_known": True, "cost_lower_usd": float(low), "cost_upper_usd": float(high)}


class BatchStore:
    def __init__(self, path, *, policy):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value NUMERIC NOT NULL);
          CREATE TABLE IF NOT EXISTS papers (sha TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS contracts (sha TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, spec TEXT NOT NULL,
            state TEXT NOT NULL, started_at TEXT, finished_at TEXT, raw_response TEXT,
            response_sha TEXT, accounting TEXT, captured TEXT, transport_error TEXT);
          CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, paper_sha TEXT NOT NULL REFERENCES papers(sha),
            contract_sha TEXT NOT NULL REFERENCES contracts(sha), request_id TEXT REFERENCES requests(id),
            state TEXT NOT NULL, result TEXT, error TEXT);
          CREATE INDEX IF NOT EXISTS cases_request ON cases(request_id);
          CREATE INDEX IF NOT EXISTS cases_pending_paper ON cases(request_id,paper_sha,id);
          CREATE INDEX IF NOT EXISTS requests_state ON requests(state);
        """)
        value = encode({"version": VERSION, "policy": policy})
        old = self.db.execute("SELECT value FROM meta WHERE key='binding'").fetchone()
        try:
            compact.require(old is None or old[0] == value, "store policy changed")
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO meta VALUES('binding',?)", (value,))
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, rid):
        row = self.db.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
        compact.require(row is not None, "unknown request")
        return dict(row)

    def _bump(self, **values):
        # Called only within the same transaction as the corresponding data
        # mutation. Ordinary status reads never scan request/response payloads.
        self.db.executemany("INSERT INTO counters VALUES(?,?) ON CONFLICT(name) "
                            "DO UPDATE SET value=counters.value+excluded.value",
                            [(key, value) for key, value in values.items() if value])

    def _state_change(self, old, new):
        if old != new:
            self._bump(**{"case_state:" + old: -1, "case_state:" + new: 1})

    def _usage_change(self, before, after):
        before = before or {}
        values = {k: (after.get(k) or 0) - (before.get(k) or 0)
                  for k in ("input_tokens", "output_tokens", "cost_lower_usd", "cost_upper_usd")}
        values["requests_with_unknown_usage"] = int(not after["usage_known"]) - int(bool(before) and not before["usage_known"])
        self._bump(**values)

    def stage_cases(self, cases, contracts):
        """Spool unordered cached cases once; source text is stored once on disk."""
        with self.db:
            for case in cases:
                cid = case["case_id"]
                con = contracts[cid]
                # Retain the same strict input checks as normal request planning.
                compact.unpack(compact.prepare([case], {cid: con}))
                source_sha, contract_sha = contract.digest(case["source"]), contract.digest(con)
                old = self.db.execute("SELECT paper_sha,contract_sha FROM cases WHERE id=?", (cid,)).fetchone()
                if old:
                    compact.require(tuple(old) == (source_sha, contract_sha), "staged case changed")
                    continue
                self._bump(papers=self.db.execute("INSERT OR IGNORE INTO papers VALUES(?,?)", (source_sha, encode(case["source"]))).rowcount)
                self._bump(contracts=self.db.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (contract_sha, encode(con))).rowcount)
                self.db.execute("INSERT INTO cases VALUES(?,?,?,NULL,'PENDING',NULL,NULL)", (cid, source_sha, contract_sha))
                self._bump(requested_cases=1, unassigned_cases=1, **{"case_state:PENDING": 1})

    def staged_batch(self, *, papers_per_request=5, cases_per_request=100):
        compact.require(type(papers_per_request) is int and papers_per_request > 0
                        and type(cases_per_request) is int and cases_per_request > 0, "batch limits")
        papers = [r[0] for r in self.db.execute("SELECT paper_sha FROM cases WHERE request_id IS NULL "
            "GROUP BY paper_sha ORDER BY paper_sha LIMIT ?", (papers_per_request,))]
        if not papers:
            return [], {}
        marks = ",".join("?" for _ in papers)
        selected = self.db.execute("SELECT c.id,p.payload AS source,k.payload AS claim_contract FROM cases c "
            "JOIN papers p ON p.sha=c.paper_sha JOIN contracts k ON k.sha=c.contract_sha "
            f"WHERE c.request_id IS NULL AND c.paper_sha IN ({marks}) ORDER BY c.paper_sha,c.id LIMIT ?",
            (*papers, cases_per_request)).fetchall()
        cases, contracts = [], {}
        for row in selected:
            con = json.loads(row["claim_contract"])
            cases.append({"case_id": row["id"], "source": json.loads(row["source"]), "proposition": con["proposition"]})
            contracts[row["id"]] = con
        return cases, contracts

    def prepare_request(self, cases, contracts, *, body, model, input_format=compact.VERSION,
                        origin="local_test", reserved_usd=0, rates_per_million=None, provenance=None):
        compact.require(origin in {"historical_replay", "local_test", "new_request"}, "request origin")
        compact.require(input_format in {compact.VERSION, "legacy_v1"}, "input format")
        compact.require(isinstance(model, str) and bool(model), "model identity")
        compact.require(isinstance(body, dict) and body.get("model") == model, "request model binding")
        reservation = Decimal(str(reserved_usd))
        compact.require(reservation.is_finite() and reservation >= 0, "invalid cost reservation")
        rates = rates_per_million or {"input": 0, "cached_input": 0, "output": 0}
        compact.require(set(rates) == {"input", "cached_input", "output"}, "price fields")
        compact.require(all(Decimal(str(v)).is_finite() and Decimal(str(v)) >= 0 for v in rates.values()), "invalid price")
        compact.require(Decimal(str(rates["cached_input"])) <= Decimal(str(rates["input"])), "cached price exceeds input")
        packed = compact.prepare(cases, contracts)
        restored, restored_contracts = compact.unpack(packed)
        compact.require(bool(restored), "empty request")
        messages = body.get("messages", [])
        compact.require(len(messages) == 2 and messages[0].get("role") == "system"
                        and messages[1].get("role") == "user", "request message binding")
        wire = compact.strict_json(messages[1]["content"])
        if input_format == compact.VERSION:
            wire_cases, wire_contracts = compact.unpack(wire)
            compact.require(messages[0]["content"] == compact.PROMPT, "compact prompt changed")
            compact.require(wire_cases == restored and wire_contracts == restored_contracts, "request inputs differ from cases")
        else:
            wire_cases = {}
            for paper in wire["papers"]:
                for comparison in paper["comparisons"]:
                    cid = comparison["case_id"]
                    compact.require(cid in restored and cid not in wire_cases, "legacy request case binding")
                    con = restored_contracts[cid]
                    compact.require(comparison["contract_id"] == contract.digest(con), "legacy contract binding")
                    target = {"proposition": con["proposition"], "requirements": {
                        d: v["target"] for d, v in con["dimensions"].items() if v["requirement"] == "asserted"}}
                    compact.require(wire["contracts"][comparison["contract_id"]] == target, "legacy request target differs")
                    wire_cases[cid] = {"case_id": cid, "source": paper["source"], "proposition": con["proposition"]}
            compact.require(wire_cases == restored, "legacy request source differs")
        spec = {"version": VERSION, "input_format": input_format, "model": model,
                "origin": origin, "body": body, "case_bindings": {
                cid: {"source": contract.digest(c["source"]), "contract": contract.digest(restored_contracts[cid])}
                for cid, c in restored.items()}, "reserved_usd": float(reservation),
                "rates_per_million": rates, "provenance": provenance or {}}
        spec_text = encode(spec)
        rid = contract.digest(spec)
        old = self.db.execute("SELECT spec FROM requests WHERE id=?", (rid,)).fetchone()
        if old:
            compact.require(old[0] == spec_text, "request digest collision")
            return rid
        with self.db:
            self.db.execute("INSERT INTO requests(id,spec,state) VALUES(?,?,'PREPARED')", (rid, spec_text))
            self._bump(requests_prepared=1)
            for cid, case in restored.items():
                binding = spec["case_bindings"][cid]
                self._bump(papers=self.db.execute("INSERT OR IGNORE INTO papers VALUES(?,?)", (binding["source"], encode(case["source"]))).rowcount)
                self._bump(contracts=self.db.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (binding["contract"], encode(restored_contracts[cid]))).rowcount)
                old_case = self.db.execute("SELECT paper_sha,contract_sha,request_id FROM cases WHERE id=?", (cid,)).fetchone()
                if old_case:
                    compact.require(tuple(old_case) == (binding["source"], binding["contract"], None),
                                    "case already assigned or input changed")
                    self.db.execute("UPDATE cases SET request_id=? WHERE id=?", (rid, cid))
                    self._bump(unassigned_cases=-1)
                else:
                    self.db.execute("INSERT INTO cases VALUES(?,?,?,?,'PENDING',NULL,NULL)",
                                    (cid, binding["source"], binding["contract"], rid))
                    self._bump(requested_cases=1, **{"case_state:PENDING": 1})
        return rid

    def start(self, rid, *, started_at=None):
        """The caller must commit this before transport; duplicate starts fail."""
        spec = json.loads(self.request(rid)["spec"])
        initial_usage = _usage(None, spec)
        with self.db:
            changed = self.db.execute("UPDATE requests SET state='IN_FLIGHT', started_at=?, accounting=? "
                                     "WHERE id=? AND state='PREPARED'",
                                     (started_at or now(), encode(initial_usage), rid)).rowcount
            compact.require(changed == 1, "request already started; no automatic retry")
            self._bump(request_attempts_recorded=1, new_model_requests_recorded=int(spec["origin"] == "new_request"),
                       historical_attempts_replayed=int(spec["origin"] == "historical_replay"))
            self._usage_change(None, initial_usage)

    def record_response(self, rid, envelope):
        """Durably save usage and raw provider data before any row validation."""
        text = encode(envelope)
        sha = contract.digest(envelope)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            request = self.request(rid)
            if request["raw_response"] is not None:
                compact.require(request["response_sha"] == sha and request["raw_response"] == text,
                                "conflicting response replay")
                return
            compact.require(request["state"] in {"IN_FLIGHT", "TRANSPORT_UNKNOWN"}, "request not in flight")
            spec = json.loads(request["spec"])
            usage = _usage(envelope, spec)
            self.db.execute("UPDATE requests SET raw_response=?,response_sha=?,accounting=?,state='RECEIVED',finished_at=? WHERE id=?",
                            (text, sha, encode(usage), now(), rid))
            self._bump(responses_saved=1)
            self._usage_change(json.loads(request["accounting"]), usage)

    def mark_transport_unknown(self, rid, reason):
        compact.require(isinstance(reason, str) and reason.strip(), "transport failure reason")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            request = self.request(rid)
            if request["state"] == "TRANSPORT_UNKNOWN":
                compact.require(request["transport_error"] == reason, "changed transport failure")
                return
            compact.require(request["state"] == "IN_FLIGHT" and request["raw_response"] is None, "response already captured")
            old_states = dict(self.db.execute("SELECT state,count(*) FROM cases WHERE request_id=? GROUP BY state", (rid,)).fetchall())
            for state, count in old_states.items():
                self._bump(**{"case_state:" + state: -count})
            self._bump(**{"case_state:TRANSPORT_UNKNOWN": sum(old_states.values())})
            self.db.execute("UPDATE requests SET state='TRANSPORT_UNKNOWN',finished_at=?,transport_error=? WHERE id=?", (now(), reason, rid))
            self.db.execute("UPDATE cases SET state='TRANSPORT_UNKNOWN',error=? WHERE request_id=?", (reason, rid))

    def cases(self, rid=None):
        sql = ("SELECT c.*,p.payload AS source,k.payload AS claim_contract FROM cases c "
               "JOIN papers p ON p.sha=c.paper_sha JOIN contracts k ON k.sha=c.contract_sha")
        args = ()
        if rid is not None:
            sql += " WHERE c.request_id=?"
            args = (rid,)
        for row in self.db.execute(sql + " ORDER BY c.id", args):
            source, claim = json.loads(row["source"]), json.loads(row["claim_contract"])
            compact.require(contract.digest(source) == row["paper_sha"] and contract.digest(claim) == row["contract_sha"],
                            "cached case input changed")
            yield {"case_id": row["id"], "source": source, "proposition": claim["proposition"]}, claim

    def reconcile(self, rid):
        """Finish a received response after restart, without another model call."""
        request = self.request(rid)
        if request["state"] == "CAPTURED":
            return json.loads(request["captured"])
        compact.require(request["state"] == "RECEIVED", "no saved response to reconcile")
        spec, envelope = json.loads(request["spec"]), json.loads(request["raw_response"])
        compact.require(contract.digest(envelope) == request["response_sha"], "saved response changed")
        cases, contracts = {}, {}
        for case, con in self.cases(rid):
            cid = case["case_id"]
            compact.require(spec["case_bindings"][cid] == {"source": contract.digest(case["source"]), "contract": contract.digest(con)}, "request input binding changed")
            cases[cid], contracts[cid] = case, con
        try:
            compact.require(isinstance(envelope, dict) and envelope.get("model") == spec["model"], "response model mismatch")
            compact.require(envelope.get("done") is True and envelope.get("done_reason") == "stop", "incomplete provider response")
            content = envelope["message"]["content"]
            compact.require(isinstance(content, str), "response content")
            captured = compact.capture(content, cases, contracts, input_format=spec["input_format"])
        except (ValueError, KeyError, TypeError) as exc:
            captured = {"valid": [], "errors": [{"row": None, "case_id": None, "reason": str(exc)}],
                        "missing": sorted(cases), "returned_rows": 0, "response_parsed": False}
        valid = {r["case_id"]: r for r in captured["valid"]}
        errors = {r["case_id"]: r for r in captured["errors"] if r["case_id"] in cases}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.request(rid)
            compact.require(current["response_sha"] == request["response_sha"] and current["state"] in {"RECEIVED", "CAPTURED"}, "response state changed")
            if current["state"] == "CAPTURED":
                return json.loads(current["captured"])
            old_states = dict(self.db.execute("SELECT id,state FROM cases WHERE request_id=?", (rid,)).fetchall())
            for cid in cases:
                if cid in valid:
                    state, result, error = "CAPTURED", encode(valid[cid]), None
                elif cid in errors:
                    state, result, error = "INVALID", None, encode(errors[cid])
                else:
                    state, result, error = "MISSING", None, "No returned row"
                self._state_change(old_states[cid], state)
                if cid in valid:
                    row = valid[cid]
                    self._bump(full_support_suggestions=int(row["full_support_suggestion"]),
                               support_review_queue=int(row["review_priority"] == "support"),
                               legacy_original_validation_failures_preserved=int(row["legacy"] is not None and not row["legacy"]["passes_original_validator"]))
                self.db.execute("UPDATE cases SET state=?,result=?,error=? WHERE id=?", (state, result, error, cid))
            self.db.execute("UPDATE requests SET state='CAPTURED',captured=? WHERE id=?", (encode(captured), rid))
        return captured

    def resume_saved(self):
        ids = [r[0] for r in self.db.execute("SELECT id FROM requests WHERE state='RECEIVED' ORDER BY id")]
        return {rid: self.reconcile(rid) for rid in ids}

    def results(self):
        for row in self.db.execute("SELECT result FROM cases WHERE result IS NOT NULL ORDER BY id"):
            yield json.loads(row[0])

    def accounting(self):
        result = []
        for row in self.db.execute("SELECT id,state,started_at,accounting,raw_response IS NOT NULL AS response_saved,"
                "json_extract(spec,'$.model') AS model,json_extract(spec,'$.origin') AS origin,"
                "json_extract(spec,'$.provenance') AS provenance FROM requests ORDER BY started_at,id"):
            result.append({"request_id": row["id"], "state": row["state"], "model": row["model"],
                           "origin": row["origin"], "started_at": row["started_at"],
                           "response_saved": bool(row["response_saved"]),
                           "provenance": json.loads(row["provenance"]),
                           "usage": json.loads(row["accounting"]) if row["accounting"] else None})
        return result

    def summary(self):
        counters = dict(self.db.execute("SELECT name,value FROM counters").fetchall())
        count_names = ("requested_cases", "papers", "contracts", "requests_prepared", "request_attempts_recorded",
                       "unassigned_cases", "new_model_requests_recorded", "historical_attempts_replayed",
                       "responses_saved", "requests_with_unknown_usage", "input_tokens", "output_tokens",
                       "full_support_suggestions", "support_review_queue", "legacy_original_validation_failures_preserved")
        result = {key: int(counters.get(key, 0)) for key in count_names}
        result.update(version=VERSION, scientific_approvals=0, production_writes=0,
                      case_states={key.removeprefix("case_state:"): int(value)
                                   for key, value in counters.items() if key.startswith("case_state:") and value})
        result.update({key: round(float(counters.get(key, 0)), 12) for key in ("cost_lower_usd", "cost_upper_usd")})
        return result
