"""Single door for candidate checks (consolidation step 2).

Before this module, four places decided whether a candidate was acceptable, and
each reported a different shape:

* ``neurooracle/scripts/paper_batch_model.py`` -- literal/structural errors;
* ``neurooracle/src/pilot_validation.py`` -- which of those errors are *our*
  validator gap and must be exempted;
* ``neurooracle/src/pilot_validation_v2.py`` -- source-bound direction and
  significance;
* ``neurooracle/src/pilot_validation_v3.py`` -- the structured result contract.

This module adds no scientific rule. It calls those same implementations behind
one interface and returns four explicit outcomes:

``blocked``        the candidate must not be used;
``advisory``       a reviewer must look, but it is not proven wrong;
``exempted``       a structural error caused by a known validator gap; this was
                   previously invisible, and is now reported;
``not_applicable`` the rule needs a schema version the candidate does not carry.

The ``not_applicable`` outcome is the important boundary: a candidate written
before the structured contract exists is not silently reclassified as failing,
and is not silently upgraded either.

``structural_errors`` is passed in rather than imported, because its owner lives
in ``neurooracle/scripts`` and the pure rules live in ``neurooracle/src``. Passing
it keeps the dependency direction the right way round.
"""
from __future__ import annotations

from collections import namedtuple

from neurooracle.src import pilot_validation as exemptions
from neurooracle.src import pilot_validation_v2 as consistency
from neurooracle.src import pilot_validation_v3 as contract


STRUCTURED_SCHEMA = contract.SCHEMA

Rule = namedtuple("Rule", "name layer severity applies_to")

RULES = (
    Rule("structural_source_binding", "extraction", "block", "any"),
    Rule("validator_limitation_exemption", "extraction", "exempt", "any"),
    Rule("direction_and_significance_source_bound", "extraction", "block", "any"),
    Rule("result_contract_fields", "extraction", "block", "structured"),
    Rule("encoding_diagnostics", "extraction", "advisory", "any"),
)


def has_structured_results(candidate):
    """True when any primary result carries the structured contract.

    ``any`` rather than ``all`` on purpose: a candidate that carries the contract
    on *some* observations is treated as structured, so the ones missing it are
    reported as contract failures. Using ``all`` would let a producer emit the
    contract once and skip the remaining observations to escape validation.
    """
    primary = [item for item in (candidate or {}).get("observations") or []
               if isinstance(item, dict) and item.get("role") == "primary_result"]
    return any(isinstance(item.get("structured_result"), dict) for item in primary)


def _named(reasons):
    return sorted(set(reasons))


def check_candidate(candidate, abstract, structural_errors=()):
    """Return the four-way verdict for one candidate. Never mutates input."""
    structural_errors = list(structural_errors or [])
    findings = {}

    exempted = list(exemptions.validator_limitations(structural_errors, candidate, abstract or ""))
    blocked_structural = [error for error in structural_errors if error not in set(exempted)]
    findings["structural_source_binding"] = {
        "status": "blocked" if blocked_structural else "pass",
        "reasons": _named(blocked_structural),
    }
    findings["validator_limitation_exemption"] = {
        "status": "exempted" if exempted else "none",
        "reasons": _named(exempted),
    }

    consistency_errors = consistency.candidate_consistency_errors(candidate)
    findings["direction_and_significance_source_bound"] = {
        "status": "blocked" if consistency_errors else "pass",
        "reasons": _named("observation_%s:%s" % (item.get("observation_index"), item.get("reason"))
                          for item in consistency_errors),
    }

    if has_structured_results(candidate):
        contract_errors = contract.validate_candidate(candidate, abstract or "")
        findings["result_contract_fields"] = {
            "status": "blocked" if contract_errors else "pass",
            "reasons": _named("%s:%s" % (item.get("observation_index"), item.get("reason"))
                              for item in contract_errors),
        }
    else:
        findings["result_contract_fields"] = {
            "status": "not_applicable",
            "reasons": ["structured_result_absent"],
        }

    risks = contract.candidate_risks(candidate)
    findings["encoding_diagnostics"] = {
        "status": "advisory" if risks else "pass",
        "reasons": _named("%s:%s" % (item["observation_index"], item["reason"]) for item in risks),
    }

    blocked = _named(reason for name, finding in findings.items()
                     if finding["status"] == "blocked" and name != "validator_limitation_exemption"
                     for reason in finding["reasons"])
    advisory = findings["encoding_diagnostics"]["reasons"]
    statuses = {finding["status"] for name, finding in findings.items()
                if name != "validator_limitation_exemption"}
    if "blocked" in statuses:
        verdict = "blocked"
    elif "advisory" in statuses:
        verdict = "advisory"
    elif "not_applicable" in statuses:
        verdict = "contract_not_applicable"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "blocked": blocked,
        "advisory": advisory,
        "exempted": findings["validator_limitation_exemption"]["reasons"],
        "findings": findings,
        "rules": [rule._asdict() for rule in RULES],
        "policy": {
            "merge_allowed": False,
            "not_applicable_means": "candidate predates the structured contract; never reclassified or upgraded",
            "exempted_means": "structural error attributed to our validator gap, not the model",
        },
    }
