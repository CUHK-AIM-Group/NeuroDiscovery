"""Source-bound checks for effect direction and claimed significance.

A confirmed defect: PMID 28138428 reports only that DTI measures "correlated
with duration of epilepsy", yet the candidate recorded direction ``positive``
and significance ``reported_significant``. The structural validator accepted it
because the quote was literal and the number fields were empty. A signed
direction must be traceable to a signed word or a signed statistic in the
source, not inferred from a bare association.

The revised checks (2026-09-27) fix three further defects the first version had:

* a signed ``r`` (``r = -0.42``), a p-value operator/interval (``p < 0.05``,
  ``0.01 < p < 0.05``) or a confidence interval is now read from the source, so
  a true signed correlation is no longer blocked merely because the paper never
  prints the word "positive";
* HTML entities and non-breaking spaces are normalized before matching, so a
  literal quote copied from the compiled corpus is not rejected on whitespace;
* *reported* significance is separated from significance that can only be
  *derived* from a numeric interval. A derivable-but-unreported result is
  recorded as an advisory caveat, not silently promoted to
  ``reported_significant`` and not silently dropped.

These checks report problems instead of rewriting the candidate; the extraction
artifact is never mutated.
"""
from __future__ import annotations

import html
import re

SIGNED_POSITIVE_WORDS = (
    "positive", "positively", "greater", "higher", "increased", "increase in",
    "larger", "higher than", "greater than", "superior", "positively related",
    "positively correlated", "directly related", "directly correlated",
    # Base forms and clear synonyms of the same signed class. Their absence
    # blocked literal signed wording ("a decrease in connectivity") merely
    # because the paper did not print the inflected form already listed.
    "increase", "increases", "elevated", "elevation", "improved", "improvement",
    "improvements", "improve", "improves", "improving", "enhanced", "enhance",
    "enhances",
)
SIGNED_NEGATIVE_WORDS = (
    "negative", "negatively", "lower", "smaller", "reduced", "decreased",
    "less than", "lower than", "smaller than", "inversely", "shrinkage",
    "negatively related", "negatively correlated", "inversely related",
    "inversely correlated", "atrophy",
    "decrease", "decrease in", "decreases", "decline", "declined", "low",
    "impaired", "impairment", "disrupted", "disruption", "attenuated",
    "attenuate", "thinner", "inverse", "reduction", "slow", "slowed", "slowing",
    "suppress", "suppressed",
)
# Words that assert a *quantity* is down without being a bare comparative. A
# paper that quotes "deficits in PV and CB" has stated the sign; blocking it for
# lacking the literal word "lower" is a validator gap, not a model mistake.
SIGNED_NEGATIVE_QUANTITY_WORDS = (
    "deficit", "deficits", "diminished", "fewer", "loss", "losses", "lowered",
    "depleted", "thinning", "sparse",
)
UNSIGNED_ASSOCIATION = ("correlat", "associat", "relat", "link", "linked", "co-varies", "covar")

# A signed word must be a standalone token. Substring matching wrongly read the
# cell-type marker in "CR-positive cells" as the direction word "positive",
# which then "contradicted" a correctly typed lower-direction claim.
# An optional regular English inflection is allowed, so "reduction"/"reductions"
# and "impair"/"impaired"/"impairment" do not each need a hand-listed form.
_INFLECTION = r"(?:s|es|ed|d|ing)?"
_SIGNED_WORD_PATTERNS = {
    word: re.compile(r"(?<![-\w])" + re.escape(word) + _INFLECTION + r"(?![-\w])",
                     re.IGNORECASE)
    for word in (SIGNED_POSITIVE_WORDS + SIGNED_NEGATIVE_WORDS + SIGNED_NEGATIVE_QUANTITY_WORDS)
}

# "negative symptoms" and "positive symptoms" are clinical terms, not effect
# signs; matching the adjective inside them blocked correctly typed claims.
_CLINICAL_SYMPTOM_TERM = re.compile(
    r"\b(?:positive|negative|false\s+positive|false\s+negative)\s+"
    r"(?:symptom|symptoms|syndrome|syndromes)\b", re.IGNORECASE)


def signed_word_hits(text, words):
    """Return the signed words present as standalone tokens in ``text``."""
    scrubbed = _CLINICAL_SYMPTOM_TERM.sub(" ", text)
    return [word for word in words if _SIGNED_WORD_PATTERNS[word].search(scrubbed)]

# Significance that the source states outright. Only markers compatible with a
# *significant* claim are listed; "p > 0.05" must not make a claim look supported.
SIGNIFICANCE_MARKERS = (
    "significan", "p =", "p=", "p <", "p<", "p \u2264", "p\u2264", "p-value", "p value",
)
# A statistical result that lets significance be *derived* rather than quoted.
DERIVED_STATISTIC_MARKERS = ("confidence interval", "95% ci", "95%ci", "95 % ci",
                             "odds ratio", "hazard ratio", "beta coefficient")
ADJUSTMENT_MARKERS = (
    "fdr", "bonferroni", "false discovery", "holm", "adjusted for", "adjusting for",
    "adjustment for", "covariate", "multiple comparison", "corrected for",
)

# Directions that assert a sign and therefore require signed source support.
SIGNED_DIRECTIONS = {
    "positive": True, "higher": True, "increase": True, "increased": True,
    "greater": True, "larger": True, "elevated": True,
    "negative": False, "lower": False, "decrease": False, "decreased": False,
    "reduced": False, "smaller": False, "reduction": False,
}

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_R_VALUE = re.compile(r"\br\s*(?:=|:|\u2248|~)\s*(?P<value>%s)" % _NUMBER, re.IGNORECASE)
# Abstracts routinely interpose a short descriptor between "p" and its operator
# ("p for non-linear < 0.0001", "p for interaction = 0.03"). Requiring the
# operator to be adjacent to "p" read those literal p values as unreported.
_P_VALUE = re.compile(
    r"\bp\b(?:\s+[A-Za-z][\w-]*){0,4}\s*(?:<=|>=|=|<|>|\u2264|\u2265|:)?\s*(?P<value>%s)"
    % _NUMBER, re.IGNORECASE)
_WHITESPACE = re.compile(r"[\s\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")


def normalize_source(text):
    """Decode HTML entities and collapse unicode whitespace for matching only."""
    if text is None:
        return ""
    return _WHITESPACE.sub(" ", html.unescape(str(text))).strip()


def _source_text(observation):
    """The literal source anchors only (quotes).

    A signed direction or a significance claim must be traceable to the *paper's
    own words*, so the blocking checks read only the quotes. The model's
    ``statement`` is its own phrasing; using it here would let a model self-
    certify a sign it never read, which is exactly the reported defect.
    """
    quotes = " ".join(normalize_source(quote) for quote in ((observation or {}).get("quotes") or []))
    return quotes.lower()


def _text(observation):
    """Statement plus quotes; used only for non-blocking advisories."""
    statement = normalize_source((observation or {}).get("statement") or "")
    return (statement + " " + _source_text(observation)).lower()


def _has_anchor(observation):
    return any((quote or "").strip() for quote in ((observation or {}).get("quotes") or []))


def signed_statistics(text):
    """Return the signs carried by numeric statistics in the text.

    ``True`` means the source statistic is positive, ``False`` negative. Signs
    are collected from ``r``/beta values and from a bounded p-value direction is
    NOT treated as a signed effect, because p only speaks about detectability.
    """
    signs = set()
    for match in _R_VALUE.finditer(text):
        raw = match.group("value")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value < 0:
            signs.add(False)
        elif value > 0:
            signs.add(True)
    return signs


def bounded_p_values(text):
    """Return every p-value literal written with an inequality operator."""
    values = []
    for match in _P_VALUE.finditer(text):
        start = max(0, match.start() - 3)
        operator = text[start:match.start("value")]
        if any(mark in operator for mark in ("<", ">", "\u2264", "\u2265")):
            try:
                values.append(float(match.group("value")))
            except ValueError:
                continue
    return values


def p_values(text):
    """Return ``(operator, value)`` for every p-value literal in the text."""
    found = []
    for match in _P_VALUE.finditer(text):
        start = max(0, match.start() - 3)
        prefix = text[start:match.start("value")]
        operator = "".join(char for char in prefix if char in "=<>:\u2264\u2265")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        found.append((operator, value))
    return found


NEGATED_SIGNIFICANCE = (
    "not significant", "not significantly", "no significant", "non significant",
    "non-significant", "nonsignificant", "without significance",
    "did not reach significance", "failed to reach significance",
    "not statistically significant", "was not significant",
)

# British/European spelling of "significant" ("significative impairment"). The
# significance check already matches the stem "significan"; this covers the
# variant so a plainly stated result is not read as unreported.
_SIGNIFICANCE_STEM = re.compile(r"signific(?:an|ativ)", re.IGNORECASE)


def significance_supported_by_text(text):
    """True when the text states a significant result rather than implying one."""
    if _SIGNIFICANCE_STEM.search(text) and not any(marker in text for marker in NEGATED_SIGNIFICANCE):
        return True
    for operator, value in p_values(text):
        if "<" in operator and value <= 0.1:
            return True
        if "\u2264" in operator and value <= 0.05:
            return True
        if operator in {"=", ":", "=="} and value <= 0.05:
            return True
    return False


def direction_conflicts(observation):
    """Return reasons a signed direction is not supported by the supplied text."""
    proposition = (observation or {}).get("proposition") or {}
    result = (observation or {}).get("result") or {}
    # Blocking checks bind to the source anchors only, never the model's
    # self-authored statement.
    text = _source_text(observation)
    if not text:
        return ["observation_no_source_anchor"]
    words_positive = bool(signed_word_hits(text, SIGNED_POSITIVE_WORDS))
    words_negative = bool(signed_word_hits(text, SIGNED_NEGATIVE_WORDS)
                          or signed_word_hits(text, SIGNED_NEGATIVE_QUANTITY_WORDS))
    stats = signed_statistics(text)
    stats_positive = True in stats
    stats_negative = False in stats
    positive = words_positive or stats_positive
    negative = words_negative or stats_negative
    reasons = []
    for field, value in (("result.direction", result.get("direction")),
                         ("proposition.direction", proposition.get("direction"))):
        if not isinstance(value, str):
            continue
        wanted = SIGNED_DIRECTIONS.get(value.strip().lower())
        if wanted is None:
            continue
        if wanted and negative and not positive:
            reasons.append("%s_contradicts_source" % field)
        elif not wanted and positive and not negative:
            reasons.append("%s_contradicts_source" % field)
        elif not positive and not negative and any(word in text for word in UNSIGNED_ASSOCIATION):
            reasons.append("%s_not_source_bound" % field)
    return sorted(set(reasons))


def significance_conflicts(observation):
    """Report significance claimed without any source significance wording."""
    result = (observation or {}).get("result") or {}
    if result.get("significance") != "reported_significant":
        return []
    text = _source_text(observation)
    if not text:
        return ["result.significance_no_source_anchor"]
    if significance_supported_by_text(text):
        return []
    return ["result.significance_not_source_bound"]


def significance_advisories(observation):
    """Caveats that must be preserved even though they do not block the candidate.

    A significance that is only derivable from a CI (never stated as significant
    in the text) and a result that rests on an adjusted/FDR model are recorded so
    the reviewer can see the difference from a plainly reported p-value.
    """
    result = (observation or {}).get("result") or {}
    text = _text(observation)
    advisories = []
    if result.get("significance") == "reported_significant":
        stated = significance_supported_by_text(text)
        derives_from_interval = any(marker in text for marker in DERIVED_STATISTIC_MARKERS)
        if not stated and derives_from_interval:
            advisories.append("result.significance_derived_not_reported")
    if not any(marker in text for marker in ADJUSTMENT_MARKERS) and (
            result.get("significance") == "reported_significant"):
        advisories.append("result.no_adjustment_caveat_reported")
    return sorted(set(advisories))


def candidate_consistency_errors(candidate):
    """Per-observation source-consistency problems for one candidate object."""
    errors = []
    for index, observation in enumerate((candidate or {}).get("observations") or []):
        if not isinstance(observation, dict):
            continue
        for reason in direction_conflicts(observation) + significance_conflicts(observation):
            errors.append({"observation_index": index, "reason": reason,
                           "statement": observation.get("statement")})
    return errors


def candidate_advisories(candidate):
    """Per-observation non-blocking caveats for one candidate object."""
    advisories = []
    for index, observation in enumerate((candidate or {}).get("observations") or []):
        if not isinstance(observation, dict):
            continue
        for reason in significance_advisories(observation):
            advisories.append({"observation_index": index, "reason": reason,
                               "statement": observation.get("statement")})
    return advisories
