"""Pilot-only validation helpers for source-bound model output.

These separate two *limitations of the reused structural validator* from real
model errors. They never relax the scientific checks; they only stop a validator
representational gap from being scored as a model mistake.

1. ``validate_candidate`` models a p value as one ``operator`` plus one
   ``value``. Abstracts frequently report a *range* (``P = .0007-.02``), which
   cannot be expressed that way, so the validator flags it even though the quote
   is literal and ``value`` is honestly null.
2. The validator requires a sample count to appear as a digit inside its own
   quote. Abstracts often spell counts out ("Thirty virally suppressed PWH"),
   so a correct quote fails ``sample_number_not_source_bound``.
3. The same check rejects an honest ``n = null`` (the abstract states no count),
   a total that must be *summed* from the names of its components ("young adult
   (6 females, 2 males)" → 8) and a single patient described by an article
   ("A 73-year-old man" → 1). None of those is an invented number.
4. A p value written with a descriptor between ``p`` and the operator
   (``p for non-linear < 0.0001``) is literal but does not match the validator's
   adjacency pattern, so the numeric fields are reported as unparsed.

Everything else stays a hard error.
"""
from __future__ import annotations

import re

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

# Longer spellings must be tried first so "sixty-six" is not read as "six".
_NUMBER_WORD = "(?:%s)" % "|".join(sorted(WORD_NUMBERS, key=len, reverse=True))
_NUMBER_PHRASE = re.compile(_NUMBER_WORD + r"(?:[\s-]+" + _NUMBER_WORD + r")*", re.IGNORECASE)


def spell_number(text):
    """Parse a small English number phrase; returns None when unsure."""
    tokens = [token for token in re.split(r"[\s-]+", (text or "").lower()) if token]
    if not tokens or any(token not in WORD_NUMBERS for token in tokens):
        return None
    total = 0
    current = 0
    for token in tokens:
        value = WORD_NUMBERS[token]
        if value == 100:
            current = max(current, 1) * 100
        elif value == 1000:
            total += max(current, 1) * 1000
            current = 0
        else:
            current += value
    return total + current


def spelled_numbers(text):
    return [value for value in (spell_number(group) for group in _NUMBER_PHRASE.findall(text or ""))
            if value is not None]


def digits_match(count, raw):
    """True when ``count`` appears in ``raw`` ignoring numeric separators.

    Abstracts and OCR'd text use space, comma, period or thin-space thousands
    separators ("479 723", "n.110"). The validator only accepts contiguous
    digits, so a faithful count is otherwise reported as unbound.
    """
    if type(count) not in {int, float}:
        return False
    separators = "\\s\\u00a0\\u2009\\u2007\\u202f,._"
    pattern = r"[\d]+(?:[" + separators + r"]\d+)*"
    for token in re.findall(pattern, raw or ""):
        digits = re.sub(r"\D", "", token)
        if not digits:
            continue
        if float(count).is_integer() and int(digits) == int(count):
            return True
    return False


_PLAIN_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_HYPHENATED_DESCRIPTOR = re.compile(r"\b\d[\d.]*-[A-Za-z][\w-]*")
_SINGULAR_SUBJECT = re.compile(
    r"\b(?:a|an|one)\s+(?:[\w./-]+\s+){0,3}"
    r"(?:case|patient|subject|participant|individual|man|woman|boy|girl|child|"
    r"person|autopsy|brain)\b", re.IGNORECASE)
_P_WITH_DESCRIPTOR = re.compile(
    r"\bp\b(?:\s+[A-Za-z][\w-]*){1,4}\s*(?:<=|>=|=|<|>|\u2264|\u2265)\s*\.?\d", re.IGNORECASE)


def _clauses(raw):
    """Split a sample quote into clauses so an age/dose cannot join a head count."""
    return [part for part in re.split(r"[();]", raw or "") if part.strip()]


def component_sum_matches(count, raw):
    """True when the sample total is the sum of the numbers named in its quote.

    Abstracts frequently give only the components ("young adult (6 females,
    2 males)"), so a faithful ``n`` is the arithmetic sum of the literal parts
    rather than any single literal. This is arithmetic over the source, not an
    invented value. Numbers are summed **within a clause**, so an unrelated age
    or dose in a later parenthesis cannot be pulled into the total.
    """
    if type(count) not in {int, float}:
        return False
    for clause in _clauses(raw):
        if not re.search(r"[;,]", clause) and " and " not in clause.lower():
            continue
        parts = [float(value) for value in _PLAIN_NUMBER.findall(clause)]
        parts += [float(value) for value in spelled_numbers(clause)]
        if len(parts) < 2 or any(abs(part - float(count)) < 1e-9 for part in parts):
            continue
        if abs(sum(parts) - float(count)) < 1e-9:
            return True
    return False


def singular_subject_match(count, raw):
    """True when a count of one is written as a single case ("A 73-year-old man")."""
    if type(count) not in {int, float} or float(count) != 1.0:
        return False
    residue = _HYPHENATED_DESCRIPTOR.sub(" ", raw or "")
    if _PLAIN_NUMBER.search(residue) or spelled_numbers(residue):
        return False
    return bool(_SINGULAR_SUBJECT.search(raw or ""))


def sample_number_is_validator_gap(number, raw):
    """True when ``sample_number_not_source_bound`` reflects the validator, not the model."""
    if number is None:
        # ``n = null`` is the honest "the abstract states no count" encoding. The
        # structural check asks whether the model *invented* a number; with no
        # number supplied there is nothing to fabricate. A dropped count is a
        # coverage question, measured against the root reference, not a
        # structural hard block.
        return True
    if type(number) not in {int, float}:
        return False
    return (any(float(value) == float(number) for value in spelled_numbers(raw))
            or digits_match(number, raw)
            or component_sum_matches(number, raw)
            or singular_subject_match(number, raw))


def validator_limitations(structural_errors, candidate, source_text):
    """Return structural errors that are validator gaps, not model mistakes."""
    if not structural_errors or not isinstance(candidate, dict):
        return []
    limitations = set()
    study = candidate.get("study") or {}
    samples = study.get("samples") if isinstance(study, dict) else None
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            raw = sample.get("raw")
            number = sample.get("n")
            if not isinstance(raw, str) or not raw or not isinstance(source_text, str):
                continue
            if raw not in source_text:
                continue
            if sample_number_is_validator_gap(number, raw):
                limitations.add("sample_number_not_source_bound")
                break
    observations = candidate.get("observations")
    if isinstance(observations, list):
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                continue
            statistics = observation.get("statistics")
            if not isinstance(statistics, list):
                continue
            for statistic in statistics:
                if not isinstance(statistic, dict) or statistic.get("kind") != "p_value":
                    continue
                raw = statistic.get("raw") or ""
                if re.search(r"[=<>]\s*\.?\d+(?:\.\d+)?\s*[-\u2013\u2014]\s*\.?\d", raw):
                    limitations.add("observation_%d_p_operator_missing" % index)
                    limitations.add("observation_%d_p_value_or_operator_changed" % index)
                    continue
                if _P_WITH_DESCRIPTOR.search(raw):
                    # A descriptor between "p" and the operator ("p for non-linear
                    # < 0.0001") is still a literal, reported p value.
                    limitations.add("observation_%d_p_operator_missing" % index)
                    limitations.add("observation_%d_p_value_or_operator_changed" % index)
                    limitations.add("observation_%d_p_value_unparsed" % index)
                    continue
                if any(mark in raw for mark in ("\u2264", "\u2265")):
                    # The structural validator only knows ASCII operators; a
                    # literal "p ≤ 0.048" is faithful but unrecognized.
                    limitations.add("observation_%d_p_operator_missing" % index)
                    limitations.add("observation_%d_p_value_or_operator_changed" % index)
    return sorted(set(limitations) & set(structural_errors))
