"""Conservative p-value parsing; original evidence strings stay unchanged."""
import html
import re


P_LITERAL = re.compile(
    r"\bp\s*(?P<operator><\s*or\s*=|<=|>=|[=<>≤≥])\s*"
    r"(?P<number>(?:0\.\s+\d+|(?:\d*\.\d+|\d+))(?:e[-+]?\d+)?)"
    r"(?![\d.eE])", re.IGNORECASE,
)


def parse_p_literal(text):
    """Return one unambiguous operator/value, or None; never take a range endpoint."""
    normalized = html.unescape(text)
    matches = list(P_LITERAL.finditer(normalized))
    if len(matches) != 1:
        return None
    match = matches[0]
    suffix = normalized[match.end():]
    if re.match(r"\s*(?:[-–—]|to\b|[<>≤≥])\s*\.?\d", suffix, re.I):
        return None
    if re.search(r"\d\s*[<>≤≥]\s*$", normalized[:match.start()]):
        return None
    if re.match(r"\s+\d", suffix):
        return None
    value = float(re.sub(r"\s", "", match.group("number")))
    if not 0 <= value <= 1:
        return None
    operator = re.sub(r"\s", "", match.group("operator")).lower()
    operator = operator.replace("<or=", "<=").replace("≤", "<=").replace("≥", ">=")
    return operator, value
