import re


POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "loved", "like", "liked",
    "perfect", "nice", "happy", "satisfied", "recommend", "recommendation",
}
NEGATIVE = {
    "bad", "terrible", "awful", "hate", "hated", "poor", "worse", "worst",
    "broken", "disappoint", "disappointed", "return", "refund", "not", "never",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))


def predict_recommendation(title: str, body: str) -> int:
    """Heuristic placeholder: positive vs negative token count.

    Returns 1 for recommend, 0 for not recommend.
    """
    toks = _tokens(title) | _tokens(body)
    pos = len(toks & POSITIVE)
    neg = len(toks & NEGATIVE)
    if neg > pos:
        return 0
    if pos > neg:
        return 1
    # Tie-breaker: default to recommend if rating tends to be positive is unknown here.
    return 1

