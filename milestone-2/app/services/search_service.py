import re
from markupsafe import Markup, escape

def stem(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 3: return w[:-3] + "y"
    if w.endswith("es")  and len(w) > 3: return w[:-2]
    if w.endswith("s")   and len(w) > 2: return w[:-1]
    return w

def tokenize(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"\b\w+\b", text.lower())}

def match_query(product: dict, q_tokens: set[str]) -> bool:
    """Return True if the query loosely matches product fields.

    Supports partial typing like "dre" → "dress" via substring match,
    and also falls back to simple stem-token overlap.

    Fields searched: title, description, division, department, class, id.
    """
    if not q_tokens:
        return True
    haystack = " ".join([
        str(product.get("title", "")),
        str(product.get("description", "")),
        str(product.get("division", "")),
        str(product.get("department", "")),
        str(product.get("class", "")),
        str(product.get("id", "")),
    ])
    haystack_lc = haystack.lower()
    # Substring dynamic match
    for qt in q_tokens:
        if qt and qt in haystack_lc:
            return True
    # Fallback to stem-token overlap
    tokens = tokenize(haystack)
    return bool(tokens & q_tokens)

def highlight(text: str, q: str | None):
    """Escape user text first, then wrap matched tokens in <mark> safely."""
    if not q:
        return escape(text)  # return Markup-safe escaped text
    tokens = {stem(w) for w in re.findall(r"\b\w+\b", q.lower())}
    escaped = escape(text)   # escape BEFORE regex
    # Now run regex on the escaped string (works because \w matches letters/digits/_)
    def repl(m):
        w = m.group(0)
        return Markup(f"<mark>{w}</mark>") if stem(w) in tokens else w
    marked = re.sub(r"\b\w+\b", repl, escaped)
    return Markup(marked)
