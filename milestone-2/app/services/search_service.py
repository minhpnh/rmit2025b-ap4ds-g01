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
    """Return True if any stemmed query token appears in product fields.

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
