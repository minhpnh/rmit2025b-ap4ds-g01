import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Regex pattern used in Milestone 1 Task 1
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


def _m2_root() -> Path:
    # .../milestone-2/app/services -> parents[2] = milestone-2 root
    return Path(__file__).resolve().parents[2]


def _default_stopwords_path() -> Path:
    # Prefer Flask config if available; else fallback to milestone-2/data
    try:
        from flask import current_app  # type: ignore
        cfg_path = current_app.config.get("STOPWORDS_PATH")
        if cfg_path:
            return Path(cfg_path)
    except Exception:
        pass
    return _m2_root() / "data" / "stopwords_en.txt"


def _default_vocab_path() -> Path:
    # Prefer Flask config if available; else fallback to milestone-2/data
    try:
        from flask import current_app  # type: ignore
        cfg_path = current_app.config.get("VOCAB_PATH")
        if cfg_path:
            return Path(cfg_path)
    except Exception:
        pass
    return _m2_root() / "data" / "vocab.txt"


class _IdentityLemmatizer:
    def lemmatize(self, w: str) -> str:  # fallback when nltk not available
        return w


def _load_lemmatizer():
    """Load WordNetLemmatizer if available; otherwise, return identity lemmatizer.

    This mirrors Task 1's choice to lemmatize, but stays optional so importing
    this module never crashes if nltk data isn't installed in the runtime.
    """
    try:
        import nltk  # type: ignore
        from nltk.stem import WordNetLemmatizer  # type: ignore

        # Try to ensure wordnet is present; if not, skip download in production.
        try:
            # Access something from wordnet to validate presence
            from nltk.corpus import wordnet as _wn  # type: ignore
            _ = _wn.synsets("test")
        except Exception:
            # If not present, don't attempt network download here; fall back.
            return _IdentityLemmatizer()
        return WordNetLemmatizer()
    except Exception:
        return _IdentityLemmatizer()


def _load_spellchecker():
    """Return a SpellChecker instance if library is installed, else None."""
    try:
        from spellchecker import SpellChecker  # type: ignore
        return SpellChecker(distance=2)
    except Exception:
        return None


def _read_lines(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


def _load_stopwords(path: Optional[Path] = None) -> Set[str]:
    if path is None:
        path = _default_stopwords_path()
    words = set(w.lower() for w in _read_lines(path))
    # Minimal fallback if file missing
    if not words:
        words = {
            "the", "and", "a", "an", "is", "it", "to", "in", "for", "of", "on",
            "at", "by", "with", "that", "this", "was", "were", "be", "are",
        }
    return words


def _load_vocab(path: Optional[Path] = None) -> Dict[str, int]:
    if path is None:
        path = _default_vocab_path()
    vocab: Dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                word, idx = line.split(":", 1)
                try:
                    vocab[word] = int(idx)
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return vocab


def _tokenize(text: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    # strip possible leading/trailing hyphen/apostrophe remnants
    return [t.strip("-'") for t in tokens if t.strip("-'")]


def _remove_stopwords(tokens: Iterable[str], stopwords: Set[str]) -> List[str]:
    return [t for t in tokens if t not in stopwords]


def _lemmatize(tokens: Iterable[str], lemmatizer) -> List[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]


def _apply_typo_fixes(
    tokens: Iterable[str],
    spellchecker,
    approved_fixes: Optional[Dict[str, str]] = None,
    ignore_words: Optional[Set[str]] = None,
) -> List[str]:
    if approved_fixes is None:
        approved_fixes = {}
    if ignore_words is None:
        ignore_words = set()
    out: List[str] = []
    for t in tokens:
        if t in ignore_words:
            out.append(t)
            continue
        if t in approved_fixes:
            out.append(approved_fixes[t])
            continue
        if spellchecker is None:
            out.append(t)
            continue
        # Only attempt correction for words not seen in spell lexicon
        if t in spellchecker:  # type: ignore[attr-defined]
            out.append(t)
        else:
            try:
                suggestion = spellchecker.correction(t)  # type: ignore[union-attr]
            except Exception:
                suggestion = None
            out.append(suggestion or t)
    return out


def _bow_indices(tokens: Iterable[str], vocab: Dict[str, int]) -> Dict[int, int]:
    if not vocab:
        return {}
    ctr = Counter(t for t in tokens if t in vocab)
    return {vocab[t]: c for t, c in ctr.items()}


class ReviewTextProcessor:
    """Mirror of Milestone 1 Task 1 preprocessing, adapted for runtime use.

    Steps:
      1) Tokenize with the same regex pattern (handles hyphens/apostrophes)
      2) Lowercase and basic cleanup
      3) Stopword removal (uses Milestone 1 stopwords list if present)
      4) Lemmatization (WordNet if available; otherwise identity)
      5) Optional typo fixing with spellchecker + custom overrides
      6) Optional Bag-of-Words mapping using Task 1 `vocab.txt`
    """

    def __init__(
        self,
        *,
        stopwords_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        approved_fixes: Optional[Dict[str, str]] = None,
        ignore_words: Optional[Iterable[str]] = None,
    ) -> None:
        self.stopwords = _load_stopwords(stopwords_path)
        self.vocab = _load_vocab(vocab_path)
        self.lemmatizer = _load_lemmatizer()
        self.spellchecker = _load_spellchecker()
        self.approved_fixes = approved_fixes or {}
        self.ignore_words = set(ignore_words or [])

    def process_text(self, text: str) -> Dict[str, object]:
        """Process a single free-text field and return structured outputs.

        Returns dict with keys:
          - tokens: final token list (after all steps)
          - text: space-joined final text
          - bow: {index: count} using Task 1 vocab if available (else empty)
        """
        toks0 = _tokenize(text)
        toks1 = _remove_stopwords(toks0, self.stopwords)
        toks2 = _lemmatize(toks1, self.lemmatizer)
        toks3 = _apply_typo_fixes(toks2, self.spellchecker, self.approved_fixes, self.ignore_words)
        final_tokens = [t for t in toks3 if t]  # drop empties
        final_text = " ".join(final_tokens)
        bow = _bow_indices(final_tokens, self.vocab)
        return {"tokens": final_tokens, "text": final_text, "bow": bow}

    def process_review(self, title: str, body: str) -> Dict[str, object]:
        """Process a review composed of title and body.

        Returns dict with keys:
          - title: per-field processed dict (tokens, text, bow)
          - body: per-field processed dict (tokens, text, bow)
          - combined: combined processed dict using concatenated tokens
        """
        pt = self.process_text(title or "")
        pb = self.process_text(body or "")
        combo_tokens = list((pt["tokens"] or [])) + list((pb["tokens"] or []))  # type: ignore[index]
        combo_text = " ".join(combo_tokens)
        combo_bow = _bow_indices(combo_tokens, self.vocab)
        return {
            "title": pt,
            "body": pb,
            "combined": {"tokens": combo_tokens, "text": combo_text, "bow": combo_bow},
        }


# Convenience singleton with defaults resolving to Milestone 1 assets
_DEFAULT_PROCESSOR: Optional[ReviewTextProcessor] = None


def default_processor() -> ReviewTextProcessor:
    global _DEFAULT_PROCESSOR
    if _DEFAULT_PROCESSOR is None:
        _DEFAULT_PROCESSOR = ReviewTextProcessor()
    return _DEFAULT_PROCESSOR


def process_user_review(title: str, body: str) -> Dict[str, object]:
    """One-shot helper using the default processor.

    Example usage:
        from .review_text_service import process_user_review
        result = process_user_review(title, body)
        tokens = result["combined"]["tokens"]
        text = result["combined"]["text"]
        bow = result["combined"]["bow"]
    """
    return default_processor().process_review(title, body)
