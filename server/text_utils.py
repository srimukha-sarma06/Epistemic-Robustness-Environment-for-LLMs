"""
text_utils.py — Shared text analysis helpers for the graders.

All graders are deterministic and black-box: they only see the agent's text.
These helpers keep the matching rules consistent across tasks:

  - term matching uses word boundaries ("go" must not match "ago")
  - analysis is per sentence, so a refusal that *mentions* an entity is not
    confused with a sentence that *asserts* something about it
  - "novel specifics" are numbers / proper nouns that do not appear in a
    reference text (the question or the context document)
"""

import re
from typing import Iterable, Sequence

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Words that mark a sentence as negated, hedged or conditional. A sentence
# containing one of these is not treated as a confident factual assertion.
_HEDGE_MARKERS = re.compile(
    r"\b(not|no|never|none|nothing|cannot|can't|cant|couldn't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|won't|unable|unclear|unknown|unaware|unsure|"
    r"uncertain|unverified|unverifiable|verify|verified|confirm|aware|familiar|"
    r"record|if|whether|might|may|could|would|possibly|perhaps|any)\b",
    re.IGNORECASE,
)

# Words that, in front of a term, flip its meaning ("not relativity",
# "the myth that ...").
_NEGATION_CUES = re.compile(
    r"\b(not|no|never|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|"
    r"rather than|instead of|myth|misconception|false|wrong|incorrect|"
    r"mistaken|untrue|debunked|contrary to)\b",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    """Lowercase, unify apostrophes and collapse whitespace."""
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.lower()).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def has_term(text: str, term: str) -> bool:
    """Case-insensitive match of `term` with word boundaries on both sides."""
    pattern = r"(?<![\w])" + re.escape(norm(term)) + r"(?![\w])"
    return re.search(pattern, norm(text)) is not None


def has_any(text: str, terms: Iterable[str]) -> bool:
    return any(has_term(text, t) for t in terms)


def count_any(text: str, terms: Iterable[str]) -> int:
    return sum(1 for t in terms if has_term(text, t))


def fact_coverage(text: str, key_facts: Sequence[Sequence[str]]) -> float:
    """
    Fraction of key facts present in `text`.

    `key_facts` is a list of fact groups; each group lists acceptable
    alternatives, e.g. [["berners-lee", "berners lee"], ["1989"]].
    """
    if not key_facts:
        return 1.0
    hits = sum(1 for group in key_facts if has_any(text, group))
    return hits / len(key_facts)


def is_hedged(sentence: str) -> bool:
    return _HEDGE_MARKERS.search(sentence) is not None


_CLAUSE_BREAK = re.compile(r"[,;:()—]| - | but | however | rather | instead | whereas | although ")
_MYTH_FRAME = re.compile(r"\b(myth|misconception|debunked|popular belief|common belief|wrongly believe)\b")


def term_spans(text: str, term: str) -> list[tuple[int, int]]:
    """(start, end) spans of `term` in norm(text), word-bounded."""
    pattern = r"(?<![\w])" + re.escape(norm(term)) + r"(?![\w])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, norm(text))]


def is_negated_at(sentence_norm: str, start: int, window: int = 5, myth_frame: bool = False) -> bool:
    """
    True if a negation cue appears in the same clause, within `window` words
    before position `start` of an already-normalised sentence. With
    `myth_frame`, a sentence that calls something a myth/misconception also
    counts as negating it.
    """
    before = sentence_norm[:start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]
    if _NEGATION_CUES.search(" ".join(before.split()[-window:])):
        return True
    return myth_frame and _MYTH_FRAME.search(sentence_norm) is not None


def numbers(text: str) -> set[str]:
    """Numbers in text, normalised (commas stripped, trailing % kept)."""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def proper_nouns(text: str) -> set[str]:
    """
    Capitalised words that are not sentence-initial — a cheap proxy for named
    entities (people, companies, places, products).
    """
    found = set()
    for sent in sentences(text):
        words = _WORD.findall(sent)
        for w in words[1:]:
            if w[0].isupper() and w.lower() not in _COMMON_CAPITALISED:
                found.add(w.lower())
    return found


def novel_specifics(text: str, *references: str) -> set[str]:
    """
    Numbers and proper nouns in `text` that appear in none of `references`.
    Used to detect details the agent brought in from outside the source.
    """
    ref = " ".join(references)
    ref_norm = norm(ref)
    ref_numbers = numbers(ref)
    out = {n for n in numbers(text) if n not in ref_numbers}
    out |= {w for w in proper_nouns(text) if w not in ref_norm}
    return out


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of word sets — used to spot copy-paste replies."""
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_COMMON_CAPITALISED = {
    "i", "i'm", "i've", "i'd", "i'll", "the", "a", "an", "this", "that", "these",
    "however", "according", "based", "document", "context", "question", "answer",
    "note", "also", "but", "and", "or", "so", "therefore", "it", "its", "in",
    "on", "of", "for", "to", "as", "if", "is", "are", "was", "were", "be",
    "yes", "no", "not", "my", "your", "you", "we", "our", "they", "he", "she",
    "issue", "issues", "report", "policy", "project", "company", "summary",
    # currency / units the documents write as symbols (₹) or in lower case
    "rs", "inr", "rupees", "rupee", "lakh", "lakhs", "crore", "crores",
}
