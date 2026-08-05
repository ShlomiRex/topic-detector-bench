from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from math import sqrt

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Normalize Unicode and punctuation without assuming a specific language."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(_WORD_RE.findall(text))


def tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize(text).split())


def character_ngrams(text: str, n: int) -> Counter[str]:
    compact = normalize(text).replace(" ", "")
    if not compact:
        return Counter()
    if len(compact) < n:
        return Counter({compact: 1})
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def word_ngrams(text: str, n: int) -> Counter[str]:
    words = tokens(text)
    if not words:
        return Counter()
    if len(words) < n:
        return Counter({"\x1f".join(words): 1})
    return Counter("\x1f".join(words[index : index + n]) for index in range(len(words) - n + 1))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(count * right.get(item, 0) for item, count in left.items())
    left_norm = sqrt(sum(count * count for count in left.values()))
    right_norm = sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm)


def weighted_cosine(left: Counter[str], right: Counter[str], weights: Mapping[str, float]) -> float:
    """Cosine similarity with deterministic per-term weights such as IDF."""
    if not left or not right:
        return 0.0
    numerator = sum(count * right.get(item, 0) * weights.get(item, 1.0) ** 2 for item, count in left.items())
    left_norm = sqrt(sum(count * count * weights.get(item, 1.0) ** 2 for item, count in left.items()))
    right_norm = sqrt(sum(count * count * weights.get(item, 1.0) ** 2 for item, count in right.items()))
    return numerator / (left_norm * right_norm)


def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
