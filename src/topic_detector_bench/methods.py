from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log
from typing import Any

from .models import TopicDefinition
from .text import character_ngrams, cosine, jaccard, normalize, tokens, weighted_cosine, word_ngrams


Similarity = Callable[[str, str], float]


def token_jaccard(left: str, right: str) -> float:
    return jaccard(tokens(left), tokens(right))


def char_ngram_cosine(n: int) -> Similarity:
    def similarity(left: str, right: str) -> float:
        return cosine(character_ngrams(left, n), character_ngrams(right, n))

    return similarity


def word_ngram_cosine(n: int) -> Similarity:
    def similarity(left: str, right: str) -> float:
        return cosine(word_ngrams(left, n), word_ngrams(right, n))

    return similarity


def _seed_idf(topic: TopicDefinition) -> tuple[dict[str, float], float]:
    documents = [tokens(phrase) for phrase in (*topic.positive, *topic.negative)]
    document_count = len(documents)
    document_frequency = Counter(token for document in documents for token in set(document))
    idf = {token: log((document_count + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}
    average_length = sum(len(document) for document in documents) / document_count
    return idf, average_length


def tfidf_token_cosine(topic: TopicDefinition) -> Similarity:
    idf, _ = _seed_idf(topic)

    def similarity(left: str, right: str) -> float:
        return weighted_cosine(Counter(tokens(left)), Counter(tokens(right)), idf)

    return similarity


def bm25_token_similarity(topic: TopicDefinition) -> Similarity:
    idf, average_length = _seed_idf(topic)
    k1, b = 1.2, 0.75

    def similarity(query: str, reference: str) -> float:
        query_terms = set(tokens(query))
        reference_terms = Counter(tokens(reference))
        if not query_terms or not reference_terms:
            return 0.0
        denominator = sum(idf.get(term, 1.0) for term in query_terms)
        score = sum(
            idf.get(term, 1.0)
            * (count * (k1 + 1))
            / (count + k1 * (1 - b + b * len(reference_terms) / average_length))
            for term, count in reference_terms.items()
            if term in query_terms
        )
        return min(score / denominator, 1.0)

    return similarity


def sequence_ratio(left: str, right: str) -> float:
    return SequenceMatcher(a=normalize(left), b=normalize(right), autojunk=False).ratio()


def normalized_subphrase(left: str, right: str) -> float:
    normalized_left, normalized_right = normalize(left), normalize(right)
    if not normalized_left or not normalized_right:
        return 0.0
    return 1.0 if normalized_left in normalized_right or normalized_right in normalized_left else 0.0


@dataclass(frozen=True)
class Candidate:
    method: str
    negative_weight: float
    threshold: float
    ngram_size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "negative_weight": self.negative_weight,
            "threshold": self.threshold,
            "ngram_size": self.ngram_size,
        }


class Detector:
    """A fixed detector constructed only from supplied seed phrases and parameters."""

    def __init__(self, topic: TopicDefinition, candidate: Candidate):
        self.topic = topic
        self.candidate = candidate
        self._similarity = self._select_similarity(topic, candidate)

    @staticmethod
    def _select_similarity(topic: TopicDefinition, candidate: Candidate) -> Similarity:
        if candidate.method == "token_jaccard":
            return token_jaccard
        if candidate.method == "sequence_ratio":
            return sequence_ratio
        if candidate.method == "subphrase":
            return normalized_subphrase
        if candidate.method == "char_ngram_cosine":
            if candidate.ngram_size is None:
                raise ValueError("char_ngram_cosine requires ngram_size.")
            return char_ngram_cosine(candidate.ngram_size)
        if candidate.method == "word_ngram_cosine":
            if candidate.ngram_size is None:
                raise ValueError("word_ngram_cosine requires ngram_size.")
            return word_ngram_cosine(candidate.ngram_size)
        if candidate.method == "tfidf_token_cosine":
            return tfidf_token_cosine(topic)
        if candidate.method == "bm25_token":
            return bm25_token_similarity(topic)
        raise ValueError(f"Unknown method: {candidate.method}")

    def evidence(self, text: str) -> tuple[float, float]:
        positive = max(self._similarity(text, phrase) for phrase in self.topic.positive)
        negative = max(self._similarity(text, phrase) for phrase in self.topic.negative)
        return positive, negative

    def score(self, text: str) -> float:
        positive, negative = self.evidence(text)
        return positive - (self.candidate.negative_weight * negative)

    def predict(self, text: str) -> bool:
        return self.score(text) >= self.candidate.threshold


def candidate_space() -> list[Candidate]:
    """The automatic, topic-agnostic configurations searched by the benchmark."""
    candidates: list[Candidate] = []
    method_specs = [
        ("token_jaccard", None),
        ("sequence_ratio", None),
        ("subphrase", None),
        *(("char_ngram_cosine", size) for size in (2, 3, 4, 5)),
        *(("word_ngram_cosine", size) for size in (1, 2, 3)),
        ("tfidf_token_cosine", None),
        ("bm25_token", None),
    ]
    for method, ngram_size in method_specs:
        for negative_weight in (0.5, 0.75, 1.0, 1.25, 1.5):
            for threshold in (index / 20 for index in range(-10, 21)):
                candidates.append(Candidate(method, negative_weight, threshold, ngram_size))
    return candidates
