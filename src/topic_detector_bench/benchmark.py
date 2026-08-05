from __future__ import annotations

import json
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Iterable

from .methods import Candidate, Detector, candidate_space
from .models import LabeledExample, TopicDefinition


@dataclass(frozen=True)
class Metrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0

    def f_beta(self, beta: float) -> float:
        if self.precision == 0 and self.recall == 0:
            return 0.0
        beta_squared = beta * beta
        return (1 + beta_squared) * self.precision * self.recall / (beta_squared * self.precision + self.recall)


@dataclass(frozen=True)
class BenchmarkResult:
    candidate: Candidate
    metrics: Metrics
    beta: float

    @property
    def f_beta(self) -> float:
        return self.metrics.f_beta(self.beta)

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate.as_dict(),
            "metrics": asdict(self.metrics)
            | {
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "false_positive_rate": self.metrics.false_positive_rate,
                "f_beta": self.f_beta,
            },
        }


@dataclass(frozen=True)
class ScoredExample:
    text: str
    actual: bool
    predicted: bool
    score: float
    positive_evidence: float
    negative_evidence: float

    @property
    def correct(self) -> bool:
        return self.actual == self.predicted


@dataclass(frozen=True)
class InferenceTiming:
    average_wall_ns: float
    average_cpu_ns: float


@dataclass(frozen=True)
class MethodResources:
    detector_python_bytes: int
    peak_working_python_bytes: int


def latency_probe_examples(examples: Iterable[LabeledExample], topic: str, size: int = 10) -> list[LabeledExample]:
    """Choose a reproducible, roughly balanced timing sample for one topic."""
    all_examples = list(examples)
    positives = [example for example in all_examples if example.label_for(topic)]
    negatives = [example for example in all_examples if not example.label_for(topic)]
    positive_count = min(len(positives), (size + 1) // 2)
    selected = positives[:positive_count] + negatives[: size - positive_count]
    if len(selected) < size:
        selected += (positives[positive_count:] + negatives[size - positive_count:])[: size - len(selected)]
    if not selected:
        raise ValueError("Cannot time an empty dataset.")
    return selected


def average_latency_ns(detector: Detector, examples: Iterable[LabeledExample]) -> float:
    """Average inference-only latency; detector construction is intentionally excluded."""
    return average_inference_timing(detector, examples).average_wall_ns


def average_inference_timing(detector: Detector, examples: Iterable[LabeledExample]) -> InferenceTiming:
    """Wall and CPU time per prediction; detector construction is intentionally excluded."""
    sample = list(examples)
    if not sample:
        raise ValueError("Cannot time an empty sample.")
    wall_start = perf_counter_ns()
    cpu_start = process_time_ns()
    for example in sample:
        detector.predict(example.text)
    return InferenceTiming(
        average_wall_ns=(perf_counter_ns() - wall_start) / len(sample),
        average_cpu_ns=(process_time_ns() - cpu_start) / len(sample),
    )


def profile_method_resources(topic: TopicDefinition, candidate: Candidate, examples: Iterable[LabeledExample]) -> MethodResources:
    """Measure Python allocations for one method; no GPU or model storage is used."""
    sample = list(examples)
    if not sample:
        raise ValueError("Cannot profile an empty sample.")
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        detector = Detector(topic, candidate)
        after_detector, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for example in sample:
            detector.predict(example.text)
        _, peak = tracemalloc.get_traced_memory()
        return MethodResources(
            detector_python_bytes=max(after_detector - baseline, 0),
            peak_working_python_bytes=max(peak - after_detector, 0),
        )
    finally:
        tracemalloc.stop()


def load_jsonl_dataset(path: str | Path) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not isinstance(record.get("labels"), dict):
                raise ValueError(f"Invalid dataset record at line {line_number}.")
            examples.append(LabeledExample(text=record["text"], labels=record["labels"]))
    if not examples:
        raise ValueError("Dataset is empty.")
    return examples


def measure(detector: Detector, examples: Iterable[LabeledExample], topic: str) -> Metrics:
    tp = tn = fp = fn = 0
    for example in examples:
        actual, predicted = example.label_for(topic), detector.predict(example.text)
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return Metrics(tp, tn, fp, fn)


def score_examples(detector: Detector, examples: Iterable[LabeledExample], topic: str) -> list[ScoredExample]:
    """Return auditable per-prompt results for a fixed topic configuration."""
    scored: list[ScoredExample] = []
    for example in examples:
        positive, negative = detector.evidence(example.text)
        score = positive - (detector.candidate.negative_weight * negative)
        actual = example.label_for(topic)
        scored.append(
            ScoredExample(
                text=example.text,
                actual=actual,
                predicted=score >= detector.candidate.threshold,
                score=score,
                positive_evidence=positive,
                negative_evidence=negative,
            )
        )
    return scored


def _measure_evidence(
    evidence: Iterable[tuple[tuple[float, float], bool]], candidate: Candidate
) -> Metrics:
    """Measure threshold/weight variants from already-computed phrase similarity."""
    tp = tn = fp = fn = 0
    for (positive, negative), actual in evidence:
        predicted = positive - (candidate.negative_weight * negative) >= candidate.threshold
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return Metrics(tp, tn, fp, fn)


def benchmark(
    topic: TopicDefinition,
    examples: list[LabeledExample],
    *,
    min_recall: float = 0.6,
    beta: float = 1.0,
    candidates: Iterable[Candidate] | None = None,
) -> list[BenchmarkResult]:
    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be between 0 and 1.")
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")
    candidate_list = list(candidates or candidate_space())
    by_similarity: dict[tuple[str, int | None], list[Candidate]] = defaultdict(list)
    for candidate in candidate_list:
        by_similarity[(candidate.method, candidate.ngram_size)].append(candidate)

    results: list[BenchmarkResult] = []
    for (method, ngram_size), variants in by_similarity.items():
        representative = Candidate(method, negative_weight=1.0, threshold=0.0, ngram_size=ngram_size)
        detector = Detector(topic, representative)
        evidence = [(detector.evidence(example.text), example.label_for(topic.name)) for example in examples]
        results.extend(BenchmarkResult(candidate, _measure_evidence(evidence, candidate), beta) for candidate in variants)
    # First honor the required recall; then prioritize avoiding unrelated matches.
    return sorted(
        results,
        key=lambda result: (
            result.metrics.recall >= min_recall,
            result.metrics.precision if result.metrics.recall >= min_recall else result.f_beta,
            result.f_beta,
            result.metrics.recall,
            -result.metrics.false_positive_rate,
        ),
        reverse=True,
    )


def best_per_method(results: Iterable[BenchmarkResult]) -> list[BenchmarkResult]:
    """Keep the highest-ranked configuration for each method and n-gram size."""
    leaders: list[BenchmarkResult] = []
    seen: set[tuple[str, int | None]] = set()
    for result in results:
        key = (result.candidate.method, result.candidate.ngram_size)
        if key not in seen:
            leaders.append(result)
            seen.add(key)
    return leaders
