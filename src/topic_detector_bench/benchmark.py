from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
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
    results = [
        BenchmarkResult(candidate, measure(Detector(topic, candidate), examples, topic.name), beta)
        for candidate in (candidates or candidate_space())
    ]
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
