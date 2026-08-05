from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark, load_jsonl_dataset
from .methods import Candidate, Detector
from .models import TopicDefinition


def _candidate_from_file(path: str) -> Candidate:
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    candidate = data.get("candidate", data)
    return Candidate(**candidate)


def _benchmark_command(args: argparse.Namespace) -> int:
    topic = TopicDefinition.from_file(args.topic)
    results = benchmark(topic, load_jsonl_dataset(args.dataset), min_recall=args.min_recall, beta=args.beta)
    print(f"Topic: {topic.name} | candidates: {len(results)} | min recall: {args.min_recall:.0%}")
    print("method                 n  neg-wt  threshold  precision  recall  F-beta  FP  FN")
    for result in results[: args.top]:
        metric, candidate = result.metrics, result.candidate
        ngram = "-" if candidate.ngram_size is None else str(candidate.ngram_size)
        print(
            f"{candidate.method:<22} {ngram:>1}  {candidate.negative_weight:>5.2f}"
            f"  {candidate.threshold:>8.2f}  {metric.precision:>8.1%}  {metric.recall:>6.1%}"
            f"  {result.f_beta:>6.1%}  {metric.false_positive:>2}  {metric.false_negative:>2}"
        )
    if args.output:
        best = results[0].as_dict() | {"topic": topic.name, "selection": {"min_recall": args.min_recall, "beta": args.beta}}
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved recommendation to {args.output}")
    return 0


def _detect_command(args: argparse.Namespace) -> int:
    topic = TopicDefinition.from_file(args.topic)
    detector = Detector(topic, _candidate_from_file(args.recommendation))
    positive, negative = detector.evidence(args.text)
    print(json.dumps({
        "topic": topic.name,
        "match": detector.predict(args.text),
        "score": round(detector.score(args.text), 6),
        "positive_evidence": round(positive, 6),
        "negative_evidence": round(negative, 6),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deterministic topic detectors.")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = commands.add_parser("benchmark", help="Find the best configuration for one topic.")
    benchmark_parser.add_argument("--topic", required=True, help="YAML topic definition")
    benchmark_parser.add_argument("--dataset", required=True, help="JSONL labeled evaluation data")
    benchmark_parser.add_argument("--min-recall", type=float, default=0.6)
    benchmark_parser.add_argument("--beta", type=float, default=1.0, help="F-beta tie-breaker; >1 favors recall")
    benchmark_parser.add_argument("--top", type=int, default=10)
    benchmark_parser.add_argument("--output", help="Write the recommended configuration as JSON")
    benchmark_parser.set_defaults(handler=_benchmark_command)
    detect_parser = commands.add_parser("detect", help="Score text using a saved recommendation.")
    detect_parser.add_argument("--topic", required=True)
    detect_parser.add_argument("--recommendation", required=True)
    detect_parser.add_argument("--text", required=True)
    detect_parser.set_defaults(handler=_detect_command)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
