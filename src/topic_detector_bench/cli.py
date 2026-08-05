from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark, load_jsonl_dataset, measure, score_examples
from .methods import Candidate, Detector
from .models import TopicDefinition
from .report import TopicReport, render_html


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


def _benchmark_all_command(args: argparse.Namespace) -> int:
    validation_dataset = load_jsonl_dataset(args.dataset)
    test_dataset = load_jsonl_dataset(args.test_dataset) if args.test_dataset else validation_dataset
    topic_paths = sorted(Path(args.topics_dir).glob("*.yaml"))
    if not topic_paths:
        raise ValueError(f"No YAML topics found in {args.topics_dir}.")

    print("topic                    method                 test precision  test recall  F-beta  FP  FN")
    reports: list[TopicReport] = []
    for topic_path in topic_paths:
        topic = TopicDefinition.from_file(topic_path)
        best = benchmark(topic, validation_dataset, min_recall=args.min_recall, beta=args.beta)[0]
        candidate = best.candidate
        detector = Detector(topic, candidate)
        test_metrics = measure(detector, test_dataset, topic.name)
        method = candidate.method if candidate.ngram_size is None else f"{candidate.method}:{candidate.ngram_size}"
        print(
            f"{topic.name:<24} {method:<22} {test_metrics.precision:>12.1%}  {test_metrics.recall:>9.1%}"
            f"  {test_metrics.f_beta(args.beta):>6.1%}  {test_metrics.false_positive:>2}  {test_metrics.false_negative:>2}"
        )
        reports.append(
            TopicReport(
                topic.name,
                best,
                tuple(score_examples(detector, validation_dataset, topic.name)),
                test_metrics,
                tuple(score_examples(detector, test_dataset, topic.name)),
            )
        )
    if args.html_report:
        report_path = Path(args.html_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_html(reports, args.min_recall, args.beta), encoding="utf-8")
        print(f"\nSaved HTML report to {report_path}")
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
    all_parser = commands.add_parser("benchmark-all", help="Select a detector for every YAML topic in a directory.")
    all_parser.add_argument("--topics-dir", required=True)
    all_parser.add_argument("--dataset", required=True)
    all_parser.add_argument("--test-dataset", help="Held-out JSONL data; never used to select a configuration")
    all_parser.add_argument("--min-recall", type=float, default=0.6)
    all_parser.add_argument("--beta", type=float, default=1.0)
    all_parser.add_argument("--html-report", help="Write an inspectable HTML report for every topic")
    all_parser.set_defaults(handler=_benchmark_all_command)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
