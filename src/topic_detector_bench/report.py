from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

from .benchmark import BenchmarkResult, Metrics, ScoredExample


@dataclass(frozen=True)
class TopicReport:
    topic: str
    result: BenchmarkResult
    examples: tuple[ScoredExample, ...]
    test_metrics: Metrics | None = None
    test_examples: tuple[ScoredExample, ...] = ()
    method_comparisons: tuple["MethodComparison", ...] = ()
    configuration_comparisons: tuple["ConfigurationComparison", ...] = ()


@dataclass(frozen=True)
class MethodComparison:
    result: BenchmarkResult
    test_metrics: Metrics


@dataclass(frozen=True)
class ConfigurationComparison:
    result: BenchmarkResult
    average_latency_ns: float
    average_cpu_ns: float = 0.0
    detector_python_bytes: int = 0
    peak_working_python_bytes: int = 0
    configuration_storage_bytes: int = 0


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def _bytes(value: int) -> str:
    if value < 1_024:
        return f"{value} B"
    if value < 1_024 * 1_024:
        return f"{value / 1_024:.1f} KB"
    return f"{value / (1_024 * 1_024):.2f} MB"


def _example_rows(examples: Iterable[ScoredExample]) -> str:
    rows: list[str] = []
    for example in examples:
        outcome = "correct" if example.correct else "incorrect"
        expected = "match" if example.actual else "no match"
        predicted = "match" if example.predicted else "no match"
        rows.append(
            "<tr>"
            f'<td class="prompt">{escape(example.text)}</td>'
            f"<td>{expected}</td><td>{predicted}</td>"
            f"<td>{example.score:.3f}</td><td>{example.positive_evidence:.3f}</td>"
            f"<td>{example.negative_evidence:.3f}</td><td class=\"{outcome}\">{outcome}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"7\">None</td></tr>"


def _table(title: str, examples: Iterable[ScoredExample], open_by_default: bool = False) -> str:
    open_attribute = " open" if open_by_default else ""
    return (
        f"<details{open_attribute}><summary>{escape(title)}</summary>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Prompt</th><th>Expected</th><th>Predicted</th><th>Score</th>"
        "<th>Positive</th><th>Negative</th><th>Result</th>"
        "</tr></thead><tbody>"
        f"{_example_rows(examples)}</tbody></table></div></details>"
    )


def _method_comparison_table(comparisons: Iterable[MethodComparison], winner: BenchmarkResult) -> str:
    rows: list[str] = []
    for comparison in comparisons:
        result, metrics = comparison.result, comparison.test_metrics
        candidate = result.candidate
        method = candidate.method if candidate.ngram_size is None else f"{candidate.method} (n={candidate.ngram_size})"
        marker = " <b>Winner</b>" if candidate == winner.candidate else ""
        rows.append(
            "<tr>"
            f"<td>{escape(method)}{marker}</td><td>{candidate.negative_weight:.2f}</td><td>{candidate.threshold:.2f}</td>"
            f"<td>{_percentage(result.metrics.precision)}</td><td>{_percentage(result.metrics.recall)}</td>"
            f"<td>{_percentage(metrics.precision)}</td><td>{_percentage(metrics.recall)}</td><td>{_percentage(metrics.f_beta(result.beta))}</td>"
            "</tr>"
        )
    return (
        "<h3>Method comparison</h3><p class=\"selection\">Each row is that method’s best configuration selected on validation; test results are then measured without retuning.</p>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Method</th><th>Negative weight</th><th>Threshold</th>"
        "<th>Validation precision</th><th>Validation recall</th><th>Test precision</th><th>Test recall</th><th>Test F-beta</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table></div>"
    )


def _configuration_latency_table(
    comparisons: Iterable[ConfigurationComparison], winner: BenchmarkResult
) -> str:
    rows: list[str] = []
    for comparison in sorted(comparisons, key=lambda item: item.average_latency_ns):
        result = comparison.result
        candidate, metrics = result.candidate, result.metrics
        method = candidate.method if candidate.ngram_size is None else f"{candidate.method} (n={candidate.ngram_size})"
        marker = " <b>Winner</b>" if candidate == winner.candidate else ""
        rows.append(
            "<tr>"
            f"<td>{escape(method)}{marker}</td><td>{candidate.negative_weight:.2f}</td><td>{candidate.threshold:.2f}</td>"
            f"<td>{_percentage(metrics.precision)}</td><td>{_percentage(metrics.recall)}</td><td>{_percentage(result.f_beta)}</td>"
            f"<td>{comparison.average_latency_ns / 1_000:.2f} µs</td>"
            f"<td>{comparison.average_cpu_ns / 1_000:.2f} µs ({(comparison.average_cpu_ns / comparison.average_latency_ns * 100) if comparison.average_latency_ns else 0:.0f}%)</td>"
            f"<td>{_bytes(comparison.detector_python_bytes)} / {_bytes(comparison.peak_working_python_bytes)}</td>"
            f"<td>CPU only (0 B)</td><td>0 B model / {_bytes(comparison.configuration_storage_bytes)} config</td>"
            "</tr>"
        )
    return (
        f"<details><summary>All configurations — average inference latency on 10 prompts ({len(rows)})</summary>"
        "<p class=\"selection\">Rows are sorted fastest first. Timing excludes detector construction and includes only the configuration's evaluation loop."
        " CPU is process CPU time per prompt and its percentage of wall time (one-core utilization). RAM is Python allocations: detector-resident / peak additional working allocation."
        " These methods use no GPU and no serialized model; storage shows the configuration metadata only. Validation metrics are shown so speed and quality can be compared.</p>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Method</th><th>Negative weight</th><th>Threshold</th>"
        "<th>Validation precision</th><th>Validation recall</th><th>Validation F-beta</th><th>Wall / prompt</th><th>CPU / prompt</th>"
        "<th>Python RAM (detector / peak)</th><th>GPU</th><th>Storage</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table></div></details>"
    )


def render_html(reports: Iterable[TopicReport], min_recall: float, beta: float) -> str:
    """Render a self-contained, dependency-free HTML benchmark report."""
    sections: list[str] = []
    for report in reports:
        result, metrics = report.result, report.result.metrics
        candidate = result.candidate
        review_examples = report.test_examples or report.examples
        review_metrics = report.test_metrics or metrics
        failures = tuple(example for example in review_examples if not example.correct)
        successes = tuple(example for example in review_examples if example.correct)
        method = candidate.method
        if candidate.ngram_size is not None:
            method += f" (n={candidate.ngram_size})"
        sections.append(
            "<section>"
            f"<h2>{escape(report.topic)}</h2>"
            "<div class=\"config\">"
            f"<span><b>Method</b> {escape(method)}</span>"
            f"<span><b>Negative weight</b> {candidate.negative_weight:.2f}</span>"
            f"<span><b>Threshold</b> {candidate.threshold:.2f}</span>"
            "</div>"
            "<div class=\"metrics\">"
            f"<div><b>Test precision</b><strong>{_percentage(review_metrics.precision)}</strong></div>"
            f"<div><b>Test recall</b><strong>{_percentage(review_metrics.recall)}</strong></div>"
            f"<div><b>Test F-beta</b><strong>{_percentage(review_metrics.f_beta(result.beta))}</strong></div>"
            f"<div><b>Test false positives</b><strong>{review_metrics.false_positive}</strong></div>"
            f"<div><b>Test false negatives</b><strong>{review_metrics.false_negative}</strong></div>"
            "</div>"
            f"<p class=\"selection\">Selected on validation: precision {_percentage(metrics.precision)}, recall {_percentage(metrics.recall)}, F-beta {_percentage(result.f_beta)}.</p>"
            f"{_method_comparison_table(report.method_comparisons, result) if report.method_comparisons else ''}"
            f"{_configuration_latency_table(report.configuration_comparisons, result) if report.configuration_comparisons else ''}"
            f"{_table(f'Needs review — incorrect test predictions ({len(failures)})', failures, True)}"
            f"{_table(f'Passed — correct test predictions ({len(successes)})', successes)}"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Topic detector benchmark report</title>
<style>
:root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; color: #172033; background: #f5f7fb; }}
body {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 80px; }} h1 {{ margin-bottom: 4px; }} .subtitle {{ color: #536076; margin-top: 0; }}
section {{ background: white; border: 1px solid #dbe1ed; border-radius: 12px; margin-top: 22px; padding: 22px; box-shadow: 0 1px 2px #1720330d; }} h2 {{ margin-top: 0; }}
.config {{ display: flex; gap: 20px; flex-wrap: wrap; color: #43516a; margin-bottom: 18px; }} .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 18px; }}
.metrics div {{ background: #edf3ff; border-radius: 8px; padding: 12px; }} .metrics b {{ display: block; color: #536076; font-size: 0.78rem; }} .metrics strong {{ font-size: 1.25rem; }}
details {{ border-top: 1px solid #dbe1ed; padding: 12px 0; }} summary {{ cursor: pointer; font-weight: 650; }} .table-wrap {{ overflow-x: auto; margin-top: 12px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }} th, td {{ text-align: left; padding: 9px; border-bottom: 1px solid #e6eaf2; vertical-align: top; }} th {{ color: #536076; white-space: nowrap; }} .prompt {{ min-width: 300px; }} .correct {{ color: #087443; font-weight: 650; }} .incorrect {{ color: #b42318; font-weight: 700; }}
</style></head><body>
<h1>Topic detector benchmark</h1>
<p class="subtitle">Selection is precision-first after reaching {min_recall:.0%} validation recall. F-{beta:g} is used as a tie-breaker. Test rows were not used to choose a method or threshold. “Passed” means the prediction matched the test label.</p>
{''.join(sections)}
</body></html>"""
