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


def _percentage(value: float) -> str:
    return f"{value:.1%}"


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
