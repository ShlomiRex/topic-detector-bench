from topic_detector_bench.benchmark import BenchmarkResult, Metrics, ScoredExample
from topic_detector_bench.methods import Candidate
from topic_detector_bench.report import TopicReport, render_html


def test_html_report_escapes_prompt_text_and_includes_failures() -> None:
    result = BenchmarkResult(Candidate("token_jaccard", 1.0, 0.1), Metrics(1, 0, 1, 0), 1.0)
    report = TopicReport("example", result, (ScoredExample("<unsafe>", False, True, 0.4, 0.4, 0.0),))
    html = render_html([report], min_recall=0.6, beta=1.0)
    assert "&lt;unsafe&gt;" in html
    assert "Needs review — incorrect predictions (1)" in html
    assert "Negative weight" in html
