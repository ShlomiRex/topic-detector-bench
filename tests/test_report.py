from topic_detector_bench.benchmark import BenchmarkResult, Metrics, ScoredExample
from topic_detector_bench.methods import Candidate
from topic_detector_bench.report import TopicReport, render_html


def test_html_report_escapes_prompt_text_and_includes_failures() -> None:
    result = BenchmarkResult(Candidate("token_jaccard", 1.0, 0.1), Metrics(1, 0, 1, 0), 1.0)
    report = TopicReport("example", result, (ScoredExample("<unsafe>", False, True, 0.4, 0.4, 0.0),))
    html = render_html([report], min_recall=0.6, beta=1.0)
    assert "&lt;unsafe&gt;" in html
    assert "Needs review — incorrect test predictions (1)" in html
    assert "Negative weight" in html


def test_html_report_uses_held_out_test_metrics_when_available() -> None:
    result = BenchmarkResult(Candidate("token_jaccard", 1.0, 0.1), Metrics(1, 1, 0, 0), 1.0)
    test_example = ScoredExample("test", True, False, 0.0, 0.0, 0.0)
    report = TopicReport("example", result, (), Metrics(0, 1, 0, 1), (test_example,))
    html = render_html([report], min_recall=0.6, beta=1.0)
    assert "Test recall" in html
    assert "incorrect test predictions (1)" in html
