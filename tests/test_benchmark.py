from topic_detector_bench.benchmark import benchmark
from topic_detector_bench.methods import Candidate
from topic_detector_bench.models import LabeledExample, TopicDefinition


def test_benchmark_prioritizes_precision_after_meeting_minimum_recall() -> None:
    topic = TopicDefinition.from_mapping({
        "topic": "x",
        "positive": ["dangerous action"],
        "negative": ["safe action"],
    })
    data = [
        LabeledExample("dangerous action", {"x": True}),
        LabeledExample("safe action", {"x": False}),
    ]
    results = benchmark(
        topic,
        data,
        min_recall=1.0,
        candidates=[
            Candidate("subphrase", negative_weight=1.0, threshold=0.5),
            Candidate("subphrase", negative_weight=1.0, threshold=0.0),
        ],
    )
    assert results[0].metrics.precision == 1.0
    assert results[0].metrics.recall == 1.0
