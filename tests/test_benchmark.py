from topic_detector_bench.benchmark import average_inference_timing, average_latency_ns, benchmark, latency_probe_examples, profile_method_resources
from topic_detector_bench.methods import Candidate, Detector
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


def test_unlisted_topic_label_is_a_negative_label() -> None:
    example = LabeledExample("unrelated", {"other_topic": True})
    assert example.label_for("x") is False


def test_latency_probe_and_average_latency_use_the_requested_sample() -> None:
    topic = TopicDefinition.from_mapping({"topic": "x", "positive": ["match"], "negative": ["ignore"]})
    data = [LabeledExample("match", {"x": True})] + [LabeledExample(f"ignore {i}", {}) for i in range(12)]
    sample = latency_probe_examples(data, "x")
    assert len(sample) == 10
    assert average_latency_ns(Detector(topic, Candidate("token_jaccard", 1.0, 0.0)), sample) >= 0
    assert average_inference_timing(Detector(topic, Candidate("token_jaccard", 1.0, 0.0)), sample).average_cpu_ns >= 0
    assert profile_method_resources(topic, Candidate("token_jaccard", 1.0, 0.0), sample).detector_python_bytes >= 0
