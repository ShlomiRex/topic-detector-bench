from topic_detector_bench.methods import Candidate, Detector
from topic_detector_bench.models import TopicDefinition


def test_negative_evidence_can_veto_a_related_but_benign_prompt() -> None:
    topic = TopicDefinition.from_mapping({
        "topic": "fraud",
        "positive": ["buy stolen credit card details"],
        "negative": ["protect myself from credit card fraud"],
    })
    detector = Detector(topic, Candidate("token_jaccard", negative_weight=1.5, threshold=0.1))

    assert detector.predict("buy stolen credit card details online")
    assert not detector.predict("how do I protect myself from credit card fraud")


def test_tfidf_and_bm25_methods_are_available_without_a_model() -> None:
    topic = TopicDefinition.from_mapping({
        "topic": "fraud",
        "positive": ["buy stolen credit card details"],
        "negative": ["protect credit card from fraud"],
    })
    for method in ("tfidf_token_cosine", "bm25_token"):
        detector = Detector(topic, Candidate(method, negative_weight=1.0, threshold=0.1))
        assert detector.evidence("buy stolen card details")[0] > 0
