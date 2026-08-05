from topic_detector_bench.text import normalize, tokens


def test_normalize_preserves_hebrew_words_and_removes_punctuation() -> None:
    assert normalize("  איך משתמשים—בכרטיס?!  ") == "איך משתמשים בכרטיס"
    assert tokens("HELLO, world") == ("hello", "world")
