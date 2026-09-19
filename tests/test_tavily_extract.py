from xword.tavily import extract_words, query_for


def test_extract_keeps_pattern_hits():
    text = "The answer is CAT, also maybe DOG or ARE."
    assert extract_words(text, "...") == ["CAT", "DOG", "ARE"]
    assert extract_words(text, "C..") == ["CAT"]


def test_query_includes_length():
    q = query_for("Whiskered house pet", "...")
    assert "3 letters" in q
    assert "Whiskered house pet" in q
