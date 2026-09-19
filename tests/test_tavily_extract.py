import io
import urllib.error

import xword.tavily as tavily_mod
from xword.tavily import extract_words, query_for, search_clue


def test_extract_keeps_pattern_hits():
    text = "The answer is CAT, also maybe DOG or ARE."
    assert extract_words(text, "...") == ["CAT", "DOG", "ARE"]
    assert extract_words(text, "C..") == ["CAT"]


def test_query_includes_length():
    q = query_for("Whiskered house pet", "...")
    assert "3 letters" in q
    assert "Whiskered house pet" in q


def test_paygo_limit_stops_further_http(monkeypatch):
    tavily_mod.exhausted = False
    calls = {"n": 0}

    def boom(_request, timeout=20):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            tavily_mod.TAVILY_URL,
            433,
            "Forbidden",
            None,
            io.BytesIO(b'{"detail":{"error":"pay-as-you-go limit"}}'),
        )

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen", boom)
    assert search_clue("Whiskered pet", "...") == []
    assert tavily_mod.exhausted is True
    assert search_clue("Another clue", "...") == []
    assert calls["n"] == 1
    tavily_mod.exhausted = False
