"""Tavily search for crossword clue lookup (Nebius Blueprint grounding layer)."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT
from .solver import matches
from .traces import maybe_traceable

TAVILY_URL = "https://api.tavily.com/search"
WORD_RE = re.compile(r"\b[A-Za-z]{2,15}\b")
SearchFn = Callable[[str], list[str]]
# 402/433: Tavily pay-as-you-go (or credit) cap. Further HTTP is wasted.
PAYGO_CODES = {402, 433}
exhausted = False
STOP = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "HAVE", "WERE",
    "BEEN", "THEY", "THEM", "WHAT", "WHEN", "YOUR", "WILL", "WOULD",
}


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")


def query_for(clue: str, pattern: str) -> str:
    n = len(pattern)
    return f'crossword clue "{clue}" {n} letters'


def extract_words(text: str, pattern: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in WORD_RE.findall(text or ""):
        word = raw.upper()
        if word in seen or word in STOP or not matches(pattern, word):
            continue
        seen.add(word)
        found.append(word)
    return found


@maybe_traceable("tavily.search")
def search_clue(clue: str, pattern: str, *, max_results: int = 3) -> list[str]:
    """Return pattern-fitting words mentioned in Tavily snippets."""
    global exhausted
    if exhausted:
        return []
    load_env()
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    payload = {
        "api_key": key,
        "query": query_for(clue, pattern),
        "max_results": min(max_results, 3),
        "search_depth": "basic",
        "include_answer": True,
    }
    request = urllib.request.Request(
        TAVILY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data: dict[str, Any] = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last = RuntimeError(f"Tavily HTTP {exc.code}: {detail}")
            if exc.code in PAYGO_CODES:
                exhausted = True
                return []
            if exc.code not in {429, 503} or attempt == 5:
                raise last from exc
            time.sleep(1.5 * (2**attempt))
    else:
        raise last or RuntimeError("Tavily failed")

    chunks = [str(data.get("answer") or "")]
    for hit in data.get("results") or []:
        chunks.append(str(hit.get("title") or ""))
        chunks.append(str(hit.get("content") or ""))
    return extract_words(" ".join(chunks), pattern)
