"""Tavily search for crossword clue lookup (Nebius Blueprint grounding layer)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT
from .solver import matches

TAVILY_URL = "https://api.tavily.com/search"
WORD_RE = re.compile(r"\b[A-Za-z]{2,15}\b")
SearchFn = Callable[[str], list[str]]
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


def search_clue(clue: str, pattern: str, *, max_results: int = 5) -> list[str]:
    """Return pattern-fitting words mentioned in Tavily snippets."""
    load_env()
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    payload = {
        "api_key": key,
        "query": query_for(clue, pattern),
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    request = urllib.request.Request(
        TAVILY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data: dict[str, Any] = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Tavily HTTP {exc.code}: {detail}") from exc

    chunks = [str(data.get("answer") or "")]
    for hit in data.get("results") or []:
        chunks.append(str(hit.get("title") or ""))
        chunks.append(str(hit.get("content") or ""))
    return extract_words(" ".join(chunks), pattern)
