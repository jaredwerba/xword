"""Token Factory: leftover wordplay guesses only."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT
from .solver import matches

TF_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
TF_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")


def guess_words(clue: str, pattern: str, k: int = 8) -> tuple[list[str], int]:
    """Return (guesses, total_tokens)."""
    load_env()
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    from openai import OpenAI

    prompt = (
        "You are filling one crossword slot.\n"
        f"Clue: {clue}\n"
        f"Pattern: {pattern} (. = unknown)\n"
        f"Length: {len(pattern)}\n"
        "Return only JSON {\"guesses\": [\"WORD\", ...]} with "
        f"{k} distinct uppercase letter-only answers of that exact length "
        "that fit the pattern."
    )
    client = OpenAI(base_url=TF_BASE_URL, api_key=key)
    response = client.chat.completions.create(
        model=os.getenv("NEBIUS_MODEL") or TF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens = int((getattr(usage, "total_tokens", 0) or 0) if usage else 0)
    try:
        match = JSON_RE.search(text)
        data: dict[str, Any] = json.loads(match.group(0) if match else text)
        raw = [str(g).strip().upper() for g in data.get("guesses") or []]
    except (ValueError, AttributeError):
        raw = []
    guesses = [g for g in raw if g.isalpha() and matches(pattern, g)]
    return list(dict.fromkeys(guesses)), tokens
