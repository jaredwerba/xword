"""Token Factory: leftover wordplay guesses only."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT
from .solver import matches
from .traces import maybe_traceable

TF_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
TF_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")


@maybe_traceable("tf.guess")
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


@maybe_traceable("tf.fill_region")
def fill_region(slots: list[dict[str, str]]) -> tuple[dict[str, str], int]:
    """Fill leftover slots in one component with a single Token Factory call."""
    load_env()
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    from openai import OpenAI

    listing = "\n".join(
        f"{s['id']}: clue={s['clue']!r} pattern={s['pattern']} length={s['length']}"
        for s in slots
    )
    prompt = (
        "Fill these remaining crossword slots. Patterns use . for unknowns. "
        "Answers must be uppercase letters matching the pattern and length. "
        "Crossings must agree. Do not restrict answers to a short-word list; "
        "theme entries and 6+ letter words are expected.\n"
        f"{listing}\n"
        'Return only JSON {"fills": {"1A": "CAT", ...}}'
    )
    # One JSON object per leftover slot; 400 tokens is too small for a 15x15.
    max_tokens = max(800, 60 * max(len(slots), 1))
    client = OpenAI(base_url=TF_BASE_URL, api_key=key)
    response = client.chat.completions.create(
        model=os.getenv("NEBIUS_MODEL") or TF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens = int((getattr(usage, "total_tokens", 0) or 0) if usage else 0)
    try:
        match = JSON_RE.search(text)
        data: dict[str, Any] = json.loads(match.group(0) if match else text)
        raw = data.get("fills") or {}
    except (ValueError, AttributeError):
        raw = {}
    fills = {
        str(sid).upper(): str(word).strip().upper()
        for sid, word in raw.items()
        if str(word).strip().isalpha()
    }
    return fills, tokens
