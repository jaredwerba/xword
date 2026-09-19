"""Stamp independent 3x3 minis into an NxN tiled crossword (N multiple of 4)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from xword.grid import Grid, puzzle_from_mapping
from xword.paths import PUZZLES_DIR, WORDLIST_PATH
from xword.solver import fill_grid, verify_puzzle

MINI = ["...", "...", "..."]


def band_row(n: int) -> str:
    return ("..." + "#") * (n // 4)


def template_n(n: int) -> list[str]:
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    street = "#" * n
    return ([band_row(n)] * 3 + [street]) * (n // 4)


def stamp(n: int, seed: int) -> Grid:
    words = [w for w in WORDLIST_PATH.read_text().split() if len(w) == 3]
    big = Grid(template_n(n))
    bands = n // 4
    for br in range(bands):
        for bc in range(bands):
            mini = Grid(MINI)
            rng = random.Random(seed + br * 1000 + bc)
            if not fill_grid(mini, words, rng=rng, max_steps=50_000):
                raise RuntimeError(f"could not fill mini at {br},{bc}")
            r0, c0 = br * 4, bc * 4
            for r in range(3):
                for c in range(3):
                    big.cells[r0 + r][c0 + c] = mini.cells[r][c]
    return big


def unique_answers(grid: Grid) -> dict[str, str]:
    return {sid: grid.slot_pattern(sid) for sid in grid.slots}


def load_known_clues() -> dict[str, str]:
    known: dict[str, str] = {}
    path = PUZZLES_DIR / "race_16x16.json"
    if not path.exists():
        return known
    doc = json.loads(path.read_text())
    g = puzzle_from_mapping(doc).make_grid()
    g.set_rows(doc["solution"])
    for sid, slot in g.slots.items():
        known[g.slot_pattern(sid)] = slot.clue
    return known


def clue_unknown(words: list[str]) -> dict[str, str]:
    if not words:
        return {}
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
    )
    out: dict[str, str] = {}
    for i in range(0, len(words), 40):
        batch = words[i : i + 40]
        listing = "\n".join(batch)
        prompt = (
            "Write one short crossword clue per answer. Never include the answer. "
            "Under 40 characters.\n"
            f"{listing}\n"
            'JSON only: {"clues": {"CAT": "Whiskered pet", ...}}'
        )
        response = client.chat.completions.create(
            model=os.getenv("NEBIUS_MODEL") or "deepseek-ai/DeepSeek-V4-Pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1800,
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start : end + 1])
        part = data.get("clues") or data
        for word in batch:
            clue = part.get(word) or part.get(word.title())
            out[word] = str(clue or f"Three letters, starts with {word[0]}").replace(
                word, "___"
            )
    return out


def write_puzzle(n: int, seed: int = 16) -> Path:
    grid = stamp(n, seed)
    answers = unique_answers(grid)
    by_word = load_known_clues()
    missing = sorted({word for word in answers.values() if word not in by_word})
    print(f"{n}x{n} slots={len(answers)} unique_words={len(set(answers.values()))} need_clues={len(missing)}")
    by_word.update(clue_unknown(missing))
    clues = {"across": {}, "down": {}}
    for sid, word in answers.items():
        slot = grid.slots[sid]
        clues[slot.direction][str(slot.number)] = by_word.get(
            word, f"Three letters, starts with {word[0]}"
        )
    doc = {
        "id": f"race_{n}x{n}",
        "title": f"{n}×{n} tiled minis",
        "grid": list(grid.template),
        "clues": clues,
        "solution": grid.render().split("\n"),
    }
    puzzle = puzzle_from_mapping(doc)
    problems = verify_puzzle(puzzle)
    if problems:
        raise RuntimeError(problems[:8])
    out = PUZZLES_DIR / f"race_{n}x{n}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", out)
    return out


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [32, 64]
    for n in sizes:
        write_puzzle(n)
