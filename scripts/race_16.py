"""Race search-jev vs LangGraph V4 Pro on the 16x16 fixture."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/jkw/nebiusFDE/src")
sys.path.insert(0, "/Users/jkw/nebiusFDE")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv("/Users/jkw/nebiusFDE/.env")

from eval.metrics import score_grid
from xword.agent import SearchJevSolver
from xword.grid import load_puzzle

PUZZLE = ROOT / "data" / "puzzles" / "race_16x16.json"


def race_new(puzzle):
    started = time.perf_counter()
    result = SearchJevSolver(verbose=True, max_steps=200).solve(puzzle)
    wall_ms = int((time.perf_counter() - started) * 1000)
    score = score_grid(result.grid, puzzle.solution)
    return {
        "agent": "search-jev",
        "wall_ms": wall_ms,
        "steps": result.steps,
        "submitted": result.submitted,
        "tavily_calls": result.tavily_calls,
        "jev_calls": result.jev_calls,
        "tf_calls": result.tf_calls,
        "tf_tokens": result.tf_tokens,
        "jev_input_tokens": result.jev_input_tokens,
        **score,
    }


def race_old(puzzle):
    from nebius_xword.agent import CrosswordAgent
    from nebius_xword.grid import load_puzzle as load_old

    puzzle = load_old(PUZZLE)
    started = time.perf_counter()
    agent = CrosswordAgent(
        model="deepseek-ai/DeepSeek-V4-Pro",
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
        max_turns=120,
        verbose=True,
        history_window=48,
    )
    result = agent.solve(puzzle)
    wall_ms = int((time.perf_counter() - started) * 1000)
    score = score_grid(result.grid, puzzle.solution)
    return {
        "agent": "langgraph-tf",
        "wall_ms": wall_ms,
        "steps": result.turns,
        "submitted": result.submitted,
        "tavily_calls": 0,
        "jev_calls": 0,
        "tf_calls": result.turns,
        "tf_tokens": result.total_tokens,
        "jev_input_tokens": 0,
        **score,
    }


def main() -> None:
    puzzle = load_puzzle(PUZZLE)
    print(f"puzzle {puzzle.id} slots={len(puzzle.make_grid().slots)} 16x16")
    who = sys.argv[1] if len(sys.argv) > 1 else "both"
    rows = []
    if who in {"new", "both"}:
        print("\n=== search-jev ===")
        rows.append(race_new(puzzle))
        print(json.dumps(rows[-1], indent=2))
    if who in {"old", "both"}:
        print("\n=== langgraph-tf ===")
        rows.append(race_old(puzzle))
        print(json.dumps(rows[-1], indent=2))
    out = ROOT / "eval" / "bench-16x16.json"
    existing = json.loads(out.read_text()) if out.exists() else {"rows": []}
    by_agent = {r["agent"]: r for r in existing.get("rows", [])}
    for r in rows:
        by_agent[r["agent"]] = r
    payload = {"puzzle": "race_16x16", "rows": list(by_agent.values())}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
