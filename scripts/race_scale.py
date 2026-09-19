"""Race search-jev vs LangGraph on a scale fixture (16/32/64)."""

from __future__ import annotations

import json
import os
import sys
import threading
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
# Scale races blow the LangSmith unique-traces cap. Keep demo tracing on
# for 3x3/5x5; force it off here even if .env says true.
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from eval.metrics import score_grid

TF_USD_PER_M = 1.75
JEV_USD_PER_M = 0.042
TAVILY_USD = 0.008


def costs(row: dict) -> dict:
    tf = (row.get("tf_tokens") or 0) / 1_000_000 * TF_USD_PER_M
    jev = (row.get("jev_input_tokens") or 0) / 1_000_000 * JEV_USD_PER_M
    tavily = (row.get("tavily_calls") or 0) * TAVILY_USD
    return {
        "usd_tf": round(tf, 6),
        "usd_jev": round(jev, 6),
        "usd_tavily": round(tavily, 6),
        "usd_total": round(tf + jev + tavily, 6),
    }


def _heartbeat(label: str, get_counts, stop: threading.Event) -> None:
    while not stop.wait(30):
        print(f"... {label} {get_counts()}", flush=True)


def race_new(path: Path, max_steps: int) -> dict:
    from xword.agent import SearchJevSolver
    from xword.grid import load_puzzle

    puzzle = load_puzzle(path)
    solver = SearchJevSolver(verbose=False, max_steps=max_steps)
    stop = threading.Event()
    threading.Thread(
        target=_heartbeat,
        args=(
            "search-jev",
            lambda: (
                f"tavily={solver.tavily_calls} jev={solver.jev_calls} "
                f"tf={solver.tf_calls} tok={solver.tf_tokens}"
            ),
            stop,
        ),
        daemon=True,
    ).start()
    started = time.perf_counter()
    try:
        result = solver.solve(puzzle)
    finally:
        stop.set()
    wall_ms = int((time.perf_counter() - started) * 1000)
    score = score_grid(result.grid, puzzle.solution)
    row = {
        "agent": "search-jev",
        "puzzle": puzzle.id,
        "slots": len(puzzle.make_grid().slots),
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
    row.update(costs(row))
    return row


def race_old(path: Path, max_turns: int) -> dict:
    from nebius_xword.agent import CrosswordAgent
    from nebius_xword.grid import load_puzzle as load_old

    puzzle = load_old(path)
    agent = CrosswordAgent(
        model="deepseek-ai/DeepSeek-V4-Pro",
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
        max_turns=max_turns,
        verbose=False,
        history_window=48,
    )
    stop = threading.Event()
    threading.Thread(
        target=_heartbeat,
        args=(
            "langgraph-tf",
            lambda: f"turns~ max={max_turns} tok={getattr(agent, 'total_tokens', '?')}",
            stop,
        ),
        daemon=True,
    ).start()
    started = time.perf_counter()
    try:
        result = agent.solve(puzzle)
    finally:
        stop.set()
    wall_ms = int((time.perf_counter() - started) * 1000)
    score = score_grid(result.grid, puzzle.solution)
    row = {
        "agent": "langgraph-tf",
        "puzzle": puzzle.id,
        "slots": len(puzzle.make_grid().slots),
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
    row.update(costs(row))
    return row


def main() -> None:
    size = int(sys.argv[1])
    who = sys.argv[2] if len(sys.argv) > 2 else "both"
    path = ROOT / "data" / "puzzles" / f"race_{size}x{size}.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run scripts/make_scale_puzzle.py {size}")
    out = ROOT / "eval" / f"bench-{size}x{size}.json"
    existing = json.loads(out.read_text()) if out.exists() else {"rows": []}
    by_agent = {r["agent"]: r for r in existing.get("rows", [])}
    if who in {"new", "both"}:
        print(f"=== search-jev {size}x{size} ===", flush=True)
        row = race_new(path, max_steps=max(200, size * 8))
        print(json.dumps(row, indent=2), flush=True)
        by_agent[row["agent"]] = row
    if who in {"old", "both"}:
        print(f"=== langgraph-tf {size}x{size} ===", flush=True)
        # 32x32 needed 82 turns for 64 minis (~1.3/mini). 64x64 has 256 minis.
        row = race_old(path, max_turns=max(80, size * 6))
        print(json.dumps(row, indent=2), flush=True)
        by_agent[row["agent"]] = row
    payload = {"puzzle": f"race_{size}x{size}", "rows": list(by_agent.values())}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
