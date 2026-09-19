"""Eval harness: empty / oracle / backtrack / search-jev.

Usage (repo root):
    python -m eval.run_eval --solver oracle
    python -m eval.run_eval --solver search-jev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import score_grid
from xword.grid import Grid, Puzzle, load_puzzle
from xword.paths import PUZZLES_DIR, WORDLIST_PATH
from xword.solver import fill_grid


def solve_empty(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    return puzzle.make_grid(), {}


def solve_oracle(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    grid = puzzle.make_grid()
    grid.set_rows(puzzle.solution)
    return grid, {}


def solve_backtrack(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    grid = puzzle.make_grid()
    fill_grid(grid, ctx["words"], rng=random.Random(run))
    return grid, {}


def solve_search_jev(puzzle: Puzzle, run: int, ctx: dict) -> tuple[Grid, dict]:
    started = time.perf_counter()
    result = ctx["agent"].solve(puzzle)
    return result.grid, {
        "wall_ms": result.wall_ms or int((time.perf_counter() - started) * 1000),
        "jev_calls": result.jev_calls,
        "tavily_calls": result.tavily_calls,
        "tf_calls": result.tf_calls,
        "tf_tokens": result.tf_tokens,
        "jev_input_tokens": result.jev_input_tokens,
        "steps": result.steps,
    }


SOLVERS = {
    "empty": solve_empty,
    "oracle": solve_oracle,
    "backtrack": solve_backtrack,
    "search-jev": solve_search_jev,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", default=str(PUZZLES_DIR))
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="oracle")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ctx: dict = {}
    if args.solver == "backtrack":
        ctx["words"] = WORDLIST_PATH.read_text().split()
    if args.solver == "search-jev":
        from xword.agent import SearchJevSolver

        ctx["agent"] = SearchJevSolver(verbose=True)

    paths = sorted(Path(args.puzzles).glob("*.json"))
    if not paths:
        raise SystemExit(f"no puzzles in {args.puzzles}")

    results = []
    for path in paths:
        puzzle = load_puzzle(path)
        if puzzle.solution is None:
            print(f"skipping {puzzle.id}: no answer key")
            continue
        runs = []
        for run in range(args.runs):
            grid, stats = SOLVERS[args.solver](puzzle, run, ctx)
            runs.append({**score_grid(grid, puzzle.solution), **stats})
        n = len(runs)
        row = {
            "puzzle": puzzle.id,
            "runs": n,
            "letter_accuracy": sum(r["letter_accuracy"] for r in runs) / n,
            "word_accuracy": sum(r["word_accuracy"] for r in runs) / n,
            "solved": sum(1 for r in runs if r["solved"]),
        }
        for key in ("wall_ms", "jev_calls", "tavily_calls", "tf_calls", "tf_tokens", "steps"):
            if any(key in r for r in runs):
                row[key] = sum(r.get(key, 0) for r in runs) / n
        results.append(row)

    payload = {"solver": args.solver, "runs": args.runs, "results": results}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"solver: {args.solver} | runs: {args.runs}\n")
    print(f"{'puzzle':<22} {'letters':>8} {'words':>8} {'solved':>8}")
    for row in results:
        print(
            f"{row['puzzle']:<22} {row['letter_accuracy']:>7.0%} "
            f"{row['word_accuracy']:>8.0%} {row['solved']:>5}/{row['runs']}"
        )
        extras = []
        if "wall_ms" in row:
            extras.append(f"{row['wall_ms']:.0f}ms")
        if "tf_tokens" in row:
            extras.append(f"tf_tokens={row['tf_tokens']:.0f}")
        if "tavily_calls" in row:
            extras.append(f"tavily={row['tavily_calls']:.0f}")
        if "jev_calls" in row:
            extras.append(f"jev={row['jev_calls']:.0f}")
        if extras:
            print("  " + " ".join(extras))


if __name__ == "__main__":
    main()
