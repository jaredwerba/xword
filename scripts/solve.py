"""Solve one puzzle with search-jev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from xword.agent import SearchJevSolver
from xword.grid import load_puzzle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle")
    args = parser.parse_args()
    puzzle = load_puzzle(args.puzzle)
    result = SearchJevSolver(verbose=True).solve(puzzle)
    print(f"\n{puzzle.title}\n{result.grid.render()}")
    print(
        f"submitted={result.submitted} steps={result.steps} "
        f"wall_ms={result.wall_ms} tavily={result.tavily_calls} "
        f"jev={result.jev_calls} tf_tokens={result.tf_tokens}"
    )
    if puzzle.solution:
        from eval.metrics import score_grid

        print("score", score_grid(result.grid, puzzle.solution))


if __name__ == "__main__":
    main()
