"""Race search-jev against the LangGraph baseline on the same puzzle ids."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OLD = "https://nebius-xword.vercel.app"
NEW = "https://xword-lime.vercel.app"
PUZZLES = ["example_3x3", "example_mini_5x5", "example_5x5_b"]


def _sse(url: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    done = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = ""
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            while "\n\n" in buf:
                part, buf = buf.split("\n\n", 1)
                line = "".join(
                    ln[5:].strip() for ln in part.split("\n") if ln.startswith("data:")
                )
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") in {"done", "result", "error"}:
                    done = event
                    break
            if done is not None:
                break
    wall_ms = int((time.perf_counter() - started) * 1000)
    if done is None:
        raise RuntimeError(f"no terminal event from {url}")
    done["client_wall_ms"] = wall_ms
    return done


def bench_old(puzzle_id: str) -> dict:
    event = _sse(
        f"{OLD}/api/solve/stream",
        {
            "puzzle_id": puzzle_id,
            "solver": "llm",
            "model": "deepseek-ai/DeepSeek-V4-Pro",
        },
    )
    stats = event.get("stats") or event
    score = event.get("score") or {}
    tokens = int(stats.get("tokens") or 0)
    return {
        "agent": "langgraph-tf",
        "puzzle": puzzle_id,
        "wall_ms": event["client_wall_ms"],
        "solved": bool((score or {}).get("solved")),
        "letter_accuracy": (score or {}).get("letter_accuracy"),
        "tokens": tokens,
        "tf_tokens": tokens,
        "tavily_calls": 0,
        "jev_calls": 0,
        "error": event.get("message") if event.get("event") == "error" else None,
    }


def bench_new(puzzle_id: str) -> dict:
    event = _sse(f"{NEW}/api/solve/stream", {"puzzle_id": puzzle_id})
    score = event.get("score") or {}
    return {
        "agent": "search-jev",
        "puzzle": puzzle_id,
        "wall_ms": event["client_wall_ms"],
        "solved": bool((score or {}).get("solved")),
        "letter_accuracy": (score or {}).get("letter_accuracy"),
        "tokens": int(event.get("tf_tokens") or 0),
        "tf_tokens": int(event.get("tf_tokens") or 0),
        "tavily_calls": int(event.get("tavily_calls") or 0),
        "jev_calls": int(event.get("jev_calls") or 0),
        "jev_input_tokens": int(event.get("jev_input_tokens") or 0),
        "error": event.get("message") if event.get("event") == "error" else None,
    }


def cost(row: dict) -> float:
    # Listed Token Factory DeepSeek V4 Pro ~$1.75/M in; treat tokens as input-heavy.
    tf = (row.get("tf_tokens") or 0) / 1_000_000 * 1.75
    jev = (row.get("jev_input_tokens") or 0) / 1_000_000 * 0.042
    tavily = (row.get("tavily_calls") or 0) * 0.008
    return tf + jev + tavily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles", nargs="*", default=PUZZLES)
    parser.add_argument("--skip-old", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    rows = []
    for puzzle_id in args.puzzles:
        print(f"\n=== {puzzle_id} ===")
        if not args.skip_old:
            try:
                old = bench_old(puzzle_id)
            except Exception as exc:  # noqa: BLE001 — bench must record a failed leg
                old = {"agent": "langgraph-tf", "puzzle": puzzle_id, "error": str(exc)[:200]}
            old["est_usd"] = cost(old) if not old.get("error") else None
            rows.append(old)
            print("old", {k: old.get(k) for k in ("wall_ms", "solved", "tf_tokens", "est_usd", "error")})
        try:
            new = bench_new(puzzle_id)
        except Exception as exc:  # noqa: BLE001 — bench must record a failed leg
            new = {"agent": "search-jev", "puzzle": puzzle_id, "error": str(exc)[:200]}
        new["est_usd"] = cost(new) if not new.get("error") else None
        rows.append(new)
        print("new", {k: new.get(k) for k in ("wall_ms", "solved", "tf_tokens", "tavily_calls", "jev_calls", "est_usd", "error")})
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
