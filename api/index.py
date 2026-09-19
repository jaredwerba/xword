"""FastAPI entry for xword. Vercel rewrites every /api path here."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from eval.metrics import score_grid
from xword.agent import SearchJevSolver
from xword.grid import load_puzzle
from xword.paths import PUZZLES_DIR

PUBLIC = ROOT / "public"
app = FastAPI(title="xword", version="0.1.0")


@app.middleware("http")
async def vercel_bridge(request: Request, call_next):
    original = request.query_params.get("__path")
    if original and original.startswith("/api/"):
        request.scope["path"] = original
        request.scope["raw_path"] = original.encode()
    oidc = request.headers.get("x-vercel-oidc-token")
    if oidc:
        os.environ["VERCEL_OIDC_TOKEN"] = oidc
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(PUBLIC / "index.html")


@app.get("/nebius-logo.svg")
def logo():
    return FileResponse(PUBLIC / "nebius-logo.svg")


@app.get("/api/puzzles")
def list_puzzles():
    items = []
    for path in sorted(PUZZLES_DIR.glob("*.json")):
        puzzle = load_puzzle(path)
        items.append({"id": puzzle.id, "title": puzzle.title})
    return {"puzzles": items}


@app.get("/api/puzzles/{puzzle_id}")
def get_puzzle(puzzle_id: str):
    path = (PUZZLES_DIR / f"{puzzle_id}.json").resolve()
    if path.parent != PUZZLES_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"no such puzzle: {puzzle_id}")
    puzzle = load_puzzle(path)
    return {
        "id": puzzle.id,
        "title": puzzle.title,
        "grid": puzzle.grid_rows,
        "clues": puzzle.clues,
    }


class SolveBody(BaseModel):
    puzzle_id: str


@app.post("/api/solve/stream")
def solve_stream(body: SolveBody):
    path = (PUZZLES_DIR / f"{body.puzzle_id}.json").resolve()
    if path.parent != PUZZLES_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"no such puzzle: {body.puzzle_id}")
    puzzle = load_puzzle(path)

    def gen():
        agent = SearchJevSolver(verbose=False)
        try:
            for event in agent.stream(puzzle):
                if event["event"] == "result":
                    result = event["result"]
                    payload = {
                        "event": "done",
                        "grid": result.grid.render().split("\n"),
                        "submitted": result.submitted,
                        "steps": result.steps,
                        "wall_ms": result.wall_ms,
                        "jev_calls": result.jev_calls,
                        "tavily_calls": result.tavily_calls,
                        "tf_calls": result.tf_calls,
                        "tf_tokens": result.tf_tokens,
                        "jev_input_tokens": result.jev_input_tokens,
                    }
                    if puzzle.solution:
                        payload["score"] = score_grid(result.grid, puzzle.solution)
                else:
                    payload = event
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:  # noqa: BLE001 — surface any solver failure on the SSE stream
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
