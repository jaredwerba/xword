"""Search-jev solver.

Code picks slots (MRV). Constrained wordlist + Jev ranks tight patterns.
Token Factory fill_region writes leftover components (no wordlist filter).
Tavily is last resort on slots still empty after region fill.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .grid import EMPTY, Grid, Puzzle
from .jev import JevAnswers, system_one
from .paths import WORDLIST_PATH
from .solver import candidates
from .tavily import search_clue
from .tf import fill_region, guess_words
from .traces import maybe_traceable

WORDLIST_MAX = 12
NONE = "none"
NOUL_YES = 0.75
CHOICE_MIN = 0.50
MAX_STEPS = 40
MAX_REGION_FILLS = 6
EvaluateFn = Callable[[Any, dict[str, Any]], JevAnswers]
SearchFn = Callable[[str, str], list[str]]
GuessFn = Callable[[str, str, int], tuple[list[str], int]]
RegionFillFn = Callable[[list[dict[str, str]]], tuple[dict[str, str], int]]


@dataclass
class SolveResult:
    grid: Grid
    submitted: bool
    steps: int
    wall_ms: int
    jev_calls: int
    tavily_calls: int
    tf_calls: int
    jev_input_tokens: int
    tf_tokens: int
    log: list[dict[str, Any]] = field(default_factory=list)


def _load_words() -> list[str]:
    return WORDLIST_PATH.read_text().split()


class SearchJevSolver:
    def __init__(
        self,
        *,
        evaluate: EvaluateFn | None = None,
        search: SearchFn | None = None,
        guess: GuessFn | None = None,
        region_fill: RegionFillFn | None = None,
        words: list[str] | None = None,
        max_steps: int = MAX_STEPS,
        verbose: bool = False,
    ):
        self.evaluate = evaluate or system_one
        self.search = search or search_clue
        self.guess = guess or guess_words
        self.region_fill = region_fill or fill_region
        self.words = words if words is not None else _load_words()
        self._wordlist_lengths = {len(w) for w in self.words}
        self.max_steps = max_steps
        self.verbose = verbose
        self.jev_calls = 0
        self.tavily_calls = 0
        self.tf_calls = 0
        self.jev_input_tokens = 0
        self.tf_tokens = 0
        self._search_cache: dict[tuple[str, str], list[str]] = {}
        self._io_lock = threading.Lock()
        self._grid_lock = threading.Lock()
        self._tavily_sem = threading.Semaphore(2)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _jev(self, state: Any, questions: dict[str, Any]) -> JevAnswers:
        answers = self.evaluate(state, questions)
        with self._io_lock:
            self.jev_calls += 1
            self.jev_input_tokens += answers.input_tokens
        return answers

    def _tavily(self, clue: str, pattern: str) -> list[str]:
        key = (clue, pattern)
        with self._io_lock:
            if key in self._search_cache:
                return self._search_cache[key]
        from . import tavily as tavily_mod

        billed = True
        with self._tavily_sem:
            if getattr(tavily_mod, "exhausted", False):
                hits = []
                billed = False
            else:
                hits = self.search(clue, pattern)
                if getattr(tavily_mod, "exhausted", False) and not hits:
                    billed = False
        with self._io_lock:
            if billed:
                self.tavily_calls += 1
            self._search_cache[key] = hits
        self._log(f"    tavily {hits[:8]}")
        return hits

    def _open(self, grid: Grid, allowed: set[str] | None = None) -> list[str]:
        sids = [sid for sid in grid.slots if EMPTY in grid.slot_pattern(sid)]
        if allowed is not None:
            sids = [sid for sid in sids if sid in allowed]
        return sids

    def _components(self, grid: Grid) -> list[set[str]]:
        parent = {sid: sid for sid in grid.slots}

        def find(sid: str) -> str:
            while parent[sid] != sid:
                parent[sid] = parent[parent[sid]]
                sid = parent[sid]
            return sid

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            parent[rb] = ra

        cells: dict[tuple[int, int], list[str]] = {}
        for sid, slot in grid.slots.items():
            for cell in slot.cells():
                cells.setdefault(cell, []).append(sid)
        for sids in cells.values():
            for a, b in pairwise(sids):
                union(a, b)
        groups: dict[str, set[str]] = {}
        for sid in grid.slots:
            groups.setdefault(find(sid), set()).add(sid)
        return list(groups.values())

    def _wordlist(self, grid: Grid, slot_id: str, rejected: set[tuple[str, str]]) -> list[str]:
        pattern = grid.slot_pattern(slot_id)
        found = [
            w
            for w in candidates(pattern, self.words)
            if (slot_id, w) not in rejected
        ]
        filled = sum(ch != EMPTY for ch in pattern)
        if filled >= 1 and 1 <= len(found) <= WORDLIST_MAX:
            return found
        return []

    def _crossings(self, grid: Grid, slot_id: str) -> list[dict[str, str]]:
        slot = grid.slots[slot_id]
        cells = set(slot.cells())
        out: list[dict[str, str]] = []
        for other_id, other in grid.slots.items():
            if other_id == slot_id or other.direction == slot.direction:
                continue
            if cells.isdisjoint(other.cells()):
                continue
            out.append(
                {
                    "slot": other_id,
                    "clue": other.clue,
                    "pattern": grid.slot_pattern(other_id),
                }
            )
        return out

    def _crossings_live(
        self, grid: Grid, slot_id: str, word: str, rejected: set[tuple[str, str]]
    ) -> bool:
        """False if writing word would leave a wordlist-length crossing with 0 cands."""
        letters = dict(zip(grid.slots[slot_id].cells(), word.upper()))
        for other_id, other in grid.slots.items():
            if other_id == slot_id:
                continue
            if other.length not in self._wordlist_lengths:
                continue
            shares = False
            chars: list[str] = []
            for cell in other.cells():
                if cell in letters:
                    shares = True
                    chars.append(letters[cell])
                else:
                    chars.append(grid.cells[cell[0]][cell[1]])
            if not shares:
                continue
            pattern = "".join(chars)
            found = [
                w
                for w in candidates(pattern, self.words)
                if (other_id, w) not in rejected
            ]
            if not found:
                return False
        return True

    def _select(
        self,
        grid: Grid,
        skipped: set[str],
        rejected: set[tuple[str, str]],
        allowed: set[str] | None = None,
    ) -> str | None:
        ranked: list[tuple[int, int, str]] = []
        for slot_id in self._open(grid, allowed):
            if slot_id in skipped:
                continue
            pattern = grid.slot_pattern(slot_id)
            filled = sum(ch != EMPTY for ch in pattern)
            short = self._wordlist(grid, slot_id, rejected)
            remaining = len(short) if short else 10_000
            ranked.append((remaining, -filled, slot_id))
        if not ranked:
            return None
        ranked.sort()
        return ranked[0][2]

    def _rank(self, grid: Grid, slot_id: str, options: list[str], source: str) -> str | None:
        options = list(dict.fromkeys(options))
        if not options:
            return None
        slot = grid.slots[slot_id]
        pattern = grid.slot_pattern(slot_id)
        state = {
            "slot": slot_id,
            "clue": slot.clue,
            "pattern": pattern,
            "source": source,
            "options": options,
            "crossings": self._crossings(grid, slot_id),
        }
        if len(options) == 1:
            word = options[0]
            filled = sum(ch != EMPTY for ch in pattern)
            if filled >= 2:
                return word
            answers = self._jev(
                state,
                {
                    "fits": {
                        "type": "noul",
                        "instructions": (
                            f"Does {word} answer the crossword clue {slot.clue!r} "
                            f"given pattern {pattern}?"
                        ),
                    }
                },
            )
            return word if answers.probability("fits") >= NOUL_YES else None
        criteria = {w: f"{w}: fits {pattern}, clue {slot.clue!r}" for w in options}
        criteria[NONE] = "None of these is the answer."
        answers = self._jev(
            state,
            {
                "word": {
                    "type": "choice",
                    "instructions": (
                        "Which option is the crossword answer for this clue "
                        "and pattern, given the crossing clues? "
                        "Choose none if every option is a stretch."
                    ),
                    "criteria": criteria,
                }
            },
        )
        picked = answers.choice("word")
        if picked == NONE or picked not in options:
            return None
        if answers.probability("word") < CHOICE_MIN:
            return None
        return picked

    def _apply_region(
        self,
        grid: Grid,
        leftover: list[str],
        rejected: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Token Factory fill of leftover slots. Not filtered to the wordlist."""
        if not leftover:
            return []
        payload = [
            {
                "id": sid,
                "clue": grid.slots[sid].clue,
                "pattern": grid.slot_pattern(sid),
                "length": str(grid.slots[sid].length),
            }
            for sid in leftover
        ]
        try:
            fills, tokens = self.region_fill(payload)
        except RuntimeError:
            return []
        with self._io_lock:
            self.tf_calls += 1
            self.tf_tokens += tokens
        written: list[tuple[str, str]] = []
        leftover_set = set(leftover)
        for sid, word in fills.items():
            sid = sid.upper()
            word = word.strip().upper()
            if sid not in leftover_set or not word.isalpha():
                continue
            if len(word) != grid.slots[sid].length:
                continue
            if (sid, word) in rejected:
                continue
            with self._grid_lock:
                if grid.fill_slot(sid, word):
                    rejected.add((sid, word))
                    continue
            written.append((sid, word))
            self._log(f"    region {sid}={word}")
        return written

    def _tavily_last(
        self,
        grid: Grid,
        leftover: list[str],
        rejected: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Search only slots still empty after fill_region."""
        written: list[tuple[str, str]] = []
        for sid in leftover:
            if EMPTY not in grid.slot_pattern(sid):
                continue
            pattern = grid.slot_pattern(sid)
            hits = [
                w
                for w in self._tavily(grid.slots[sid].clue, pattern)
                if (sid, w) not in rejected and len(w) == grid.slots[sid].length
            ]
            if not hits:
                continue
            if len(hits) == 1:
                word = hits[0]
            else:
                word = self._rank(grid, sid, hits, "tavily")
            if not word:
                continue
            with self._grid_lock:
                if grid.fill_slot(sid, word):
                    rejected.add((sid, word))
                    continue
            written.append((sid, word))
            self._log(f"    tavily-last {sid}={word}")
        return written

    def _fill_component(
        self,
        grid: Grid,
        allowed: set[str],
        *,
        max_local: int,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        rejected: set[tuple[str, str]] = set()
        skipped: set[str] = set()
        fill_stack: list[tuple[str, str]] = []
        steps = 0
        region_budget = 0

        def note(event: dict[str, Any]) -> None:
            if emit:
                emit(event)

        def commit(sid: str, word: str, source: str) -> None:
            nonlocal steps
            steps += 1
            fill_stack.append((sid, word))
            skipped.clear()
            event = {
                "event": "fill",
                "slot": sid,
                "word": word,
                "source": source,
                "grid": grid.render().split("\n"),
                "tavily_calls": self.tavily_calls,
                "jev_calls": self.jev_calls,
                "tf_calls": self.tf_calls,
                "tf_tokens": self.tf_tokens,
            }
            note(event)
            self._log(f"[{steps}] {sid}={word} via {source}")

        while steps < max_local:
            with self._grid_lock:
                if not self._open(grid, allowed):
                    break

            unique_filled = False
            with self._grid_lock:
                for sid in list(self._open(grid, allowed)):
                    short = self._wordlist(grid, sid, rejected)
                    if len(short) != 1:
                        continue
                    pattern = grid.slot_pattern(sid)
                    if sum(ch != EMPTY for ch in pattern) < 2:
                        continue
                    word = short[0]
                    if not self._crossings_live(grid, sid, word, rejected):
                        continue
                    if grid.fill_slot(sid, word):
                        rejected.add((sid, word))
                        continue
                    unique_filled = True
                    commit(sid, word, "wordlist")
            if unique_filled:
                continue

            with self._grid_lock:
                slot_id = self._select(grid, skipped, rejected, allowed)

            leftover = self._open(grid, allowed)
            if slot_id is None:
                wrote = self._apply_region(grid, leftover, rejected)
                region_budget += 1
                for sid, word in wrote:
                    commit(sid, word, "tf-region")
                if wrote:
                    continue
                wrote = self._tavily_last(grid, leftover, rejected)
                for sid, word in wrote:
                    commit(sid, word, "tavily")
                if wrote:
                    wrote = self._apply_region(
                        grid, self._open(grid, allowed), rejected
                    )
                    for sid, word in wrote:
                        commit(sid, word, "tf-region")
                    continue
                if not fill_stack:
                    break
                with self._grid_lock:
                    undone_slot, undone_word = fill_stack.pop()
                    grid.clear_slot(undone_slot)
                    rejected.add((undone_slot, undone_word))
                    fill_stack = [
                        (sid, word)
                        for sid, word in fill_stack
                        if EMPTY not in grid.slot_pattern(sid)
                        and grid.slot_pattern(sid) == word
                    ]
                skipped.clear()
                note({"event": "undo", "slot": undone_slot, "word": undone_word})
                continue

            pattern = grid.slot_pattern(slot_id)
            filled = sum(ch != EMPTY for ch in pattern)
            short = self._wordlist(grid, slot_id, rejected)
            if filled >= 2 and short:
                note(
                    {
                        "event": "consider",
                        "turn": steps + 1,
                        "slot": slot_id,
                        "pattern": pattern,
                        "clue": grid.slots[slot_id].clue,
                    }
                )
                word = self._rank(grid, slot_id, short, "wordlist")
                if not word:
                    wrote = self._apply_region(grid, leftover, rejected)
                    region_budget += 1
                    for sid, w in wrote:
                        commit(sid, w, "tf-region")
                    if not wrote:
                        skipped.add(slot_id)
                    continue
                with self._grid_lock:
                    conflicts = grid.fill_slot(slot_id, word)
                if conflicts:
                    rejected.add((slot_id, word))
                    skipped.add(slot_id)
                    note(
                        {
                            "event": "conflict",
                            "slot": slot_id,
                            "word": word,
                            "source": "wordlist",
                        }
                    )
                    continue
                commit(slot_id, word, "wordlist")
                continue

            # Underconstrained: one region write, then Tavily only on leftovers.
            wrote = self._apply_region(grid, leftover, rejected)
            region_budget += 1
            for sid, word in wrote:
                commit(sid, word, "tf-region")
            if wrote:
                continue
            wrote = self._tavily_last(grid, leftover, rejected)
            for sid, word in wrote:
                commit(sid, word, "tavily")
            if wrote:
                continue
            skipped.add(slot_id)
            if region_budget >= MAX_REGION_FILLS:
                break
        return steps

    def _events(self, puzzle: Puzzle) -> Iterator[dict[str, Any]]:
        grid = puzzle.make_grid()
        log: list[dict[str, Any]] = []
        started = time.perf_counter()
        yield {"event": "start", "puzzle": puzzle.id, "title": puzzle.title}

        pending: list[dict[str, Any]] = []

        def emit(event: dict[str, Any]) -> None:
            pending.append(event)
            log.append(event)

        regions = self._components(grid)
        steps = 0
        for region in regions:
            steps += self._fill_component(
                grid,
                region,
                max_local=max(self.max_steps, len(region) * 3, 28),
                emit=emit,
            )
            while pending:
                yield pending.pop(0)

        if not self._open(grid):
            event = {
                "event": "submit",
                "grid": grid.render().split("\n"),
                "complete": grid.is_complete(),
            }
            log.append(event)
            yield event

        wall_ms = int((time.perf_counter() - started) * 1000)
        result = SolveResult(
            grid=grid,
            submitted=not self._open(grid),
            steps=steps,
            wall_ms=wall_ms,
            jev_calls=self.jev_calls,
            tavily_calls=self.tavily_calls,
            tf_calls=self.tf_calls,
            jev_input_tokens=self.jev_input_tokens,
            tf_tokens=self.tf_tokens,
            log=log,
        )
        yield {"event": "result", "result": result}

    def solve(self, puzzle: Puzzle) -> SolveResult:
        def _run() -> SolveResult:
            grid = puzzle.make_grid()
            regions = self._components(grid)
            if len(regions) < 4:
                result = None
                for event in self._events(puzzle):
                    if event["event"] == "result":
                        result = event["result"]
                assert result is not None
                return result
            started = time.perf_counter()
            workers = min(4, len(regions))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(
                    pool.map(
                        lambda region: self._fill_component(
                            grid,
                            region,
                            max_local=max(28, len(region) * 3),
                        ),
                        regions,
                    )
                )
            wall_ms = int((time.perf_counter() - started) * 1000)
            return SolveResult(
                grid=grid,
                submitted=not self._open(grid),
                steps=sum(1 for _ in grid.slots),
                wall_ms=wall_ms,
                jev_calls=self.jev_calls,
                tavily_calls=self.tavily_calls,
                tf_calls=self.tf_calls,
                jev_input_tokens=self.jev_input_tokens,
                tf_tokens=self.tf_tokens,
            )

        return maybe_traceable("xword.solve")(_run)()

    def stream(self, puzzle: Puzzle) -> Iterator[dict[str, Any]]:
        yield from self._events(puzzle)
