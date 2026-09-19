"""Search-jev solver: code picks slots, Tavily grounds, Jev ranks, TF is last."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .grid import EMPTY, Grid, Puzzle
from .jev import JevAnswers, system_one
from .paths import WORDLIST_PATH
from .solver import candidates
from .tavily import search_clue
from .tf import guess_words
from .traces import maybe_traceable

WORDLIST_MAX = 12
NONE = "none"
NOUL_YES = 0.75
CHOICE_MIN = 0.50
MAX_STEPS = 40
EvaluateFn = Callable[[Any, dict[str, Any]], JevAnswers]
SearchFn = Callable[[str, str], list[str]]
GuessFn = Callable[[str, str, int], tuple[list[str], int]]


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
        words: list[str] | None = None,
        max_steps: int = MAX_STEPS,
        verbose: bool = False,
    ):
        self.evaluate = evaluate or system_one
        self.search = search or search_clue
        self.guess = guess or guess_words
        self.words = words if words is not None else _load_words()
        self.max_steps = max_steps
        self.verbose = verbose
        self.jev_calls = 0
        self.tavily_calls = 0
        self.tf_calls = 0
        self.jev_input_tokens = 0
        self.tf_tokens = 0
        self._search_cache: dict[tuple[str, str], list[str]] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _jev(self, state: Any, questions: dict[str, Any]) -> JevAnswers:
        answers = self.evaluate(state, questions)
        self.jev_calls += 1
        self.jev_input_tokens += answers.input_tokens
        return answers

    def _tavily(self, clue: str, pattern: str) -> list[str]:
        key = (clue, pattern)
        if key in self._search_cache:
            return self._search_cache[key]
        hits = self.search(clue, pattern)
        self.tavily_calls += 1
        self._search_cache[key] = hits
        self._log(f"    tavily {hits[:8]}")
        return hits

    def _tf(self, clue: str, pattern: str) -> list[str]:
        guesses, tokens = self.guess(clue, pattern, 8)
        self.tf_calls += 1
        self.tf_tokens += tokens
        self._log(f"    tf {guesses}")
        return guesses

    def _open(self, grid: Grid) -> list[str]:
        return [sid for sid in grid.slots if EMPTY in grid.slot_pattern(sid)]

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

    def _select(self, grid: Grid, skipped: set[str], rejected: set[tuple[str, str]]) -> str | None:
        ranked: list[tuple[int, int, str]] = []
        for slot_id in self._open(grid):
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
                        "and pattern? Choose none if every option is a stretch."
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

    def _pick(
        self, grid: Grid, slot_id: str, rejected: set[tuple[str, str]]
    ) -> tuple[str | None, str]:
        pattern = grid.slot_pattern(slot_id)
        clue = grid.slots[slot_id].clue
        filled = sum(ch != EMPTY for ch in pattern)
        short = self._wordlist(grid, slot_id, rejected)
        if filled >= 2 and short:
            word = self._rank(grid, slot_id, short, "wordlist")
            if word:
                return word, "wordlist"
        hits = [
            w
            for w in self._tavily(clue, pattern)
            if (slot_id, w) not in rejected
        ]
        extras = [
            w
            for w in candidates(pattern, self.words)
            if (slot_id, w) not in rejected
        ][:WORDLIST_MAX]
        mixed = list(dict.fromkeys(hits + extras))
        # Wordlist-only Jev on a first letter is how NAME beat NEXT for "Who's ___?".
        # Need Tavily or TF when the slot is still underconstrained.
        if hits:
            word = self._rank(grid, slot_id, mixed or hits, "tavily")
            if word in hits:
                return word, "tavily"
            word = self._rank(grid, slot_id, hits, "tavily")
            if word:
                return word, "tavily"
        if filled >= 2 and (short or extras):
            word = self._rank(grid, slot_id, short or extras, "wordlist")
            if word:
                return word, "wordlist"
        wordset = {w.upper() for w in self.words}
        guesses = [
            w
            for w in self._tf(clue, pattern)
            if (slot_id, w) not in rejected and w in wordset
        ]
        word = self._rank(grid, slot_id, guesses, "token-factory")
        return word, "token-factory"

    def _events(self, puzzle: Puzzle) -> Iterator[dict[str, Any]]:
        grid = puzzle.make_grid()
        rejected: set[tuple[str, str]] = set()
        skipped: set[str] = set()
        fill_stack: list[tuple[str, str]] = []
        log: list[dict[str, Any]] = []
        steps = 0
        started = time.perf_counter()
        yield {"event": "start", "puzzle": puzzle.id, "title": puzzle.title}

        while steps < self.max_steps:
            if not self._open(grid):
                yield {
                    "event": "submit",
                    "grid": grid.render().split("\n"),
                    "complete": grid.is_complete(),
                }
                log.append({"event": "submit"})
                break
            # Unique wordlist hits don't need Jev or Tavily.
            unique_filled = False
            for sid in list(self._open(grid)):
                short = self._wordlist(grid, sid, rejected)
                if len(short) != 1:
                    continue
                pattern = grid.slot_pattern(sid)
                if sum(ch != EMPTY for ch in pattern) < 2:
                    continue
                word = short[0]
                if grid.fill_slot(sid, word):
                    rejected.add((sid, word))
                    continue
                steps += 1
                unique_filled = True
                fill_stack.append((sid, word))
                event = {
                    "event": "fill",
                    "slot": sid,
                    "word": word,
                    "source": "wordlist",
                    "grid": grid.render().split("\n"),
                    "tavily_calls": self.tavily_calls,
                    "jev_calls": self.jev_calls,
                    "tf_calls": self.tf_calls,
                    "tf_tokens": self.tf_tokens,
                }
                log.append(event)
                self._log(f"[{steps}] {sid}={word} via wordlist")
                yield event
            if unique_filled:
                skipped.clear()
                continue
            slot_id = self._select(grid, skipped, rejected)
            if slot_id is None:
                if not fill_stack:
                    break
                undone_slot, undone_word = fill_stack.pop(0)
                grid.clear_slot(undone_slot)
                rejected.add((undone_slot, undone_word))
                fill_stack = [
                    (sid, word)
                    for sid, word in fill_stack
                    if EMPTY not in grid.slot_pattern(sid)
                    and grid.slot_pattern(sid) == word
                ]
                skipped.clear()
                event = {"event": "undo", "slot": undone_slot, "word": undone_word}
                log.append(event)
                yield event
                continue
            steps += 1
            yield {
                "event": "consider",
                "turn": steps,
                "slot": slot_id,
                "pattern": grid.slot_pattern(slot_id),
                "clue": grid.slots[slot_id].clue,
            }
            word, source = self._pick(grid, slot_id, rejected)
            if not word:
                skipped.add(slot_id)
                log.append({"event": "skip", "slot": slot_id})
                continue
            conflicts = grid.fill_slot(slot_id, word)
            if conflicts:
                rejected.add((slot_id, word))
                skipped.add(slot_id)
                event = {
                    "event": "conflict",
                    "slot": slot_id,
                    "word": word,
                    "source": source,
                }
                log.append(event)
                yield event
                continue
            fill_stack.append((slot_id, word))
            skipped.clear()
            event = {
                "event": "fill",
                "slot": slot_id,
                "word": word,
                "source": source,
                "grid": grid.render().split("\n"),
                "tavily_calls": self.tavily_calls,
                "jev_calls": self.jev_calls,
                "tf_calls": self.tf_calls,
                "tf_tokens": self.tf_tokens,
            }
            log.append(event)
            self._log(f"[{steps}] {slot_id}={word} via {source}")
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
            result = None
            for event in self._events(puzzle):
                if event["event"] == "result":
                    result = event["result"]
            assert result is not None
            return result

        return maybe_traceable("xword.solve")(_run)()

    def stream(self, puzzle: Puzzle) -> Iterator[dict[str, Any]]:
        yield from self._events(puzzle)
