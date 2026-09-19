"""Deterministic (non-LLM) solving utilities.

Pattern matching, puzzle-file validation, and a wordlist-backed backtracking
filler. The filler serves three roles: a structure-only baseline for the eval
harness, a construction tool for building new puzzles, and (eventually) a
repair step for near-complete LLM fills.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from .grid import BLOCK, EMPTY, Grid, Puzzle


def matches(pattern: str, word: str) -> bool:
    """True if ``word`` fits ``pattern`` ('.' matches any letter)."""
    word = word.upper()
    return len(word) == len(pattern) and all(
        p == EMPTY or p == w for p, w in zip(pattern.upper(), word)
    )


def candidates(pattern: str, words: Iterable[str]) -> list[str]:
    """All words from a wordlist that fit a slot pattern."""
    return [w.upper() for w in words if matches(pattern, w)]


def fill_grid(
    grid: Grid,
    words: Iterable[str],
    rng: random.Random | None = None,
    max_steps: int = 200_000,
) -> bool:
    """Fill every empty slot with wordlist words consistent at all crossings.

    Backtracking with minimum-remaining-values slot ordering and no duplicate
    words. Mutates ``grid`` in place; returns True on success (grid filled) or
    False (grid restored to its starting state). Pre-filled letters are
    respected, so seeding slots before calling constrains the fill.
    """
    by_len: dict[int, list[str]] = {}
    for word in words:
        word = word.strip().upper()
        if word.isalpha():
            by_len.setdefault(len(word), []).append(word)
    if rng is not None:
        for pool in by_len.values():
            rng.shuffle(pool)
    wordset = {w for pool in by_len.values() for w in pool}

    # Index words by (length, position, letter) so a partly-filled pattern
    # narrows to its candidates by set intersection instead of a full scan.
    index: dict[int, dict[tuple[int, str], set[str]]] = {}
    rank: dict[int, dict[str, int]] = {}
    for length, pool in by_len.items():
        buckets: dict[tuple[int, str], set[str]] = {}
        for word in pool:
            for position, letter in enumerate(word):
                buckets.setdefault((position, letter), set()).add(word)
        index[length] = buckets
        rank[length] = {word: i for i, word in enumerate(pool)}  # preserves rng order

    def candidates_for(pattern: str) -> list[str]:
        length = len(pattern)
        pool = by_len.get(length)
        if pool is None:
            return []
        fixed = [(i, ch) for i, ch in enumerate(pattern) if ch != EMPTY]
        if not fixed:
            return [w for w in pool if w not in used]
        sets = sorted((index[length].get(key, frozenset()) for key in fixed), key=len)
        found = set(sets[0])
        for other in sets[1:]:
            found &= other
            if not found:
                break
        found -= used
        return sorted(found, key=rank[length].__getitem__)

    # Slots complete before we start (seeds) are accepted as-is.
    pre_filled = {s for s in grid.slots if EMPTY not in grid.slot_pattern(s)}
    used = {grid.slot_pattern(s) for s in pre_filled}
    steps = 0

    def backtrack() -> bool:
        nonlocal steps
        # Minimum-remaining-values: fill the most constrained slot first.
        # Slots completed passively by crossings must still spell real words.
        best_slot, best_cands = None, None
        for slot_id in grid.slots:
            pattern = grid.slot_pattern(slot_id)
            if EMPTY not in pattern:
                if slot_id not in pre_filled and pattern not in wordset:
                    return False
                continue
            cands = candidates_for(pattern)
            if best_cands is None or len(cands) < len(best_cands):
                best_slot, best_cands = slot_id, cands
                if not cands:
                    return False  # dead end
        if best_slot is None:
            return True  # everything filled
        cells = grid.slots[best_slot].cells()
        saved = [grid.cells[r][c] for r, c in cells]
        for word in best_cands:
            steps += 1
            if steps > max_steps:
                break
            for (r, c), letter in zip(cells, word):
                grid.cells[r][c] = letter
            used.add(word)
            if backtrack():
                return True
            used.discard(word)
            for (r, c), letter in zip(cells, saved):
                grid.cells[r][c] = letter
        return False

    return backtrack()


def verify_puzzle(puzzle: Puzzle) -> list[str]:
    """Sanity-check a puzzle file. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    grid = puzzle.make_grid()
    for slot in grid.slots.values():
        if not slot.clue:
            problems.append(f"{slot.id} has no clue")
    if puzzle.solution is None:
        return problems
    sol = puzzle.solution
    if len(sol) != grid.rows or any(len(row) != grid.cols for row in sol):
        return problems + ["solution dimensions do not match grid"]
    for r in range(grid.rows):
        for c in range(grid.cols):
            tpl_block = grid.template[r][c] == BLOCK
            sol_ch = sol[r][c]
            if tpl_block != (sol_ch == BLOCK):
                problems.append(f"block mismatch at ({r},{c})")
            elif not tpl_block and not sol_ch.isalpha():
                problems.append(f"solution cell ({r},{c}) is not a letter: {sol_ch!r}")
    return problems
