"""Crossword grid model: cells, slot detection, numbering, and fills.

A puzzle is defined by a rectangular template where ``#`` marks a block and
``.`` marks an open cell. Slots (across/down entries) and their numbers are
derived from the template using standard crossword numbering rules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BLOCK = "#"
EMPTY = "."


@dataclass(frozen=True)
class Slot:
    """One across or down entry in the grid."""

    number: int
    direction: str  # "across" | "down"
    row: int
    col: int
    length: int
    clue: str = ""

    @property
    def id(self) -> str:
        return f"{self.number}{'A' if self.direction == 'across' else 'D'}"

    def cells(self) -> list[tuple[int, int]]:
        dr, dc = (0, 1) if self.direction == "across" else (1, 0)
        return [(self.row + i * dr, self.col + i * dc) for i in range(self.length)]


class Grid:
    """Mutable fill state over a fixed template."""

    def __init__(self, template: Sequence[str], clues: Mapping | None = None):
        if not template or any(len(row) != len(template[0]) for row in template):
            raise ValueError("template must be a non-empty rectangle")
        if any(ch not in (BLOCK, EMPTY) for row in template for ch in row):
            raise ValueError(f"template may only contain {BLOCK!r} and {EMPTY!r}")
        self.template = [str(row) for row in template]
        self.rows = len(template)
        self.cols = len(template[0])
        self.cells: list[list[str]] = [list(row) for row in self.template]
        self.slots: dict[str, Slot] = self._find_slots(clues or {})

    def _find_slots(self, clues: Mapping) -> dict[str, Slot]:
        slots: dict[str, Slot] = {}
        number = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.template[r][c] == BLOCK:
                    continue
                starts_across = (c == 0 or self.template[r][c - 1] == BLOCK) and (
                    c + 1 < self.cols and self.template[r][c + 1] != BLOCK
                )
                starts_down = (r == 0 or self.template[r - 1][c] == BLOCK) and (
                    r + 1 < self.rows and self.template[r + 1][c] != BLOCK
                )
                if not (starts_across or starts_down):
                    continue
                number += 1
                if starts_across:
                    length = self._run_length(r, c, 0, 1)
                    clue = str(clues.get("across", {}).get(str(number), ""))
                    slot = Slot(number, "across", r, c, length, clue)
                    slots[slot.id] = slot
                if starts_down:
                    length = self._run_length(r, c, 1, 0)
                    clue = str(clues.get("down", {}).get(str(number), ""))
                    slot = Slot(number, "down", r, c, length, clue)
                    slots[slot.id] = slot
        return slots

    def _run_length(self, r: int, c: int, dr: int, dc: int) -> int:
        length = 0
        while 0 <= r < self.rows and 0 <= c < self.cols and self.template[r][c] != BLOCK:
            length += 1
            r, c = r + dr, c + dc
        return length

    # ------------------------------------------------------------------ state

    def slot_pattern(self, slot_id: str) -> str:
        """Current letters for a slot, with ``.`` for unknown cells."""
        slot = self._get_slot(slot_id)
        return "".join(self.cells[r][c] for r, c in slot.cells())

    def fill_slot(self, slot_id: str, word: str, overwrite: bool = False) -> list[str]:
        """Write ``word`` into a slot.

        Returns a list of conflict descriptions. If conflicts exist and
        ``overwrite`` is False, the grid is left unchanged.
        """
        slot = self._get_slot(slot_id)
        word = word.strip().upper()
        if len(word) != slot.length:
            raise ValueError(f"{slot_id} needs {slot.length} letters, got {len(word)}")
        if not word.isalpha():
            raise ValueError(f"word must be letters only, got {word!r}")
        conflicts = [
            f"cell ({r},{c}) holds {self.cells[r][c]!r}, {slot_id} wants {letter!r}"
            for (r, c), letter in zip(slot.cells(), word)
            if self.cells[r][c] not in (EMPTY, letter)
        ]
        if conflicts and not overwrite:
            return conflicts
        for (r, c), letter in zip(slot.cells(), word):
            self.cells[r][c] = letter
        return conflicts

    def clear_slot(self, slot_id: str) -> None:
        """Blank every cell of a slot (crossing entries lose that letter too)."""
        for r, c in self._get_slot(slot_id).cells():
            self.cells[r][c] = EMPTY

    def set_rows(self, rows: Sequence[str]) -> None:
        """Overwrite the whole fill from full grid rows (blocks must align)."""
        if len(rows) != self.rows or any(len(row) != self.cols for row in rows):
            raise ValueError("rows do not match grid dimensions")
        for r in range(self.rows):
            for c in range(self.cols):
                is_block = self.template[r][c] == BLOCK
                if is_block != (rows[r][c] == BLOCK):
                    raise ValueError(f"block mismatch at ({r},{c})")
                if not is_block:
                    self.cells[r][c] = rows[r][c].upper()

    def is_complete(self) -> bool:
        return all(cell != EMPTY for row in self.cells for cell in row)

    def render(self) -> str:
        return "\n".join("".join(row) for row in self.cells)

    def _get_slot(self, slot_id: str) -> Slot:
        try:
            return self.slots[slot_id.upper()]
        except KeyError:
            raise KeyError(f"no such slot {slot_id!r}; valid: {sorted(self.slots)}") from None


@dataclass
class Puzzle:
    """A puzzle file: structure, clues, and (optionally) the answer key."""

    id: str
    title: str
    grid_rows: list[str]
    clues: dict
    solution: list[str] | None = None
    path: Path | None = field(default=None, repr=False)

    def make_grid(self) -> Grid:
        return Grid(self.grid_rows, self.clues)


def puzzle_from_mapping(data: Mapping, path: Path | None = None) -> Puzzle:
    """Build a Puzzle from a decoded puzzle document."""
    fallback = path.stem if path else "puzzle"
    return Puzzle(
        id=str(data.get("id", fallback)),
        title=str(data.get("title", fallback)),
        grid_rows=[str(row) for row in data["grid"]],
        clues=dict(data.get("clues", {})),
        solution=[str(row) for row in data["solution"]] if data.get("solution") else None,
        path=path,
    )


def load_puzzle(path: str | Path) -> Puzzle:
    path = Path(path)
    return puzzle_from_mapping(json.loads(path.read_text()), path)
