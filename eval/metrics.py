"""Scoring for a filled grid against a puzzle's answer key."""

from __future__ import annotations

from collections.abc import Sequence

from xword.grid import BLOCK, Grid


def score_grid(grid: Grid, solution: Sequence[str]) -> dict:
    """Compare a fill to the answer key.

    Returns:
        letter_accuracy: correct open cells / total open cells
        word_accuracy:   fully correct slots / total slots
        solved:          True iff every open cell is correct
    """
    total_letters = correct_letters = 0
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.template[r][c] == BLOCK:
                continue
            total_letters += 1
            if grid.cells[r][c].upper() == solution[r][c].upper():
                correct_letters += 1

    total_words = correct_words = 0
    for slot in grid.slots.values():
        total_words += 1
        answer = "".join(solution[r][c] for r, c in slot.cells()).upper()
        if grid.slot_pattern(slot.id).upper() == answer:
            correct_words += 1

    return {
        "letter_accuracy": correct_letters / total_letters if total_letters else 0.0,
        "word_accuracy": correct_words / total_words if total_words else 0.0,
        "solved": correct_letters == total_letters,
    }
