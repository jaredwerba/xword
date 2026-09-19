from pathlib import Path

import pytest

from xword.grid import Grid, load_puzzle
from xword.solver import verify_puzzle

PUZZLES = Path(__file__).parents[1] / "data" / "puzzles"


@pytest.fixture
def mini():
    return load_puzzle(PUZZLES / "example_mini_5x5.json")


def test_slot_numbering(mini):
    grid = mini.make_grid()
    across = [s.id for s in grid.slots.values() if s.direction == "across"]
    down = [s.id for s in grid.slots.values() if s.direction == "down"]
    assert across == ["1A", "4A", "5A", "6A", "7A"]
    assert down == ["1D", "2D", "3D", "4D", "5D"]


def test_fill_reports_conflicts(mini):
    grid = mini.make_grid()
    assert grid.fill_slot("1A", "pit") == []
    conflicts = grid.fill_slot("1D", "QUOTA")
    assert len(conflicts) == 1
    assert grid.fill_slot("1D", "PILOT") == []


def test_example_puzzles_are_valid():
    for path in sorted(PUZZLES.glob("*.json")):
        assert verify_puzzle(load_puzzle(path)) == []


def test_rejects_bad_template():
    with pytest.raises(ValueError):
        Grid(["..", "..."])
