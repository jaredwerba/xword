from pathlib import Path

from xword.agent import SearchJevSolver
from xword.grid import load_puzzle
from xword.jev import JevAnswers

PUZZLES = Path(__file__).parents[1] / "data" / "puzzles"


def _answers(payload: dict) -> JevAnswers:
    return JevAnswers(
        answers=payload,
        model="fake",
        endpoint="fake",
        input_tokens=5,
        output_tokens=1,
        raw={"answers": payload},
    )


def _gold_3x3():
    puzzle = load_puzzle(PUZZLES / "example_3x3.json")
    gold = {
        "1A": "CAT",
        "4A": "ARE",
        "5A": "TEN",
        "1D": "CAT",
        "2D": "ARE",
        "3D": "TEN",
    }
    return puzzle, gold


def test_search_jev_solves_3x3_from_region_fill():
    puzzle, gold = _gold_3x3()
    searches = []

    def evaluate(state, questions):
        raise AssertionError("constrained Jev should not run on an empty 3x3")

    def search(clue, pattern):
        searches.append((clue, pattern))
        return ["DOG"]

    def region_fill(slots):
        return {s["id"]: gold[s["id"]] for s in slots}, 20

    result = SearchJevSolver(
        evaluate=evaluate,
        search=search,
        region_fill=region_fill,
        words=["CAT", "ARE", "TEN", "DOG"],
        verbose=False,
    ).solve(puzzle)
    assert result.grid.render() == "CAT\nARE\nTEN"
    assert result.submitted is True
    assert result.tf_calls == 1
    assert result.tavily_calls == 0
    assert searches == []


def test_tavily_only_after_empty_region_fill():
    puzzle, gold = _gold_3x3()
    by_clue = {puzzle.make_grid().slots[sid].clue: word for sid, word in gold.items()}
    searches = []

    def evaluate(state, questions):
        slot = state["slot"]
        if "fits" in questions:
            word = state["options"][0]
            p = 0.99 if word == gold[slot] else 0.01
            return _answers({"fits": {"noul": p}})
        criteria = questions["word"]["criteria"]
        winner = gold[slot] if gold[slot] in criteria else "none"
        probs = {name: (1.0 if name == winner else 0.0) for name in criteria}
        return _answers({"word": {"choice": winner, "probabilities": probs}})

    def search(clue, pattern):
        searches.append(clue)
        return [by_clue[clue]]

    def region_fill(slots):
        return {}, 5

    result = SearchJevSolver(
        evaluate=evaluate,
        search=search,
        region_fill=region_fill,
        words=["CAT", "ARE", "TEN", "DOG"],
        verbose=False,
    ).solve(puzzle)
    assert result.grid.render() == "CAT\nARE\nTEN"
    assert result.submitted is True
    assert searches
    assert result.tavily_calls >= 1


def test_single_tavily_hit_skips_jev():
    puzzle, gold = _gold_3x3()
    by_clue = {puzzle.make_grid().slots[sid].clue: word for sid, word in gold.items()}
    jev_calls = {"n": 0}

    def evaluate(state, questions):
        jev_calls["n"] += 1
        return _answers({"fits": {"noul": 0.99}})

    def search(clue, pattern):
        return [by_clue[clue]]

    def region_fill(slots):
        return {}, 1

    result = SearchJevSolver(
        evaluate=evaluate,
        search=search,
        region_fill=region_fill,
        words=["CAT", "ARE", "TEN", "DOG"],
        verbose=False,
    ).solve(puzzle)
    assert result.submitted is True
    assert jev_calls["n"] == 0


def test_unique_fill_blocked_when_crossing_has_no_candidates():
    puzzle, _gold = _gold_3x3()
    grid = puzzle.make_grid()
    solver = SearchJevSolver(words=["CAT", "ARE", "TEN"], verbose=False)
    grid.fill_slot("1A", "CAT")
    # 4A would be A.. ; unique ARE is live. Poison 1D to DOG so 4A's A is D.
    grid.cells[0][0] = "D"
    grid.cells[0][1] = "O"
    grid.cells[0][2] = "G"
    # 1D becomes D..; ORE at 4A would make 1D DO. which is not in the wordlist.
    assert solver._crossings_live(grid, "4A", "ORE", set()) is False
