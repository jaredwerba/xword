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


def test_search_jev_solves_3x3_from_tavily_shortlist():
    puzzle = load_puzzle(PUZZLES / "example_3x3.json")
    gold = {
        "1A": "CAT",
        "4A": "ARE",
        "5A": "TEN",
        "1D": "CAT",
        "2D": "ARE",
        "3D": "TEN",
    }
    by_clue = {puzzle.make_grid().slots[sid].clue: word for sid, word in gold.items()}

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
        return [by_clue[clue]]

    def guess(clue, pattern, k):
        return [by_clue[clue]], 10

    result = SearchJevSolver(
        evaluate=evaluate,
        search=search,
        guess=guess,
        words=["CAT", "ARE", "TEN", "DOG"],
        verbose=False,
    ).solve(puzzle)
    assert result.grid.render() == "CAT\nARE\nTEN"
    assert result.submitted is True
    assert result.tf_calls == 0
    assert result.tavily_calls >= 1
