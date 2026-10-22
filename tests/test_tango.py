from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from linpuzs.actions import Tap
from linpuzs.solvers import SolverError
from linpuzs.solvers.tango import EMPTY, MOON, SUN, Board, TangoSolver, detect_board, is_valid, solve_board

EXAMPLE = Path(__file__).parents[1] / "media" / "example" / "tango.png"

_, S, M = EMPTY, SUN, MOON
EXAMPLE_CELLS = np.array(
    [
        [_, _, _, M, _, M],
        [_, _, _, M, _, S],
        [_, _, _, S, M, M],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
    ]
)


@pytest.fixture
def screenshot():
    return Image.open(EXAMPLE).convert("RGB")


def test_detects_example_board(screenshot):
    board = detect_board(screenshot)
    assert board.n == 6
    assert (board.cells == EXAMPLE_CELLS).all()
    assert sorted(board.equal) == [((3, 0), (4, 0)), ((3, 1), (3, 2)), ((3, 2), (4, 2))]
    assert sorted(board.opposite) == [((3, 0), (3, 1)), ((4, 0), (5, 0)), ((4, 2), (5, 2))]


def test_solves_example_end_to_end(screenshot):
    solver = TangoSolver()
    board = solver.parse(screenshot)
    solution = solver.solve(board)
    assert is_valid(board, solution)
    actions = solver.plan(board, solution)
    assert all(isinstance(a, Tap) for a in actions)
    empty = board.cells == EMPTY
    assert len(actions) == (solution[empty] == SUN).sum() + 2 * (solution[empty] == MOON).sum()


def test_unsolvable_raises():
    # Two suns already sit next to each other, and = forces a third.
    cells = np.zeros((4, 4), dtype=int)
    cells[0, 0] = cells[0, 1] = SUN
    with pytest.raises(SolverError):
        solve_board(Board(4, cells, equal=[((0, 1), (0, 2))]))


def test_non_board_screen_raises():
    with pytest.raises(SolverError):
        detect_board(Image.new("RGB", (1080, 2400), (28, 31, 35)))


def test_queens_screen_is_not_a_tango_board():
    with pytest.raises(SolverError):
        detect_board(Image.open(EXAMPLE.with_name("queens.png")))
