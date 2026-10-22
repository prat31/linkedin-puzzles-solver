from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from linpuzs.actions import Tap
from linpuzs.solvers import SolverError
from linpuzs.solvers.mini_sudoku import Board, MiniSudokuSolver, detect_board, is_valid, solve_board

EXAMPLE = Path(__file__).parents[1] / "media" / "example" / "mini_sudoku.png"

EXAMPLE_CELLS = np.array(
    [
        [0, 0, 0, 0, 1, 0],
        [2, 0, 3, 4, 0, 0],
        [0, 5, 0, 0, 6, 0],
        [0, 3, 0, 0, 2, 0],
        [0, 0, 4, 2, 0, 1],
        [0, 2, 0, 0, 0, 0],
    ]
)


@pytest.fixture
def screenshot():
    return Image.open(EXAMPLE).convert("RGB")


def test_detects_example_board(screenshot):
    board = detect_board(screenshot)
    assert (board.n, board.box) == (6, (2, 3))
    assert (board.cells == EXAMPLE_CELLS).all()
    # Pad is 1 2 3 / 4 5 6, left to right then down.
    assert board.keys[1][1] == board.keys[2][1] == board.keys[3][1] < board.keys[4][1]
    assert board.keys[1][0] == board.keys[4][0] < board.keys[2][0] < board.keys[3][0]


def test_solves_example_end_to_end(screenshot):
    solver = MiniSudokuSolver()
    board = solver.parse(screenshot)
    solution = solver.solve(board)
    assert is_valid(board, solution)
    actions = solver.plan(board, solution)
    assert all(isinstance(a, Tap) for a in actions)
    assert len(actions) == 2 * (board.cells == 0).sum()


def test_clashing_givens_raise():
    cells = EXAMPLE_CELLS.copy()
    cells[0, 0] = 1  # second 1 in row 0
    with pytest.raises(SolverError):
        solve_board(Board(6, (2, 3), cells, [], []))


def test_non_board_screen_raises():
    with pytest.raises(SolverError):
        detect_board(Image.new("RGB", (1080, 2400), (28, 31, 35)))


@pytest.mark.parametrize("other", ["queens.png", "tango.png"])
def test_other_games_are_not_sudoku(other):
    with pytest.raises(SolverError):
        detect_board(Image.open(EXAMPLE.with_name(other)))
