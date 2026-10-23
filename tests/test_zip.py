from pathlib import Path

import pytest
from PIL import Image

from linpuzs.actions import Drag
from linpuzs.solvers import SolverError
from linpuzs.solvers.zip import Board, ZipSolver, detect_board, is_valid, solve_path

EXAMPLE = Path(__file__).parents[1] / "media" / "example" / "zip.png"

EXAMPLE_NUMBERS = {
    (1, 0): 11, (1, 1): 9, (1, 4): 4, (1, 5): 12,
    (2, 1): 10, (2, 5): 5,
    (3, 0): 8, (3, 4): 3,
    (4, 0): 7, (4, 1): 1, (4, 4): 2, (4, 5): 6,
}  # fmt: skip


@pytest.fixture
def screenshot():
    return Image.open(EXAMPLE).convert("RGB")


def test_detects_example_board(screenshot):
    board = detect_board(screenshot)
    assert board.n == 6
    assert board.numbers == EXAMPLE_NUMBERS
    assert board.walls == set()


def test_solves_example_end_to_end(screenshot):
    solver = ZipSolver()
    board = solver.parse(screenshot)
    path = solver.solve(board)
    assert is_valid(board, path)
    (drag,) = solver.plan(board, path)
    assert isinstance(drag, Drag) and len(drag.points) == 36


# 8×8 "HARD" board: its dots nearly fill their cells, which used to make every
# edge next to a dot look like a wall.
HARD_NUMBERS = {
    (1, 1): 1, (1, 2): 2, (1, 5): 3, (6, 5): 4, (6, 2): 5, (6, 1): 6, (5, 1): 7,
    (2, 1): 8, (2, 2): 9, (2, 5): 10, (3, 4): 11, (3, 3): 12, (5, 2): 13, (4, 3): 14,
    (4, 4): 15, (5, 5): 16, (2, 6): 17, (1, 6): 18, (6, 6): 19, (5, 6): 20,
}  # fmt: skip


def test_hard_board_with_big_dots():
    solver = ZipSolver()
    board = solver.parse(Image.open(EXAMPLE.with_name("zip_hard.png")))
    assert board.n == 8
    assert board.numbers == HARD_NUMBERS
    assert board.walls == set()
    path = solver.solve(board)
    assert is_valid(board, path)


def test_walls_are_respected():
    # 1 -> 2 directly is blocked, so the path has to go down and around.
    numbers = {(0, 0): 1, (0, 1): 2, (2, 2): 3}
    board = Board(3, numbers, {frozenset({(0, 0), (0, 1)})})
    path = solve_path(board)
    assert is_valid(board, path)
    assert path[:6] == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]


def test_walled_in_start_raises():
    walls = {frozenset({(0, 0), (0, 1)}), frozenset({(0, 0), (1, 0)})}
    with pytest.raises(SolverError):
        solve_path(Board(3, {(0, 0): 1, (2, 2): 2}, walls))


def test_parity_mismatch_raises():
    # On a 4×4 board start and end must be on different colours.
    with pytest.raises(SolverError):
        solve_path(Board(4, {(0, 0): 1, (0, 2): 2}))


def test_non_board_screen_raises():
    with pytest.raises(SolverError):
        detect_board(Image.new("RGB", (1080, 2400), (28, 31, 35)))


@pytest.mark.parametrize("other", ["queens.png", "tango.png", "mini_sudoku.png"])
def test_other_games_are_not_zip(other):
    with pytest.raises(SolverError):
        detect_board(Image.open(EXAMPLE.with_name(other)))
