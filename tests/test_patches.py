from pathlib import Path

import pytest
from PIL import Image

from linpuzs.actions import Drag, Tap
from linpuzs.solvers import SolverError
from linpuzs.solvers.patches import (
    ANY,
    SQUARE,
    TALL,
    WIDE,
    Board,
    Clue,
    PatchesSolver,
    detect_board,
    is_valid,
    solve_patches,
)

EXAMPLE = Path(__file__).parents[1] / "media" / "example" / "patches.png"

EXAMPLE_CLUES = {
    (0, 0): (ANY, 3), (0, 4): (TALL, 6), (1, 1): (ANY, 3), (2, 2): (ANY, 3),
    (2, 6): (TALL, 6), (3, 3): (TALL, 5), (4, 0): (TALL, 6), (4, 4): (ANY, 4),
    (5, 5): (ANY, 3), (6, 2): (WIDE, 6), (6, 6): (ANY, 4),
}  # fmt: skip


@pytest.fixture
def screenshot():
    return Image.open(EXAMPLE).convert("RGB")


def test_detects_example_board(screenshot):
    board = detect_board(screenshot)
    assert board.n == 7
    assert {cell: (clue.shape, clue.number) for cell, clue in board.clues.items()} == EXAMPLE_CLUES


def test_solves_example_end_to_end(screenshot):
    solver = PatchesSolver()
    board = solver.parse(screenshot)
    rects = solver.solve(board)
    assert is_valid(board, rects)
    actions = solver.plan(board, rects)
    assert len(actions) == len(board.clues)
    assert all(isinstance(a, (Drag, Tap)) for a in actions)


def test_shape_rules():
    # 2×2 board, one square 4 clue: the whole board.
    board = Board(2, {(0, 0): Clue(SQUARE, 4)})
    assert solve_patches(board) == [(0, 0, 2, 2)]
    # A tall 4 can't fit a 2×2 board.
    with pytest.raises(SolverError):
        solve_patches(Board(2, {(0, 0): Clue(TALL, 4)}))


def test_unnumbered_clues():
    # Two wide clues of unknown size on a 2×3 grid... on a 3×3: rows of 3.
    clues = {(0, 0): Clue(WIDE, None), (1, 1): Clue(WIDE, None), (2, 2): Clue(WIDE, None)}
    board = Board(3, clues)
    rects = solve_patches(board)
    assert is_valid(board, rects) and rects == [(0, 0, 1, 3), (1, 0, 1, 3), (2, 0, 1, 3)]


def test_non_board_screen_raises():
    with pytest.raises(SolverError):
        detect_board(Image.new("RGB", (1080, 2400), (28, 31, 35)))


@pytest.mark.parametrize("other", ["queens.png", "tango.png", "mini_sudoku.png", "zip.png"])
def test_other_games_are_not_patches(other):
    with pytest.raises(SolverError):
        detect_board(Image.open(EXAMPLE.with_name(other)))
