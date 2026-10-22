from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from linpuzs.actions import Tap
from linpuzs.solvers import SolverError
from linpuzs.solvers.queens import QueensSolver, detect_board, is_valid, solve_regions

EXAMPLES = Path(__file__).parents[1] / "media" / "example"
EXAMPLE = EXAMPLES / "queens.png"

EXAMPLE_REGIONS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 2, 2, 2, 3, 3, 1, 1],
        [0, 0, 2, 2, 4, 1, 1, 1],
        [5, 5, 6, 2, 4, 4, 1, 1],
        [5, 6, 6, 6, 4, 4, 4, 5],
        [5, 5, 7, 7, 7, 7, 7, 5],
        [5, 5, 5, 5, 5, 5, 5, 5],
    ]
)


@pytest.fixture
def screenshot():
    return Image.open(EXAMPLE).convert("RGB")


@pytest.mark.parametrize("name", ["queens.png", "queens_black_navbar.png"])
def test_detects_example_board(name):
    # queens_black_navbar.png: the phone's nav bar is near-black, which used to be
    # mistaken for the board's bottom edge.
    board = detect_board(Image.open(EXAMPLES / name))
    assert board.n == 8
    assert (board.regions == EXAMPLE_REGIONS).all()
    assert board.marked == []


def test_solves_example_end_to_end(screenshot):
    solver = QueensSolver()
    board = solver.parse(screenshot)
    queens = solver.solve(board)
    assert is_valid(board.regions, queens)
    actions = solver.plan(board, queens)
    assert len(actions) == 16 and all(isinstance(a, Tap) for a in actions)
    # Every tap lands inside the board.
    assert all(board.xs[0] <= a.x <= board.xs[-1] and board.ys[0] <= a.y <= board.ys[-1] for a in actions)


def test_refuses_board_with_marks(screenshot):
    board = detect_board(screenshot)
    x, y = board.xs[3], board.ys[3]
    ImageDraw.Draw(screenshot).ellipse((x - 30, y - 30, x + 30, y + 30), fill=(0, 0, 0))
    with pytest.raises(SolverError, match="Reset"):
        QueensSolver().parse(screenshot)


def test_unsolvable_raises():
    # Region 0 is a single cell in the corner touching region 1's only cell.
    regions = np.array([[0, 1, 2, 2], [2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 3, 3]])
    with pytest.raises(SolverError):
        solve_regions(regions)


def test_non_board_screen_raises():
    with pytest.raises(SolverError):
        detect_board(Image.new("RGB", (1080, 2400), (28, 31, 35)))
