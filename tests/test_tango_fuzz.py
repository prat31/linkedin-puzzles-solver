"""Randomised end-to-end test for the Tango solver.

Each seed builds a random valid filling, keeps a few cells as givens and a few
`=`/`×` markers taken from it, draws that the way the LinkedIn app does, and
runs it through parse -> solve -> plan. A tiny simulator then replays the taps
(each tap cycles an empty cell sun -> moon -> empty; givens can't change) and
checks the finished grid obeys every rule.

More seeds:  LINPUZS_FUZZ=2000 uv run pytest tests/test_tango_fuzz.py -q
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image, ImageDraw

from linpuzs.solvers.tango import EMPTY, LINE_RGB, MOON, SUN, Board, TangoSolver, is_valid

RUNS = int(os.environ.get("LINPUZS_FUZZ", "200"))

SCREENS = [(720, 1600), (1080, 2340), (1080, 2400), (1440, 3120), (1440, 3200)]
PAGE_BG, BOARD_BG, GIVEN_BG = (28, 31, 35), (30, 34, 38), (57, 61, 64)
SUN_RGB, MOON_RGB, MARKER_RGB = (244, 182, 70), (139, 174, 232), (170, 182, 190)


def random_filling(n: int, rng: random.Random) -> np.ndarray:
    """A full grid obeying the balance and no-three rules, by randomised backtracking."""
    grid = np.zeros((n, n), dtype=int)

    def fits(r: int, c: int) -> bool:
        v = grid[r, c]
        if (grid[r] == v).sum() > n // 2 or (grid[:, c] == v).sum() > n // 2:
            return False
        return not (c >= 2 and grid[r, c - 1] == grid[r, c - 2] == v) and not (
            r >= 2 and grid[r - 1, c] == grid[r - 2, c] == v
        )

    def fill(i: int) -> bool:
        if i == n * n:
            return True
        r, c = divmod(i, n)
        for v in rng.sample((SUN, MOON), 2):
            grid[r, c] = v
            if fits(r, c) and fill(i + 1):
                return True
        grid[r, c] = EMPTY
        return False

    assert fill(0)
    return grid


def random_puzzle(n: int, rng: random.Random) -> Board:
    full = random_filling(n, rng)
    cells = np.zeros_like(full)
    for i in rng.sample(range(n * n), rng.randint(0, n * n // 3)):
        cells[divmod(i, n)] = full[divmod(i, n)]
    edges = [((r, c), (r, c + 1)) for r in range(n) for c in range(n - 1)]
    edges += [((r, c), (r + 1, c)) for r in range(n - 1) for c in range(n)]
    board = Board(n, cells)
    for p, q in rng.sample(edges, rng.randint(0, 2 * n)):
        (board.equal if full[p] == full[q] else board.opposite).append((p, q))
    return board


@dataclass
class FakeScreen:
    image: Image.Image
    cell_rects: dict[tuple[int, int], tuple[int, int, int, int]]


def render(board: Board, rng: random.Random) -> FakeScreen:
    n = board.n
    w, h = rng.choice(SCREENS)
    s = w / 1440
    lw = max(2, round(6 * s))
    margin = round(rng.uniform(40, 80) * s)
    size = w - 2 * margin
    left, top = margin, rng.randrange(round(200 * s), h - size - round(200 * s))
    step = (size - lw) / n
    edges = [round(i * step) for i in range(n + 1)]  # line starts from the board edge

    img = Image.new("RGB", (w, h), PAGE_BG)
    d = ImageDraw.Draw(img)
    d.rectangle((left, top, left + size - 1, top + size - 1), fill=LINE_RGB)

    rects = {}
    for r in range(n):
        for c in range(n):
            x0, y0 = left + edges[c] + lw, top + edges[r] + lw
            x1, y1 = left + edges[c + 1], top + edges[r + 1]  # exclusive
            d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=GIVEN_BG if board.cells[r, c] else BOARD_BG)
            rects[(r, c)] = (x0, y0, x1, y1)
            cx, cy, rad = (x0 + x1) / 2, (y0 + y1) / 2, 0.23 * (x1 - x0)
            if board.cells[r, c] == SUN:
                d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=SUN_RGB)
            elif board.cells[r, c] == MOON:
                d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=MOON_RGB)
                bg = GIVEN_BG if board.cells[r, c] else BOARD_BG
                d.ellipse((cx - rad * 0.4, cy - rad * 1.2, cx + rad * 1.6, cy + rad * 0.8), fill=bg)

    cell = step - lw
    for kind, pairs in (("=", board.equal), ("x", board.opposite)):
        for p, q in pairs:
            (r, c), (r2, c2) = p, q
            if r == r2:  # marker on the vertical line between two columns
                x, y = left + edges[c + 1] + lw / 2, top + edges[r] + lw + cell / 2
            else:
                x, y = left + edges[c] + lw + cell / 2, top + edges[r + 1] + lw / 2
            g = 0.17 * cell
            d.rectangle((x - g, y - g, x + g, y + g), fill=BOARD_BG)  # gap in the line
            m, t = 0.09 * cell, max(2, round(0.025 * cell))
            if kind == "=":
                for dy in (-0.045 * cell, 0.045 * cell):
                    d.line((x - m, y + dy, x + m, y + dy), fill=MARKER_RGB, width=t)
            else:
                d.line((x - m * 0.8, y - m * 0.8, x + m * 0.8, y + m * 0.8), fill=MARKER_RGB, width=t)
                d.line((x - m * 0.8, y + m * 0.8, x + m * 0.8, y - m * 0.8), fill=MARKER_RGB, width=t)

    a = np.asarray(img).astype(np.int16)
    a += np.random.default_rng(rng.randrange(2**32)).integers(-3, 4, a.shape, dtype=np.int16)
    return FakeScreen(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), rects)


def play(screen: FakeScreen, board: Board, taps) -> np.ndarray:
    grid = board.cells.copy()
    for t in taps:
        hit = [cell for cell, (x0, y0, x1, y1) in screen.cell_rects.items() if x0 <= t.x < x1 and y0 <= t.y < y1]
        assert hit, f"tap {t} missed every cell"
        assert board.cells[hit[0]] == EMPTY, f"tapped given cell {hit[0]}"
        grid[hit[0]] = (grid[hit[0]] + 1) % 3
    return grid


@pytest.mark.parametrize("seed", range(RUNS))
def test_random_board_end_to_end(seed):
    rng = random.Random(seed)
    n = rng.choice([6, 6, 6, 4, 8])
    puzzle = random_puzzle(n, rng)
    screen = render(puzzle, rng)

    solver = TangoSolver()
    board = solver.parse(screen.image)
    assert board.n == n
    assert (board.cells == puzzle.cells).all()
    assert sorted(board.equal) == sorted(puzzle.equal)
    assert sorted(board.opposite) == sorted(puzzle.opposite)

    solution = solver.solve(board)
    assert is_valid(puzzle, solution)
    assert is_valid(puzzle, play(screen, puzzle, solver.plan(board, solution)))
