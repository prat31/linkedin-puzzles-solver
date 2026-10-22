"""Randomised end-to-end test for the Queens solver.

Each seed builds a random solvable puzzle, draws it the way the LinkedIn app
does (black grid, thicker region borders, pastel regions), and runs it through
parse -> solve -> plan. A tiny simulator then replays the taps (each tap cycles
a cell empty -> X -> crown -> empty) and checks the crowns form a valid answer.

More seeds:  LINPUZS_FUZZ=2000 uv run pytest tests/test_queens_fuzz.py -q
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from linpuzs.actions import Tap
from linpuzs.solvers.queens import COLOR_TOLERANCE, QueensSolver, is_valid

RUNS = int(os.environ.get("LINPUZS_FUZZ", "200"))

# The 8 colours from the real example, plus extras for bigger boards.
PALETTE = [
    (183, 164, 222), (247, 203, 154), (158, 189, 250), (188, 222, 166),
    (223, 223, 223), (238, 130, 104), (232, 243, 149), (184, 178, 160),
    (255, 154, 198), (130, 212, 210), (210, 140, 60),
]  # fmt: skip

SCREENS = [(720, 1600), (1080, 2340), (1080, 2400), (1440, 3120), (1440, 3200)]
BACKGROUNDS = [(28, 31, 35), (244, 242, 238)]  # dark and light mode


def random_queens(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """One queen per row and column with no two touching, found by randomised backtracking."""
    cols: list[int] = []

    def place(r: int) -> bool:
        if r == n:
            return True
        for c in rng.sample(range(n), n):
            if c not in cols and (not cols or abs(cols[-1] - c) > 1):
                cols.append(c)
                if place(r + 1):
                    return True
                cols.pop()
        return False

    assert place(0)
    return list(enumerate(cols))


def random_regions(n: int, queens: list[tuple[int, int]], rng: random.Random) -> np.ndarray:
    """Grow one connected region from each queen until the board is full."""
    regions = np.full((n, n), -1)
    for k, (r, c) in enumerate(queens):
        regions[r, c] = k
    while (regions < 0).any():
        k = rng.randrange(n)
        frontier = [
            (r + dr, c + dc)
            for r, c in zip(*np.nonzero(regions == k))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < n and 0 <= c + dc < n and regions[r + dr, c + dc] < 0
        ]
        if frontier:
            regions[rng.choice(frontier)] = k
    return regions


@dataclass
class FakeScreen:
    image: Image.Image
    regions: np.ndarray
    cell_rects: dict[tuple[int, int], tuple[int, int, int, int]]  # painted (x0, y0, x1, y1), exclusive ends


def render(regions: np.ndarray, rng: random.Random) -> FakeScreen:
    n = len(regions)
    w, h = rng.choice(SCREENS)
    s = w / 1440
    outer, thin, thick = round(15 * s), max(2, round(3 * s)), round(9 * s)
    margin = round(rng.uniform(40, 80) * s)
    size = w - 2 * margin
    left, top = margin, rng.randrange(round(200 * s), h - size - round(200 * s))

    a = np.empty((h, w, 3), dtype=np.int16)
    a[:] = rng.choice(BACKGROUNDS)
    # Full-width near-black bands, like a dark navigation or status bar.
    if rng.random() < 0.5:
        a[h - round(rng.uniform(40, 140) * s) :] = rng.randint(0, 18)
    if rng.random() < 0.3:
        a[: round(rng.uniform(40, 100) * s)] = rng.randint(0, 18)
    a[top : top + size, left : left + size] = 0

    colors = rng.sample(PALETTE, n)
    step = (size - outer) / n
    edges = [round(outer / 2 + i * step) for i in range(n + 1)]  # line centres from the board edge

    def border(i: int, same_region: bool) -> int:
        return outer if i in (0, n) else (thin if same_region else thick)

    rects = {}
    for r in range(n):
        for c in range(n):
            k = regions[r, c]
            x0 = left + edges[c] + (border(c, c > 0 and regions[r, c - 1] == k) + 1) // 2
            x1 = left + edges[c + 1] - border(c + 1, c + 1 < n and regions[r, c + 1] == k) // 2
            y0 = top + edges[r] + (border(r, r > 0 and regions[r - 1, c] == k) + 1) // 2
            y1 = top + edges[r + 1] - border(r + 1, r + 1 < n and regions[r + 1, c] == k) // 2
            a[y0:y1, x0:x1] = colors[k]
            rects[(r, c)] = (x0, y0, x1, y1)

    # A little sensor/compression-style noise so nothing relies on exact values.
    a += np.random.default_rng(rng.randrange(2**32)).integers(-3, 4, a.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return FakeScreen(img, regions, rects)


def play(screen: FakeScreen, taps: list[Tap]) -> list[tuple[int, int]]:
    """Replay taps the way the game handles them; return cells showing a crown."""
    state: dict[tuple[int, int], int] = {}
    for t in taps:
        hit = [cell for cell, (x0, y0, x1, y1) in screen.cell_rects.items() if x0 <= t.x < x1 and y0 <= t.y < y1]
        assert hit, f"tap {t} missed every cell"
        state[hit[0]] = (state.get(hit[0], 0) + 1) % 3
    return sorted(cell for cell, v in state.items() if v == 2)


def test_palette_is_distinguishable():
    for i, a in enumerate(PALETTE):
        for b in PALETTE[i + 1 :]:
            assert np.linalg.norm(np.subtract(a, b)) > COLOR_TOLERANCE + 10, (a, b)


@pytest.mark.parametrize("seed", range(RUNS))
def test_random_board_end_to_end(seed):
    rng = random.Random(seed)
    n = rng.randint(5, 11)
    regions = random_regions(n, random_queens(n, rng), rng)
    screen = render(regions, rng)

    solver = QueensSolver()
    board = solver.parse(screen.image)
    assert board.n == n

    # Region ids may be numbered differently; the partition must be identical.
    pairs = set(zip(board.regions.ravel().tolist(), regions.ravel().tolist()))
    assert len(pairs) == n, "detected regions don't match the drawn ones"

    queens = solver.solve(board)
    assert is_valid(regions, queens)

    actions = solver.plan(board, queens)
    assert all(isinstance(a, Tap) for a in actions)
    assert is_valid(regions, play(screen, actions))
