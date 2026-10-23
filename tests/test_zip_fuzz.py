"""Randomised end-to-end test for the Zip solver.

Each seed builds a random path through every cell (backbite moves on a snake),
puts numbered dots along it, adds a few walls the path doesn't cross, and draws
the board the way the LinkedIn app does. Digits are drawn with the solver's own
templates (cut from a real screenshot), so this exercises grid, dot, number
splitting, wall and solver logic rather than font matching; the real-font
check is test_zip.py.

A tiny simulator replays the drag: every point must land inside a cell, each
step must move to an adjacent cell without crossing a wall or revisiting, and
the finished path must be a valid answer.

More seeds:  LINPUZS_FUZZ=2000 uv run pytest tests/test_zip_fuzz.py -q
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pytest
from PIL import Image, ImageDraw

from linpuzs.actions import Drag
from linpuzs.solvers.zip import LINE_RGB, Board, ZipSolver, is_valid

RUNS = int(os.environ.get("LINPUZS_FUZZ", "200"))

SCREENS = [(720, 1600), (1080, 2340), (1080, 2400), (1440, 3120), (1440, 3200)]
PAGE_BG, DOT_RGB = (28, 31, 35), (234, 234, 235)
RING_COLOURS = [(221, 101, 47), (160, 60, 170)]  # dot 1's ring: orange or purple seen so far

with resources.files("linpuzs.util").joinpath("digits.npz").open("rb") as f:
    DIGITS = {k: v for k, v in np.load(f).items()}


def random_path(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """A random Hamiltonian path: start from a snake and apply backbite moves."""
    path = [(r, c if r % 2 == 0 else n - 1 - c) for r in range(n) for c in range(n)]
    for _ in range(n * n * 20):
        if rng.random() < 0.5:
            path.reverse()
        r, c = path[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= r + dr < n and 0 <= c + dc < n]
        v = rng.choice(nbrs)
        i = path.index(v)
        if i == len(path) - 2:
            continue
        path = path[: i + 1] + path[i + 1 :][::-1]
    return path


def random_puzzle(rng: random.Random) -> tuple[Board, list[tuple[int, int]]]:
    n = rng.randint(5, 8)
    path = random_path(n, rng)
    inner = sorted(rng.sample(range(1, n * n - 1), rng.randint(2, min(14, n * n - 2))))
    idx = [0] + inner + [n * n - 1]
    numbers = {path[i]: k for k, i in enumerate(idx, start=1)}
    used = {frozenset(e) for e in zip(path, path[1:])}
    edges = [frozenset({(r, c), (r, c + 1)}) for r in range(n) for c in range(n - 1)]
    edges += [frozenset({(r, c), (r + 1, c)}) for r in range(n - 1) for c in range(n)]
    walls = set(rng.sample([e for e in edges if e not in used], rng.randint(0, n)))
    return Board(n, numbers, walls), path


@dataclass
class FakeScreen:
    image: Image.Image
    cell_rects: dict[tuple[int, int], tuple[int, int, int, int]]


def _paste_number(a: np.ndarray, k: int, cx: float, cy: float, height: int) -> None:
    glyphs = []
    for d in str(k):
        t = DIGITS[d]
        w = max(1, round(t.shape[1] * height / t.shape[0]))
        glyphs.append(np.asarray(Image.fromarray(t.astype(np.uint8) * 255).resize((w, height), Image.NEAREST)) > 127)
    gap = max(2, height // 8)
    total = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1)
    x = round(cx - total / 2)
    y = round(cy - height / 2)
    for g in glyphs:
        a[y : y + height, x : x + g.shape[1]][g] = 0
        x += g.shape[1] + gap


def render(board: Board, rng: random.Random) -> FakeScreen:
    n = board.n
    w, h = rng.choice(SCREENS)
    s = w / 1440
    lw = max(2, round(6 * s))
    margin = round(rng.uniform(40, 80) * s)
    size = w - 2 * margin
    left, top = margin, rng.randrange(round(250 * s), h - size - round(250 * s))
    step = (size - lw) / n
    edges = [round(i * step) for i in range(n + 1)]
    cell = step - lw

    img = Image.new("RGB", (w, h), PAGE_BG)
    d = ImageDraw.Draw(img)
    for i in range(1, n):
        d.rectangle((left + edges[i], top, left + edges[i] + lw - 1, top + size - 1), fill=LINE_RGB)
        d.rectangle((left, top + edges[i], left + size - 1, top + edges[i] + lw - 1), fill=LINE_RGB)
    d.rounded_rectangle((left, top, left + size - 1, top + size - 1), radius=round(0.35 * cell), outline=LINE_RGB, width=lw)

    rects, centres = {}, {}
    for r in range(n):
        for c in range(n):
            x0, y0 = left + edges[c] + lw, top + edges[r] + lw
            x1, y1 = left + edges[c + 1], top + edges[r + 1]
            rects[(r, c)] = (x0, y0, x1, y1)
            centres[(r, c)] = ((x0 + x1) / 2, (y0 + y1) / 2)

    wall_w = max(lw * 3, round(0.08 * cell))
    for e in board.walls:
        (r, c), (r2, c2) = sorted(e)
        if r == r2:  # vertical wall between columns c and c + 1
            x = left + edges[c + 1] + lw / 2
            d.rectangle((x - wall_w / 2, top + edges[r], x + wall_w / 2, top + edges[r + 1] + lw), fill=LINE_RGB)
        else:
            y = top + edges[r + 1] + lw / 2
            d.rectangle((left + edges[c], y - wall_w / 2, left + edges[c + 1] + lw, y + wall_w / 2), fill=LINE_RGB)

    # Dots are ~0.63 of a cell on small boards and up to ~0.82 on 8×8 ones,
    # where they come close to the grid lines.
    rad = rng.uniform(0.31, 0.41) * cell
    ring_rgb = rng.choice(RING_COLOURS)
    for cellpos, k in board.numbers.items():
        cx, cy = centres[cellpos]
        if k == 1:
            ring = rad + 0.045 * cell
            d.ellipse((cx - ring, cy - ring, cx + ring, cy + ring), fill=ring_rgb)
        d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=DOT_RGB)

    a = np.asarray(img).copy()
    for cellpos, k in board.numbers.items():
        _paste_number(a, k, *centres[cellpos], height=max(10, round(0.224 * cell)))
    a = a.astype(np.int16) + np.random.default_rng(rng.randrange(2**32)).integers(-3, 4, a.shape, dtype=np.int16)
    return FakeScreen(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), rects)


def play(screen: FakeScreen, board: Board, drag: Drag) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for x, y in drag.points:
        hit = [cell for cell, (x0, y0, x1, y1) in screen.cell_rects.items() if x0 <= x < x1 and y0 <= y < y1]
        assert hit, f"drag point {(x, y)} missed every cell"
        cell = hit[0]
        if path:
            p = path[-1]
            assert abs(p[0] - cell[0]) + abs(p[1] - cell[1]) == 1, f"jumped from {p} to {cell}"
            assert frozenset({p, cell}) not in board.walls, f"crossed wall {p}-{cell}"
            assert cell not in path, f"revisited {cell}"
        path.append(cell)
    return path


@pytest.mark.parametrize("seed", range(RUNS))
def test_random_board_end_to_end(seed):
    rng = random.Random(seed)
    puzzle, _ = random_puzzle(rng)
    screen = render(puzzle, rng)

    solver = ZipSolver()
    board = solver.parse(screen.image)
    assert board.n == puzzle.n
    assert board.numbers == puzzle.numbers
    assert board.walls == puzzle.walls

    path = solver.solve(board)
    assert is_valid(puzzle, path)
    (drag,) = solver.plan(board, path)
    assert is_valid(puzzle, play(screen, puzzle, drag))
