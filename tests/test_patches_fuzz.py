"""Randomised end-to-end test for the Patches solver.

Each seed splits the grid into random rectangles, puts a clue badge in each
(its shape matching the rectangle, sometimes the "any" cross, sometimes with no
number), and draws the board the way the LinkedIn app does: dashed inner lines,
solid rounded border, coloured badges with white digits. Digits come from the
real-font templates, so font reading is covered by test_patches.py instead.

A tiny simulator replays the actions (a drag makes the rectangle spanning its
first and last cells; a tap makes a 1×1 patch) and checks the result.

More seeds:  LINPUZS_FUZZ=2000 uv run pytest tests/test_patches_fuzz.py -q
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pytest
from PIL import Image, ImageDraw

from linpuzs.actions import Drag, Tap
from linpuzs.solvers.patches import (
    ANY,
    INNER_GREY,
    OUTER_RGB,
    SQUARE,
    TALL,
    WIDE,
    Board,
    Clue,
    PatchesSolver,
    Rect,
    is_valid,
    rect_cells,
)  # fmt: skip

RUNS = int(os.environ.get("LINPUZS_FUZZ", "200"))

SCREENS = [(720, 1600), (1080, 2340), (1080, 2400), (1440, 3120), (1440, 3200)]
PAGE_BG = (28, 31, 35)
# Badge colours from the real example.
PALETTE = [
    (152, 84, 34), (209, 45, 31), (86, 62, 170), (47, 105, 118), (188, 146, 48), (74, 163, 90),
    (78, 172, 248), (68, 81, 90), (48, 58, 117), (156, 30, 157), (122, 87, 52),
]  # fmt: skip

with resources.files("linpuzs.util").joinpath("digits.npz").open("rb") as f:
    DIGITS = {k: v for k, v in np.load(f).items()}


def random_puzzle(rng: random.Random) -> tuple[Board, list[Rect]]:
    """Fill the grid with random rectangles, first empty cell first."""
    n = rng.randint(5, 9)
    owner = np.full((n, n), -1)
    rects: list[Rect] = []
    while (owner < 0).any():
        r, c = map(int, np.argwhere(owner < 0)[0])
        max_w = 0
        while c + max_w < n and owner[r, c + max_w] < 0:
            max_w += 1
        w = rng.randint(1, min(max_w, 4))
        h = 1
        while r + h < n and h < 4 and (owner[r + h, c : c + w] < 0).all() and rng.random() < 0.7:
            h += 1
        owner[r : r + h, c : c + w] = len(rects)
        rects.append((r, c, h, w))
    clues = {}
    for top, left, h, w in rects:
        cell = (top + rng.randrange(h), left + rng.randrange(w))
        shape = SQUARE if h == w else TALL if h > w else WIDE
        if rng.random() < 0.4:
            shape = ANY
        number = h * w if rng.random() < 0.85 else None
        clues[cell] = Clue(shape, number, rng.choice(PALETTE))
    return Board(n, clues), rects


@dataclass
class FakeScreen:
    image: Image.Image
    cell_rects: dict[tuple[int, int], tuple[int, int, int, int]]


def _dashed(d: ImageDraw.ImageDraw, x0, y0, x1, y1, width, dash, gap):
    length = max(x1 - x0, y1 - y0)
    pos = 0.0
    while pos < length:
        seg = min(dash, length - pos)
        if x1 - x0 >= y1 - y0:
            d.rectangle((x0 + pos, y0, x0 + pos + seg - 1, y0 + width - 1), fill=(INNER_GREY,) * 3)
        else:
            d.rectangle((x0, y0 + pos, x0 + width - 1, y0 + pos + seg - 1), fill=(INNER_GREY,) * 3)
        pos += dash + gap


def _paste_number(a: np.ndarray, k: int, cx: float, cy: float, height: int) -> None:
    glyphs = []
    for d in str(k):
        t = DIGITS[d]
        w = max(1, round(t.shape[1] * height / t.shape[0]))
        glyphs.append(np.asarray(Image.fromarray(t.astype(np.uint8) * 255).resize((w, height), Image.NEAREST)) > 127)
    gap = max(2, height // 8)
    total = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1)
    x, y = round(cx - total / 2), round(cy - height / 2)
    for g in glyphs:
        a[y : y + height, x : x + g.shape[1]][g] = 255
        x += g.shape[1] + gap


def render(board: Board, rng: random.Random) -> FakeScreen:
    n = board.n
    w, h = rng.choice(SCREENS)
    s = w / 1440
    outer, inner = round(7 * s), max(2, round(3 * s))
    margin = round(rng.uniform(40, 80) * s)
    size = w - 2 * margin
    left, top = margin, rng.randrange(round(250 * s), h - size - round(250 * s))
    step = (size - outer) / n
    starts = [round(outer + i * step) for i in range(n + 1)]  # cell interiors start here (+ inner line for i > 0)

    img = Image.new("RGB", (w, h), PAGE_BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((left, top, left + size - 1, top + size - 1), radius=round(25 * s), outline=OUTER_RGB, width=outer)
    dash, gap = round(12 * s) or 2, round(8 * s) or 1
    for i in range(1, n):
        pos = starts[i] - inner // 2
        _dashed(d, left + outer, top + pos, left + size - outer, top + pos, inner, dash, gap)
        _dashed(d, left + pos, top + outer, left + pos, top + size - outer, inner, dash, gap)

    rects, centres = {}, {}
    for r in range(n):
        for c in range(n):
            x0 = left + starts[c] + (inner // 2 + 1 if c else 0)
            x1 = left + (starts[c + 1] - inner // 2 if c + 1 < n else size - outer)
            y0 = top + starts[r] + (inner // 2 + 1 if r else 0)
            y1 = top + (starts[r + 1] - inner // 2 if r + 1 < n else size - outer)
            rects[(r, c)] = (x0, y0, x1, y1)
            centres[(r, c)] = ((x0 + x1) / 2, (y0 + y1) / 2)

    cell = step - inner
    rad = round(0.06 * cell)
    for pos, clue in board.clues.items():
        cx, cy = centres[pos]
        if clue.shape == ANY:
            for bw, bh in ((0.75, 0.45), (0.45, 0.75)):
                d.rounded_rectangle((cx - bw * cell / 2, cy - bh * cell / 2, cx + bw * cell / 2, cy + bh * cell / 2), radius=rad, fill=clue.color)
        else:
            bw, bh = {SQUARE: (0.62, 0.62), TALL: (0.51, 0.76), WIDE: (0.76, 0.51)}[clue.shape]
            d.rounded_rectangle((cx - bw * cell / 2, cy - bh * cell / 2, cx + bw * cell / 2, cy + bh * cell / 2), radius=rad, fill=clue.color)

    a = np.asarray(img).copy()
    for pos, clue in board.clues.items():
        if clue.number is not None:
            _paste_number(a, clue.number, *centres[pos], height=max(10, round(0.26 * cell)))
    a = a.astype(np.int16) + np.random.default_rng(rng.randrange(2**32)).integers(-3, 4, a.shape, dtype=np.int16)
    return FakeScreen(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), rects)


def play(screen: FakeScreen, actions) -> list[Rect]:
    def cell_at(x, y):
        hit = [p for p, (x0, y0, x1, y1) in screen.cell_rects.items() if x0 <= x < x1 and y0 <= y < y1]
        assert hit, f"point {(x, y)} missed every cell"
        return hit[0]

    made: list[Rect] = []
    for act in actions:
        if isinstance(act, Tap):
            (r, c) = cell_at(act.x, act.y)
            made.append((r, c, 1, 1))
        else:
            assert isinstance(act, Drag)
            (r0, c0), (r1, c1) = cell_at(*act.points[0]), cell_at(*act.points[-1])
            made.append((min(r0, r1), min(c0, c1), abs(r1 - r0) + 1, abs(c1 - c0) + 1))
    covered = [p for rect in made for p in rect_cells(rect)]
    assert len(covered) == len(set(covered)), "patches overlap"
    return made


def obeys_rules(board: Board, rects: list[Rect]) -> bool:
    """The rules written out again, independently of the solver's own helpers."""
    seen = set()
    for top, left, h, w in rects:
        cells = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
        if cells & seen:
            return False
        seen |= cells
        clue_cells = [p for p in cells if p in board.clues]
        if len(clue_cells) != 1:
            return False
        clue = board.clues[clue_cells[0]]
        if clue.number is not None and clue.number != h * w:
            return False
        if (clue.shape == SQUARE and h != w) or (clue.shape == TALL and h <= w) or (clue.shape == WIDE and w <= h):
            return False
    return len(seen) == board.n * board.n


@pytest.mark.parametrize("seed", range(RUNS))
def test_random_board_end_to_end(seed):
    rng = random.Random(seed)
    puzzle, _ = random_puzzle(rng)
    screen = render(puzzle, rng)

    solver = PatchesSolver()
    board = solver.parse(screen.image)
    assert board.n == puzzle.n
    assert {p: (c.shape, c.number) for p, c in board.clues.items()} == {
        p: (c.shape, c.number) for p, c in puzzle.clues.items()
    }

    rects = solver.solve(board)
    assert is_valid(puzzle, rects) and obeys_rules(puzzle, rects)
    assert obeys_rules(puzzle, play(screen, solver.plan(board, rects)))
