"""Randomised end-to-end test for the Mini Sudoku solver.

Each seed builds a random solved 6×6 grid, blanks out some cells, and draws it
the way the LinkedIn app does: grid with thick box borders, bold digits, and a
number pad (1 2 3 Erase / 4 5 6 Undo) underneath. It then runs parse -> solve
-> plan, and a tiny simulator replays the taps (tapping a cell selects it,
tapping a pad digit fills the selected cell unless it's a given) and checks the
finished grid.

More seeds:  LINPUZS_FUZZ=2000 uv run pytest tests/test_mini_sudoku_fuzz.py -q
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from linpuzs.solvers.mini_sudoku import BUTTON_GREY, LINE_GREY, Board, MiniSudokuSolver, is_valid

RUNS = int(os.environ.get("LINPUZS_FUZZ", "200"))

N, BOX = 6, (2, 3)
SCREENS = [(720, 1600), (1080, 2340), (1080, 2400), (1440, 3120), (1440, 3200)]
PAGE_BG, INK, UNDO_OFF = (28, 31, 35), (240, 240, 240), (45, 46, 48)


def random_solution(rng: random.Random) -> np.ndarray:
    grid = np.zeros((N, N), dtype=int)
    bh, bw = BOX

    def fill(i: int) -> bool:
        if i == N * N:
            return True
        r, c = divmod(i, N)
        br, bc = r - r % bh, c - c % bw
        used = set(grid[r]) | set(grid[:, c]) | set(grid[br : br + bh, bc : bc + bw].ravel())
        for d in rng.sample(range(1, N + 1), N):
            if d not in used:
                grid[r, c] = d
                if fill(i + 1):
                    return True
        grid[r, c] = 0
        return False

    assert fill(0)
    return grid


@dataclass
class FakeScreen:
    image: Image.Image
    cell_rects: dict[tuple[int, int], tuple[int, int, int, int]]
    key_rects: dict[int | str, tuple[int, int, int, int]]


def _centred_text(d: ImageDraw.ImageDraw, box, text: str, font, stroke: int = 0) -> None:
    x0, y0, x1, y1 = box
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font, fill=INK, anchor="mm", stroke_width=stroke, stroke_fill=INK)


def render(cells: np.ndarray, rng: random.Random) -> FakeScreen:
    w, h = rng.choice(SCREENS)
    s = w / 1440
    thin, thick = max(2, round(3 * s)), round(14 * s)
    margin = round(rng.uniform(40, 80) * s)
    size = w - 2 * margin
    left, top = margin, rng.randrange(round(250 * s), round(700 * s))

    img = Image.new("RGB", (w, h), PAGE_BG)
    d = ImageDraw.Draw(img)
    d.rectangle((left, top, left + size - 1, top + size - 1), fill=(LINE_GREY,) * 3)

    bh, bw = BOX
    step = (size - thick) / N
    edges = [round(thick / 2 + i * step) for i in range(N + 1)]  # line centres from the board edge

    def half(i: int, box: int) -> int:
        return thick if i % box == 0 else thin

    digit_font = ImageFont.load_default(size=round(step * 0.45))
    rects = {}
    for r in range(N):
        for c in range(N):
            x0 = left + edges[c] + (half(c, bw) + 1) // 2
            x1 = left + edges[c + 1] - half(c + 1, bw) // 2
            y0 = top + edges[r] + (half(r, bh) + 1) // 2
            y1 = top + edges[r + 1] - half(r + 1, bh) // 2
            d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=PAGE_BG)
            rects[(r, c)] = (x0, y0, x1, y1)
            if cells[r, c]:
                _centred_text(d, (x0, y0, x1, y1), str(cells[r, c]), digit_font, stroke=max(1, round(2 * s)))

    # Number pad: 4 columns, 2 rows, below the grid.
    gap, bwid = round(46 * s), (size - 3 * round(46 * s)) / 4
    bhgt, py = round(225 * s), top + size + round(rng.uniform(150, 260) * s)
    key_font = ImageFont.load_default(size=round(bhgt * 0.4))
    label_font = ImageFont.load_default(size=round(bhgt * 0.15))
    undo_on = rng.random() < 0.5
    keys = {}
    layout = [[1, 2, 3, "erase"], [4, 5, 6, "undo"]]
    for row, labels in enumerate(layout):
        for col, label in enumerate(labels):
            x0 = round(left + col * (bwid + gap))
            y0 = py + row * (bhgt + gap)
            box = (x0, y0, round(x0 + bwid), y0 + bhgt)
            off = label == "undo" and not undo_on
            d.rounded_rectangle(box, radius=round(30 * s), fill=UNDO_OFF if off else (BUTTON_GREY,) * 3)
            if isinstance(label, int):
                _centred_text(d, box, str(label), key_font)
            else:
                _centred_text(d, (box[0], box[1], box[2], box[3] - bhgt // 3), "×" if label == "erase" else "↺", key_font)
                _centred_text(d, (box[0], box[1] + bhgt // 2, box[2], box[3]), label.title(), label_font)
            keys[label] = box

    a = np.asarray(img).astype(np.int16)
    a += np.random.default_rng(rng.randrange(2**32)).integers(-3, 4, a.shape, dtype=np.int16)
    return FakeScreen(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), rects, keys)


def play(screen: FakeScreen, givens: np.ndarray, taps) -> np.ndarray:
    grid, selected = givens.copy(), None
    for t in taps:
        hit = lambda rects: [k for k, (x0, y0, x1, y1) in rects.items() if x0 <= t.x < x1 and y0 <= t.y < y1]
        if cell := hit(screen.cell_rects):
            selected = cell[0]
        elif (key := hit(screen.key_rects)) and isinstance(key[0], int):
            assert selected is not None, "pressed a digit with no cell selected"
            if not givens[selected]:
                grid[selected] = key[0]
        else:
            raise AssertionError(f"tap {t} hit nothing useful ({key or 'nothing'})")
    return grid


@pytest.mark.parametrize("seed", range(RUNS))
def test_random_board_end_to_end(seed):
    rng = random.Random(seed)
    full = random_solution(rng)
    givens = full.copy()
    for i in rng.sample(range(N * N), rng.randint(12, 28)):
        givens[divmod(i, N)] = 0
    screen = render(givens, rng)

    solver = MiniSudokuSolver()
    board = solver.parse(screen.image)
    assert board.box == BOX
    assert (board.cells == givens).all()

    solution = solver.solve(board)
    ref = Board(N, BOX, givens, [], [])
    assert is_valid(ref, solution)
    assert is_valid(ref, play(screen, givens, solver.plan(board, solution)))
