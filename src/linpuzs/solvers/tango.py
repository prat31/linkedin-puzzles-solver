"""Tango: fill the grid with suns and moons so that

* no more than two of the same symbol are adjacent in a row or column,
* every row and column holds equally many suns and moons,
* cells joined by `=` match and cells joined by `×` differ.

The grid is drawn with flat slate-grey lines; the `=`/`×` markers sit in gaps
in those lines, and are told apart by shape (`=` is wider than it is tall).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from rich.text import Text

from linpuzs.actions import Action, Tap
from linpuzs.solvers import Solver, SolverError, register
from linpuzs.util.vision import find_grid

Cell = tuple[int, int]  # (row, col)

EMPTY, SUN, MOON = 0, 1, 2
# Each tap cycles an empty cell: sun -> moon -> empty.
TAPS = {SUN: 1, MOON: 2}

LINE_RGB = (66, 77, 89)
LINE_TOLERANCE = 12
SYMBOL_FRACTION = 0.03  # share of sun/moon-coloured pixels that means a cell is filled
MARKER_MIN_PIXELS = 0.0005  # of cell area², below this an edge has no marker
EQUALS_ASPECT = 1.25  # marker bounding box width / height above this is `=`


@dataclass
class Board:
    n: int
    cells: np.ndarray  # (n, n) of EMPTY / SUN / MOON
    equal: list[tuple[Cell, Cell]] = field(default_factory=list)
    opposite: list[tuple[Cell, Cell]] = field(default_factory=list)
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)


def _symbol(patch: np.ndarray) -> int:
    r, g, b = (patch[..., i].astype(int) for i in range(3))
    sun = ((r > 200) & (r - b > 100)).mean()
    moon = ((b > 200) & (b - r > 40)).mean()
    if max(sun, moon) < SYMBOL_FRACTION:
        return EMPTY
    return SUN if sun > moon else MOON


def _marker(patch: np.ndarray, min_pixels: int) -> str | None:
    lo, hi = patch.min(axis=2).astype(int), patch.max(axis=2).astype(int)
    mask = (lo > 110) & (hi - lo < 40)  # light grey
    if mask.sum() < min_pixels:
        return None
    ys, xs = np.nonzero(mask)
    aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    return "=" if aspect > EQUALS_ASPECT else "x"


def detect_board(img: Image.Image) -> Board:
    a = np.asarray(img.convert("RGB"))
    line = np.abs(a.astype(int) - LINE_RGB).max(axis=2) <= LINE_TOLERANCE
    grid = find_grid(line)
    if grid is None or grid.n % 2:
        raise SolverError("No Tango board on screen. Open today's Tango with the whole grid visible.")

    n, size = grid.n, grid.cell_size
    cells = np.zeros((n, n), dtype=int)
    for r, (y0, y1) in enumerate(grid.row_spans):
        for c, (x0, x1) in enumerate(grid.col_spans):
            m = size // 5
            cells[r, c] = _symbol(a[y0 + m : y1 - m, x0 + m : x1 - m])

    board = Board(n, cells, xs=grid.xs, ys=grid.ys)
    h, min_px = round(0.18 * size), MARKER_MIN_PIXELS * size * size
    edges = [((r, c), (r, c + 1), (sum(grid.cols[c + 1]) // 2, grid.ys[r])) for r in range(n) for c in range(n - 1)]
    edges += [((r, c), (r + 1, c), (grid.xs[c], sum(grid.rows[r + 1]) // 2)) for r in range(n - 1) for c in range(n)]
    for p, q, (x, y) in edges:
        kind = _marker(a[y - h : y + h, x - h : x + h], min_px)
        if kind == "=":
            board.equal.append((p, q))
        elif kind == "x":
            board.opposite.append((p, q))
    return board


def is_valid(board: Board, cells: np.ndarray) -> bool:
    n = board.n
    if not np.isin(cells, (SUN, MOON)).all():
        return False
    if (board.cells != EMPTY).any() and (cells[board.cells != EMPTY] != board.cells[board.cells != EMPTY]).any():
        return False
    for line in list(cells) + list(cells.T):
        if (line == SUN).sum() != n // 2:
            return False
        if any(line[i] == line[i + 1] == line[i + 2] for i in range(n - 2)):
            return False
    return all(cells[p] == cells[q] for p, q in board.equal) and all(cells[p] != cells[q] for p, q in board.opposite)


def solve_board(board: Board, metrics: dict | None = None) -> np.ndarray:
    """Backtracking in reading order; each placement is checked against the
    cells already filled above and to the left."""
    n, half = board.n, board.n // 2
    grid = board.cells.copy()
    links: dict[Cell, list[tuple[Cell, bool]]] = {}
    for p, q in board.equal:
        links.setdefault(p, []).append((q, True))
        links.setdefault(q, []).append((p, True))
    for p, q in board.opposite:
        links.setdefault(p, []).append((q, False))
        links.setdefault(q, []).append((p, False))
    nodes = 0

    def ok(r: int, c: int) -> bool:
        v = grid[r, c]
        row, col = grid[r], grid[:, c]
        if (row == v).sum() > half or (col == v).sum() > half:
            return False
        for line, i in ((row, c), (col, r)):
            for s in range(max(0, i - 2), min(n - 3, i) + 1):
                if line[s] == line[s + 1] == line[s + 2] == v:
                    return False
        for other, same in links.get((r, c), []):
            if grid[other] != EMPTY and (grid[other] == v) != same:
                return False
        return True

    def search(i: int) -> bool:
        nonlocal nodes
        if i == n * n:
            return True
        r, c = divmod(i, n)
        if board.cells[r, c] != EMPTY:
            return ok(r, c) and search(i + 1)
        for v in (SUN, MOON):
            nodes += 1
            grid[r, c] = v
            if ok(r, c) and search(i + 1):
                return True
        grid[r, c] = EMPTY
        return False

    found = search(0)
    if metrics is not None:
        metrics["nodes explored"] = nodes
    if not found:
        raise SolverError("No valid filling exists. If you placed any symbols yourself, tap Reset and try again.")
    return grid


@register("tango")
class TangoSolver(Solver[Board, np.ndarray]):
    def parse(self, screenshot: Image.Image) -> Board:
        board = detect_board(screenshot)
        self.metrics["grid"] = f"{board.n}×{board.n}"
        self.metrics["givens"] = int((board.cells != EMPTY).sum())
        self.metrics["constraints"] = f"{len(board.equal)} =, {len(board.opposite)} ×"
        return board

    def solve(self, board: Board) -> np.ndarray:
        return solve_board(board, self.metrics)

    def plan(self, board: Board, solution: np.ndarray) -> list[Action]:
        return [
            Tap(board.xs[c], board.ys[r])
            for r in range(board.n)
            for c in range(board.n)
            if board.cells[r, c] == EMPTY
            for _ in range(TAPS[solution[r, c]])
        ]

    def render(self, board: Board, solution: np.ndarray | None = None) -> Text:
        """Suns and moons with the `=`/`×` markers between them; givens are underlined."""
        cells = board.cells if solution is None else solution
        marks = {**{e: "=" for e in board.equal}, **{e: "×" for e in board.opposite}}
        glyph = {EMPTY: ("·", "dim"), SUN: ("☀", "bold #f4b646"), MOON: ("☾", "bold #8baee8")}
        out = Text()
        for r in range(board.n):
            for c in range(board.n):
                ch, style = glyph[int(cells[r, c])]
                out.append(f" {ch} ", style=style + (" underline" if board.cells[r, c] != EMPTY else ""))
                if c < board.n - 1:
                    out.append(marks.get(((r, c), (r, c + 1)), " "), style="bold")
            out.append("\n")
            if r < board.n - 1:
                out.append(" ".join(f" {marks.get(((r, c), (r + 1, c)), ' ')} " for c in range(board.n)) + "\n")
        return out
