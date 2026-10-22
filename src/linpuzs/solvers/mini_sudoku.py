"""Mini Sudoku: fill a 6×6 grid so every row, column and 2×3 box holds 1-6 once.

Digits are read without OCR: the on-screen number pad shows 1-6 in the app's
own font, so each pad glyph becomes a template and every digit in the grid is
matched against them. The thick grid lines mark the box borders, so the box
shape comes from the screenshot too.

Entering a digit is two taps: select the cell, then press the number on the pad.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter
from rich.text import Text

from linpuzs.actions import Action, Tap
from linpuzs.solvers import Solver, SolverError, register
from linpuzs.util.vision import find_grid, line_runs

Cell = tuple[int, int]  # (row, col)
Point = tuple[int, int]  # (x, y) in screen pixels

LINE_GREY, LINE_TOLERANCE = 153, 12
BUTTON_GREY, BUTTON_TOLERANCE = 77, 8
GLYPH_MIN = 150  # all channels above this count as digit ink
GLYPH_SIZE = (20, 30)  # glyphs are compared after scaling to this (w, h)
GLYPH_BLUR = 1.5  # softens stroke weight: grid digits are bold, pad digits aren't
MIN_SCORE, MIN_MARGIN = 0.6, 0.07  # how sure a digit match has to be


@dataclass
class Board:
    n: int
    box: tuple[int, int]  # (rows, cols) per box
    cells: np.ndarray  # (n, n), 0 = empty
    xs: list[int]
    ys: list[int]
    keys: dict[int, Point] = field(default_factory=dict)  # digit -> number-pad button centre


def _glyph(patch: np.ndarray) -> np.ndarray | None:
    """Crop the bright ink in a patch and scale it to GLYPH_SIZE, or None if blank."""
    ink = patch.min(axis=2) > GLYPH_MIN
    ys, xs = np.nonzero(ink)
    if len(ys) < 30:
        return None
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    small = Image.fromarray(crop).resize(GLYPH_SIZE, Image.BILINEAR).filter(ImageFilter.GaussianBlur(GLYPH_BLUR))
    return np.asarray(small, dtype=float) / 255


def _similarity(u: np.ndarray, v: np.ndarray) -> float:
    u, v = u - u.mean(), v - v.mean()
    return float((u * v).sum() / (np.sqrt((u * u).sum() * (v * v).sum()) or 1))


def _box_shape(runs: list[tuple[int, int]]) -> int:
    """Cells per box along one axis: the spacing between the thick lines."""
    widths = [b - a + 1 for a, b in runs]
    thick = [i for i, wd in enumerate(widths) if wd > 1.5 * min(widths)]
    if len(thick) < 2:
        return len(runs) - 1
    return thick[1] - thick[0]


def _number_pad(a: np.ndarray, below: int, n: int) -> dict[int, tuple[Point, np.ndarray]]:
    """Find the digit buttons under the grid; return digit -> (centre, glyph template).

    The pad is laid out in reading order (1 2 3 / 4 5 6) with Erase and Undo in
    an extra right-hand column, which is dropped.
    """
    btn = (np.abs(a[below:].astype(int) - BUTTON_GREY).max(axis=2) <= BUTTON_TOLERANCE)
    w = btn.shape[1]
    buttons = []
    for y0, y1 in line_runs(btn.sum(axis=1), 0.3 * w):
        for x0, x1 in line_runs(btn[y0 : y1 + 1].sum(axis=0), 0.5 * (y1 - y0)):
            if x1 - x0 > 0.5 * (y1 - y0):
                buttons.append((x0, y0 + below, x1, y1 + below))
    if len(buttons) > n:
        right = max(b[0] for b in buttons)
        buttons = [b for b in buttons if b[0] < right - 10]
    if len(buttons) != n:
        raise SolverError("Couldn't find the 1-6 number pad below the grid. Scroll so it's fully visible.")

    pad = {}
    for digit, (x0, y0, x1, y1) in enumerate(buttons, start=1):
        glyph = _glyph(a[y0:y1, x0:x1])
        if glyph is None:
            raise SolverError(f"Number pad button {digit} looks blank.")
        pad[digit] = (((x0 + x1) // 2, (y0 + y1) // 2), glyph)
    return pad


def detect_board(img: Image.Image) -> Board:
    a = np.asarray(img.convert("RGB"))
    line = np.abs(a.astype(int) - LINE_GREY).max(axis=2) <= LINE_TOLERANCE
    grid = find_grid(line)
    if grid is None:
        raise SolverError("No Mini Sudoku board on screen. Open today's puzzle with the grid and number pad visible.")
    n = grid.n
    box = (_box_shape(grid.rows), _box_shape(grid.cols))
    if box[0] * box[1] != n:
        raise SolverError(f"Box borders look wrong ({box[0]}×{box[1]} boxes on a {n}×{n} grid).")

    pad = _number_pad(a, grid.rows[-1][1] + 1, n)
    cells = np.zeros((n, n), dtype=int)
    for r, (y0, y1) in enumerate(grid.row_spans):
        for c, (x0, x1) in enumerate(grid.col_spans):
            m = (y1 - y0) // 10
            glyph = _glyph(a[y0 + m : y1 - m, x0 + m : x1 - m])
            if glyph is None:
                continue
            (best, digit), (second, _) = sorted(
                ((_similarity(glyph, t), d) for d, (_, t) in pad.items()), reverse=True
            )[:2]
            if best < MIN_SCORE or best - second < MIN_MARGIN:
                raise SolverError(f"Couldn't read the digit in row {r + 1}, column {c + 1}.")
            cells[r, c] = digit

    return Board(n, box, cells, grid.xs, grid.ys, {d: p for d, (p, _) in pad.items()})


def is_valid(board: Board, cells: np.ndarray) -> bool:
    n, (bh, bw) = board.n, board.box
    full = set(range(1, n + 1))
    given = board.cells != 0
    if (cells[given] != board.cells[given]).any():
        return False
    groups = list(cells) + list(cells.T)
    groups += [cells[r : r + bh, c : c + bw].ravel() for r in range(0, n, bh) for c in range(0, n, bw)]
    return all(set(g.tolist()) == full for g in groups)


def solve_board(board: Board, metrics: dict | None = None) -> np.ndarray:
    """Backtracking that always fills the empty cell with the fewest candidates."""
    n, (bh, bw) = board.n, board.box
    grid = board.cells.copy()
    nodes = 0

    def candidates(r: int, c: int) -> set[int]:
        br, bc = r - r % bh, c - c % bw
        used = set(grid[r].tolist()) | set(grid[:, c].tolist()) | set(grid[br : br + bh, bc : bc + bw].ravel().tolist())
        return set(range(1, n + 1)) - used

    def search() -> bool:
        nonlocal nodes
        empty = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0]
        if not empty:
            return True
        options = {cell: candidates(*cell) for cell in empty}
        cell = min(options, key=lambda k: len(options[k]))
        for d in sorted(options[cell]):
            nodes += 1
            grid[cell] = d
            if search():
                return True
        grid[cell] = 0
        return False

    # Givens must not already clash (e.g. a wrong digit typed in by hand).
    groups = list(grid) + list(grid.T)
    groups += [grid[r : r + bh, c : c + bw].ravel() for r in range(0, n, bh) for c in range(0, n, bw)]
    consistent = all(len(set(g[g != 0].tolist())) == (g != 0).sum() for g in groups)

    found = consistent and search()
    if metrics is not None:
        metrics["nodes explored"] = nodes
    if not found:
        raise SolverError("No valid solution. If you entered any digits yourself, tap Reset and try again.")
    return grid


@register("mini-sudoku")
class MiniSudokuSolver(Solver[Board, np.ndarray]):
    def parse(self, screenshot: Image.Image) -> Board:
        board = detect_board(screenshot)
        self.metrics["grid"] = f"{board.n}×{board.n}, {board.box[0]}×{board.box[1]} boxes"
        self.metrics["givens"] = int((board.cells != 0).sum())
        return board

    def solve(self, board: Board) -> np.ndarray:
        return solve_board(board, self.metrics)

    def plan(self, board: Board, solution: np.ndarray) -> list[Action]:
        actions: list[Action] = []
        for r in range(board.n):
            for c in range(board.n):
                if board.cells[r, c] == 0:
                    actions += [Tap(board.xs[c], board.ys[r]), Tap(*board.keys[int(solution[r, c])])]
        return actions

    def render(self, board: Board, solution: np.ndarray | None = None) -> Text:
        """The grid with box borders; givens in bold white, filled-in digits in blue."""
        cells = board.cells if solution is None else solution
        bh, bw = board.box
        rule = "─┼─".join("─" * (2 * bw - 1) for _ in range(board.n // bw))
        out = Text()
        for r in range(board.n):
            if r and r % bh == 0:
                out.append(rule + "\n", style="dim")
            for c in range(board.n):
                if c and c % bw == 0:
                    out.append(" │ ", style="dim")
                elif c:
                    out.append(" ")
                d = int(cells[r, c])
                given = board.cells[r, c] != 0
                out.append(str(d) if d else "·", style="bold white" if given else ("#8baee8" if d else "dim"))
            out.append("\n")
        return out
