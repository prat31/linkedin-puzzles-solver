"""Queens: place one crown in every row, column and colour region, with no two
crowns touching (diagonals included).

Board detection relies on the grid being drawn with pure-black lines: rows and
columns of the screenshot that are mostly black are grid lines, and the cell
centres sit between them. Each cell's colour is the median of its interior,
and cells with matching colours form a region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from rich.text import Text

from linpuzs.actions import Action, Tap
from linpuzs.solvers import Solver, SolverError, register
from linpuzs.util.vision import find_grid

Cell = tuple[int, int]  # (row, col)

LINE_MAX = 20  # a channel value at or below this counts as grid-line black
COLOR_TOLERANCE = 30.0  # max RGB distance between cells of the same region
MARK_FRACTION = 0.02  # share of dark pixels in a cell's centre that means it holds an X or crown


@dataclass
class Board:
    n: int
    regions: np.ndarray  # (n, n) region id per cell
    colors: list[tuple[int, int, int]]  # region id -> RGB
    xs: list[int]  # column centres in screen pixels
    ys: list[int]  # row centres in screen pixels
    marked: list[Cell]  # cells that already hold an X or crown


def _group_colors(samples: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Greedy colour clustering: each sample joins the first centre within tolerance."""
    centres: list[np.ndarray] = []
    labels = np.empty(len(samples), dtype=int)
    for i, c in enumerate(samples):
        for j, centre in enumerate(centres):
            if np.linalg.norm(c - centre) <= COLOR_TOLERANCE:
                labels[i] = j
                break
        else:
            labels[i] = len(centres)
            centres.append(c)
    return labels, centres


def detect_board(img: Image.Image) -> Board:
    a = np.asarray(img.convert("RGB"))
    black = a.max(axis=2) <= LINE_MAX

    grid = find_grid(black)
    if grid is None:
        raise SolverError("No Queens board on screen. Open today's Queens with the whole grid visible.")
    n = grid.n
    ys_span, xs_span = grid.row_spans, grid.col_spans

    samples, marked = [], []
    for r, (y0, y1) in enumerate(ys_span):
        for c, (x0, x1) in enumerate(xs_span):
            cell = a[y0:y1, x0:x1].reshape(-1, 3)
            samples.append(np.median(cell, axis=0))
            ch, cw = y1 - y0, x1 - x0
            centre = a[y0 + ch // 4 : y1 - ch // 4, x0 + cw // 4 : x1 - cw // 4]
            if (centre.max(axis=2) < 80).mean() > MARK_FRACTION:
                marked.append((r, c))

    labels, centres = _group_colors(np.array(samples))
    if len(centres) != n:
        raise SolverError(f"Found {len(centres)} colour regions on a {n}×{n} board; expected {n}.")

    return Board(
        n=n,
        regions=labels.reshape(n, n),
        colors=[tuple(int(v) for v in c) for c in centres],
        xs=grid.xs,
        ys=grid.ys,
        marked=marked,
    )


def is_valid(regions: np.ndarray, queens: list[Cell]) -> bool:
    n = len(regions)
    if len(queens) != n:
        return False
    rs, cs = {r for r, _ in queens}, {c for _, c in queens}
    regs = {int(regions[r][c]) for r, c in queens}
    if len(rs) != n or len(cs) != n or len(regs) != n:
        return False
    return all(
        max(abs(r1 - r2), abs(c1 - c2)) > 1 for i, (r1, c1) in enumerate(queens) for r2, c2 in queens[i + 1 :]
    )


def solve_regions(regions: np.ndarray, metrics: dict | None = None) -> list[Cell]:
    """Backtracking over colour regions, always branching on the region with the
    fewest legal cells left."""
    n = len(regions)
    cells_of = [[(r, c) for r in range(n) for c in range(n) if regions[r][c] == k] for k in range(n)]
    used_rows, used_cols = [False] * n, [False] * n
    queens: list[Cell] = []
    nodes = 0

    def legal(cell: Cell) -> bool:
        r, c = cell
        return (
            not used_rows[r]
            and not used_cols[c]
            and all(max(abs(r - qr), abs(c - qc)) > 1 for qr, qc in queens)
        )

    def search(remaining: set[int]) -> bool:
        nonlocal nodes
        if not remaining:
            return True
        options = {k: [cell for cell in cells_of[k] if legal(cell)] for k in remaining}
        region = min(options, key=lambda k: len(options[k]))
        for r, c in options[region]:
            nodes += 1
            used_rows[r] = used_cols[c] = True
            queens.append((r, c))
            if search(remaining - {region}):
                return True
            queens.pop()
            used_rows[r] = used_cols[c] = False
        return False

    found = search(set(range(n)))
    if metrics is not None:
        metrics["nodes explored"] = nodes
    if not found:
        raise SolverError("No valid placement exists. Did the board get read correctly?")
    return sorted(queens)


@register("queens")
class QueensSolver(Solver[Board, list[Cell]]):
    def parse(self, screenshot: Image.Image) -> Board:
        board = detect_board(screenshot)
        if board.marked:
            raise SolverError(f"{len(board.marked)} cell(s) already have an X or crown. Tap Reset and try again.")
        self.metrics["grid"] = f"{board.n}×{board.n}"
        return board

    def solve(self, board: Board) -> list[Cell]:
        return solve_regions(board.regions, self.metrics)

    def plan(self, board: Board, queens: list[Cell]) -> list[Action]:
        # First tap places an X, the second turns it into a crown.
        return [Tap(board.xs[c], board.ys[r]) for r, c in queens for _ in range(2)]

    def render(self, board: Board, queens: list[Cell] | None = None) -> Text:
        """Region letters on their colours, ♛ where a crown goes. Reads fine without colour too."""
        crowns = set(queens or [])
        out = Text()
        for r in range(board.n):
            for c in range(board.n):
                k = board.regions[r][c]
                bg = f"on rgb{board.colors[k]}"
                if (r, c) in crowns:
                    out.append(" ♛ ", style=f"bold black {bg}")
                else:
                    out.append(f" {chr(ord('a') + k)} ", style=f"#555555 {bg}")
            out.append("\n")
        return out
