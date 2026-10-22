"""Patches: split the grid into rectangles ("patches") so that every patch holds
exactly one clue badge. The badge's shape fixes the patch's shape (square, tall,
wide, or the dashed cross meaning any of those), and a number on the badge is
the patch's area.

Reading the board: inner grid lines are dashed grey, the outer border solid. A
cell holds a clue when part of it differs from the background; the badge's
bounding box and how solidly it's filled give its shape, and its white digits
are read with `linpuzs.util.digits`.

Solving is an exact cover over each clue's candidate rectangles. Each patch is
entered as one drag between opposite corner cells, starting from the corner
nearest its clue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from rich.text import Text

from linpuzs.actions import Action, Drag, Tap
from linpuzs.solvers import Solver, SolverError, register
from linpuzs.util.digits import read_number
from linpuzs.util.vision import find_grid

Cell = tuple[int, int]  # (row, col)
Rect = tuple[int, int, int, int]  # (top, left, height, width) in cells

SQUARE, TALL, WIDE, ANY = "square", "tall", "wide", "any"

INNER_GREY, INNER_TOLERANCE = 140, 12  # dashed inner lines
OUTER_RGB, OUTER_TOLERANCE = (56, 58, 61), 6  # solid outer border
DASHED_FRACTION = 0.35  # dashed lines cover only part of a row
BADGE_DIFF = 20  # channel difference from the background that counts as badge
BADGE_FRACTION = 0.05  # share of a cell's interior a badge covers, at least
ELONGATED = 1.25  # bounding box ratio beyond which a badge is tall or wide
SOLID_FILL = 0.93  # a squarish badge filled more than this is a square; less is the dashed "any" cross
INK_MIN = 200  # all channels above this count as (white) digit ink


@dataclass(frozen=True)
class Clue:
    shape: str
    number: int | None
    color: tuple[int, int, int] = (128, 128, 128)


@dataclass
class Board:
    n: int
    clues: dict[Cell, Clue]
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)


def rect_cells(rect: Rect) -> list[Cell]:
    top, left, h, w = rect
    return [(r, c) for r in range(top, top + h) for c in range(left, left + w)]


def fits(clue: Clue, h: int, w: int) -> bool:
    if clue.number is not None and h * w != clue.number:
        return False
    return {SQUARE: h == w, TALL: h > w, WIDE: w > h, ANY: True}[clue.shape]


# --- board --------------------------------------------------------------------


def _line_mask(a: np.ndarray) -> np.ndarray:
    ai = a.astype(int)
    return (np.abs(ai - INNER_GREY).max(axis=2) <= INNER_TOLERANCE) | (
        np.abs(ai - OUTER_RGB).max(axis=2) <= OUTER_TOLERANCE
    )


def _badge(cell: np.ndarray, bg: np.ndarray) -> Clue | None:
    # Anything off-background that isn't a stray bit of grid line.
    mask = (np.abs(cell.astype(int) - bg).max(axis=2) > BADGE_DIFF) & ~_line_mask(cell)
    if mask.mean() < BADGE_FRACTION:
        return None
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if h > ELONGATED * w:
        shape = TALL
    elif w > ELONGATED * h:
        shape = WIDE
    else:
        shape = SQUARE if mask.sum() / (h * w) > SOLID_FILL else ANY

    ink = cell.min(axis=2) > INK_MIN
    colour = tuple(int(v) for v in np.median(cell[mask & ~ink], axis=0))
    if ink.sum() < 20:
        return Clue(shape, None, colour)
    number = read_number(ink)
    if number is None:
        raise SolverError("Couldn't read a number on one of the badges.")
    return Clue(shape, number, colour)


def detect_board(img: Image.Image) -> Board:
    a = np.asarray(img.convert("RGB"))
    grid = find_grid(_line_mask(a), fraction=DASHED_FRACTION)
    if grid is None:
        raise SolverError("No Patches board on screen. Open today's Patches with the whole grid visible.")
    n = grid.n

    board_px = a[grid.rows[0][1] : grid.rows[-1][0] : 4, grid.cols[0][1] : grid.cols[-1][0] : 4]
    bg = np.median(board_px.reshape(-1, 3), axis=0)
    inset = max(2, grid.cell_size // 16)  # stay clear of the grid lines; badges sit well inside

    clues: dict[Cell, Clue] = {}
    for r, (y0, y1) in enumerate(grid.row_spans):
        for c, (x0, x1) in enumerate(grid.col_spans):
            try:
                clue = _badge(a[y0 + inset : y1 - inset, x0 + inset : x1 - inset], bg)
            except SolverError:
                raise SolverError(f"Couldn't read the number in row {r + 1}, column {c + 1}.") from None
            if clue is not None:
                clues[(r, c)] = clue

    numbers = [cl.number for cl in clues.values() if cl.number is not None]
    if not clues or sum(numbers) > n * n or (len(numbers) == len(clues) and sum(numbers) != n * n):
        raise SolverError(f"Clue numbers add up to {sum(numbers)} on a {n}×{n} grid; something was misread.")
    return Board(n, clues, grid.xs, grid.ys)


# --- solving ------------------------------------------------------------------


def candidates(board: Board, cell: Cell) -> list[Rect]:
    """Every rectangle that could be this clue's patch."""
    n, clue = board.n, board.clues[cell]
    r0, c0 = cell
    out = []
    for h in range(1, n + 1):
        for w in range(1, n + 1):
            if not fits(clue, h, w):
                continue
            for top in range(max(0, r0 - h + 1), min(r0, n - h) + 1):
                for left in range(max(0, c0 - w + 1), min(c0, n - w) + 1):
                    rect = (top, left, h, w)
                    if all(p == cell or p not in board.clues for p in rect_cells(rect)):
                        out.append(rect)
    return out


def is_valid(board: Board, rects: list[Rect]) -> bool:
    covered: list[Cell] = [p for rect in rects for p in rect_cells(rect)]
    if sorted(covered) != [(r, c) for r in range(board.n) for c in range(board.n)]:
        return False
    owners = []
    for rect in rects:
        inside = [p for p in rect_cells(rect) if p in board.clues]
        if len(inside) != 1 or not fits(board.clues[inside[0]], rect[2], rect[3]):
            return False
        owners.append(inside[0])
    return sorted(owners) == sorted(board.clues)


def solve_patches(board: Board, metrics: dict | None = None) -> list[Rect]:
    """Exact cover by backtracking: always place the clue with the fewest
    rectangles that still fit, and give up on a branch as soon as some empty
    cell can't be covered by any remaining clue's options."""
    n = board.n
    full = (1 << (n * n)) - 1

    def mask(rect: Rect) -> int:
        return sum(1 << (r * n + c) for r, c in rect_cells(rect))

    options = {cell: [(mask(rect), rect) for rect in candidates(board, cell)] for cell in board.clues}
    chosen: list[Rect] = []
    nodes = 0

    def search(covered: int, remaining: frozenset[Cell]) -> bool:
        nonlocal nodes
        if not remaining:
            return covered == full
        live = {cell: [o for o in options[cell] if not o[0] & covered] for cell in remaining}
        reach = covered
        for opts in live.values():
            if not opts:
                return False
            for m, _ in opts:
                reach |= m
        if reach != full:
            return False
        cell = min(live, key=lambda k: len(live[k]))
        for m, rect in live[cell]:
            nodes += 1
            chosen.append(rect)
            if search(covered | m, remaining - {cell}):
                return True
            chosen.pop()
        return False

    found = search(0, frozenset(board.clues))
    if metrics is not None:
        metrics["nodes explored"] = nodes
    if not found:
        raise SolverError("No way to split the grid. If you drew any patches, tap Reset and try again.")
    return sorted(chosen)


# --- solver -------------------------------------------------------------------


_SHAPE_GLYPH = {SQUARE: "□", TALL: "▯", WIDE: "▭", ANY: "✣"}


@register("patches")
class PatchesSolver(Solver[Board, list[Rect]]):
    def parse(self, screenshot: Image.Image) -> Board:
        board = detect_board(screenshot)
        self.metrics["grid"] = f"{board.n}×{board.n}"
        self.metrics["clues"] = len(board.clues)
        return board

    def solve(self, board: Board) -> list[Rect]:
        return solve_patches(board, self.metrics)

    def plan(self, board: Board, rects: list[Rect]) -> list[Action]:
        actions: list[Action] = []
        for rect in rects:
            top, left, h, w = rect
            if h == w == 1:
                actions.append(Tap(board.xs[left], board.ys[top]))
                continue
            (cr, cc), = (p for p in rect_cells(rect) if p in board.clues)
            # Start from the corner nearest the clue, end at the opposite one.
            r0 = top if cr - top <= top + h - 1 - cr else top + h - 1
            c0 = left if cc - left <= left + w - 1 - cc else left + w - 1
            r1, c1 = 2 * top + h - 1 - r0, 2 * left + w - 1 - c0
            steps = max(h, w)
            points = tuple(
                (
                    round(board.xs[c0] + (board.xs[c1] - board.xs[c0]) * i / steps),
                    round(board.ys[r0] + (board.ys[r1] - board.ys[r0]) * i / steps),
                )
                for i in range(steps + 1)
            )
            actions.append(Drag(points))
        return actions

    def render(self, board: Board, rects: list[Rect] | None = None) -> Text:
        """Each patch in its badge's colour and a letter; clue cells show their number or shape."""
        owner: dict[Cell, tuple[int, Clue]] = {}
        for i, rect in enumerate(rects or []):
            (clue_cell,) = (p for p in rect_cells(rect) if p in board.clues)
            for p in rect_cells(rect):
                owner[p] = (i, board.clues[clue_cell])
        out = Text()
        for r in range(board.n):
            for c in range(board.n):
                clue = board.clues.get((r, c))
                if (r, c) in owner:
                    i, owner_clue = owner[(r, c)]
                    style = f"bold white on rgb{owner_clue.color}"
                    label = chr(ord("a") + i % 26)
                else:
                    style, label = "dim", "·"
                if clue is not None:
                    label = str(clue.number) if clue.number is not None else _SHAPE_GLYPH[clue.shape]
                    style += " underline"
                out.append(f" {label:>2} ", style=style)
            out.append("\n")
        return out
