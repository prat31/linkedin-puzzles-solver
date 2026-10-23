"""Zip: draw one path through every cell, visiting the numbered dots in order
(1, 2, ..., K) and ending on K. Walls between cells can't be crossed.

Reading the board: the grid is drawn with light grey lines; numbered dots are
white discs with black digits, read with `linpuzs.util.digits`. A wall is a
grid-line segment that is much thicker than an ordinary line.

Solving is a depth-first search from dot 1 with pruning (see `solve_path`).
The answer is entered as one continuous drag through every cell centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from rich.text import Text

from linpuzs.actions import Action, Drag
from linpuzs.solvers import Solver, SolverError, register
from linpuzs.util.digits import read_number
from linpuzs.util.vision import find_grid

Cell = tuple[int, int]  # (row, col)
Edge = frozenset[Cell]  # two orthogonally adjacent cells

LINE_RGB = (163, 161, 156)
LINE_TOLERANCE = 14
DOT_MIN = 215  # all channels above this count as the white of a dot
DOT_FRACTION = 0.15  # share of a cell's centre that's dot-white when it holds a dot
INK_MAX = 110  # all channels below this count as digit ink (inside a dot)
WALL_FACTOR = 2.0  # an edge this many times thicker than a normal line is a wall


@dataclass
class Board:
    n: int
    numbers: dict[Cell, int]
    walls: set[Edge] = field(default_factory=set)
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)


# --- digits -------------------------------------------------------------------


def dot_ink(cell: np.ndarray) -> np.ndarray | None:
    """The digit ink inside a cell's dot, or None if the cell has no dot."""
    h, w, _ = cell.shape
    centre = cell[h // 4 : h - h // 4, w // 4 : w - w // 4]
    if (centre.min(axis=2) > DOT_MIN).mean() < DOT_FRACTION:
        return None
    white = cell.min(axis=2) > DOT_MIN
    ys, xs = np.nonzero(white)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    radius = (ys.max() - ys.min()) / 2
    yy, xx = np.mgrid[:h, :w]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 < (0.85 * radius) ** 2
    return (cell.max(axis=2) < INK_MAX) & inside


# --- board --------------------------------------------------------------------


def _edge_thickness(line: np.ndarray, centre: tuple[int, int], vertical_line: bool, reach: int) -> float:
    """Median width of the unbroken run of line-coloured pixels crossing a grid line at `centre`.

    Only the run through the line's centre counts, so a big dot sitting close to
    the line (as on 8×8 boards) doesn't make the edge look thick.
    """
    x, y = centre
    widths = []
    for off in range(-reach // 2, reach // 2 + 1, max(1, reach // 6)):
        strip = line[y + off, x - reach : x + reach + 1] if vertical_line else line[y - reach : y + reach + 1, x + off]
        if not strip[reach]:
            widths.append(0)
            continue
        lo = hi = reach
        while lo > 0 and strip[lo - 1]:
            lo -= 1
        while hi < len(strip) - 1 and strip[hi + 1]:
            hi += 1
        widths.append(hi - lo + 1)
    return float(np.median(widths))


def detect_board(img: Image.Image) -> Board:
    a = np.asarray(img.convert("RGB"))
    line = np.abs(a.astype(int) - LINE_RGB).max(axis=2) <= LINE_TOLERANCE
    grid = find_grid(line)
    if grid is None:
        raise SolverError("No Zip board on screen. Open today's Zip with the whole grid visible.")
    n, size = grid.n, grid.cell_size

    numbers: dict[Cell, int] = {}
    for r, (y0, y1) in enumerate(grid.row_spans):
        for c, (x0, x1) in enumerate(grid.col_spans):
            ink = dot_ink(a[y0:y1, x0:x1])
            if ink is None:
                continue
            number = read_number(ink)
            if number is None:
                raise SolverError(f"Couldn't read the number in row {r + 1}, column {c + 1}.")
            numbers[(r, c)] = number
    if sorted(numbers.values()) != list(range(1, len(numbers) + 1)) or len(numbers) < 2:
        raise SolverError(f"Read numbers {sorted(numbers.values())}; expected 1..K with no gaps.")

    # Walls: compare each inner edge's thickness with the board's normal line width.
    normal = float(np.median([b - a_ + 1 for a_, b in grid.rows + grid.cols]))
    reach = max(4, round(0.12 * size))
    walls: set[Edge] = set()
    for r in range(n):
        for c in range(n - 1):
            x = sum(grid.cols[c + 1]) // 2
            if _edge_thickness(line, (x, grid.ys[r]), True, reach) > WALL_FACTOR * normal:
                walls.add(frozenset({(r, c), (r, c + 1)}))
    for r in range(n - 1):
        for c in range(n):
            y = sum(grid.rows[r + 1]) // 2
            if _edge_thickness(line, (grid.xs[c], y), False, reach) > WALL_FACTOR * normal:
                walls.add(frozenset({(r, c), (r + 1, c)}))

    return Board(n, numbers, walls, grid.xs, grid.ys)


# --- solving ------------------------------------------------------------------


def is_valid(board: Board, path: list[Cell]) -> bool:
    n = board.n
    if len(path) != n * n or len(set(path)) != n * n:
        return False
    for p, q in zip(path, path[1:]):
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) != 1 or frozenset({p, q}) in board.walls:
            return False
    seen = [board.numbers[c] for c in path if c in board.numbers]
    return seen == list(range(1, len(board.numbers) + 1)) and path[-1] in board.numbers and seen[-1] == len(board.numbers)


def solve_path(board: Board, metrics: dict | None = None) -> list[Cell]:
    """Depth-first search from dot 1, one cell at a time.

    A numbered cell may only be entered when its number is next. After every
    step the branch is dropped if
      * the remaining dots can't be joined in order through free cells
        without passing another dot,
      * some unvisited cell can no longer be reached from the path's head, or
      * some unvisited cell has too few free neighbours to be passed through
        (it needs two, or one if it's the final dot), or
      * cells with exactly two options would force more links onto one cell
        than it can take.
    Positions (visited cells + head) that failed once are remembered. Moves are
    tried closest-to-the-next-number first, breaking ties by fewest onward
    options (Warnsdorff).
    """
    n, total = board.n, board.n * board.n
    k_max = len(board.numbers)
    where = {k: cell for cell, k in board.numbers.items()}
    start, end = where[1], where[k_max]

    nbrs: dict[Cell, list[Cell]] = {}
    for r in range(n):
        for c in range(n):
            nbrs[(r, c)] = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < n and 0 <= c + dc < n and frozenset({(r, c), (r + dr, c + dc)}) not in board.walls
            ]

    # Checkerboard parity: each step flips colour, so a path over every cell
    # starts and ends on the same colour exactly when the cell count is odd.
    same_colour = (sum(start) - sum(end)) % 2 == 0
    if same_colour != (total % 2 == 1):
        raise SolverError("Dot 1 and the last dot can't be joined by a path through every cell.")

    visited = {start}
    path = [start]
    nodes = 0

    def joinable(a: Cell, b: Cell) -> bool:
        """Can a reach b through free cells without passing another numbered dot?"""
        seen, stack = {a}, [a]
        while stack:
            u = stack.pop()
            for v in nbrs[u]:
                if v == b:
                    return True
                if v not in seen and v not in visited and v not in board.numbers:
                    seen.add(v)
                    stack.append(v)
        return False

    def hopeless(head: Cell, next_k: int) -> bool:
        free = total - len(visited)
        if free == 0:
            return False
        # The dots still to come must be joinable in order: head -> next, next -> next + 1, ...
        stops = [head] + [where[k] for k in range(next_k, k_max + 1)]
        if not all(joinable(p, q) for p, q in zip(stops, stops[1:])):
            return True
        # Every free cell must be reachable from the head through free cells.
        seen, stack = set(), [v for v in nbrs[head] if v not in visited]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(v for v in nbrs[u] if v not in visited and v not in seen)
        if len(seen) != free:
            return True
        # Every free cell needs an entry and an exit (just an entry for the end).
        # A cell with exactly two options must use both, which pins a link onto
        # each of those neighbours; nobody can take more links than it has left
        # (head: 1 more, end: 1, any other free cell: 2).
        pinned: dict[Cell, int] = {}
        for u in seen:
            options = [v for v in nbrs[u] if v not in visited or v == head]
            if len(options) < (1 if u == end else 2):
                return True
            if u != end and len(options) == 2:
                for v in options:
                    pinned[v] = pinned.get(v, 0) + 1
        return any(count > (1 if v in (head, end) else 2) for v, count in pinned.items())

    bit = {(r, c): 1 << (r * n + c) for r in range(n) for c in range(n)}
    # (visited cells, head) positions already shown to be dead ends. Different
    # routes often reach the same position, so this saves re-exploring it.
    dead: set[tuple[int, Cell]] = set()

    def search(next_k: int, mask: int) -> bool:
        nonlocal nodes
        head = path[-1]
        if len(path) == total:
            return head == end
        if (mask, head) in dead:
            return False
        target = where.get(next_k)
        options = []
        for v in nbrs[head]:
            if v in visited:
                continue
            k = board.numbers.get(v)
            if k is not None and k != next_k:
                continue
            if v == end and len(path) + 1 != total:
                continue
            onward = sum(1 for u in nbrs[v] if u not in visited)
            dist = abs(v[0] - target[0]) + abs(v[1] - target[1]) if target else 0
            options.append((dist, onward, v, k))
        for _, _, v, k in sorted(options):
            nodes += 1
            visited.add(v)
            path.append(v)
            after = next_k + 1 if k is not None else next_k
            if not hopeless(v, after) and search(after, mask | bit[v]):
                return True
            path.pop()
            visited.remove(v)
        dead.add((mask, head))
        return False

    found = search(2, bit[start])
    if metrics is not None:
        metrics["nodes explored"] = nodes
    if not found:
        raise SolverError("No path visits every cell in order. If you started drawing, tap Reset and try again.")
    return list(path)


# --- solver -------------------------------------------------------------------


_GLYPHS = {
    frozenset("UD"): "│", frozenset("LR"): "─", frozenset("DR"): "┌", frozenset("DL"): "┐",
    frozenset("UR"): "└", frozenset("UL"): "┘",
}  # fmt: skip


@register("zip")
class ZipSolver(Solver[Board, list[Cell]]):
    def parse(self, screenshot: Image.Image) -> Board:
        board = detect_board(screenshot)
        self.metrics["grid"] = f"{board.n}×{board.n}"
        self.metrics["dots"] = len(board.numbers)
        self.metrics["walls"] = len(board.walls)
        return board

    def solve(self, board: Board) -> list[Cell]:
        return solve_path(board, self.metrics)

    def plan(self, board: Board, path: list[Cell]) -> list[Action]:
        return [Drag(tuple((board.xs[c], board.ys[r]) for r, c in path))]

    def render(self, board: Board, path: list[Cell] | None = None) -> Text:
        """The path in box-drawing characters, numbers in bold, walls in red."""
        links: set[Edge] = {frozenset(e) for e in zip(path or [], (path or [])[1:])}

        def linked(p: Cell, q: Cell) -> bool:
            return frozenset({p, q}) in links

        out = Text()
        n = board.n
        for r in range(n):
            for c in range(n):
                dirs = {
                    d
                    for d, (dr, dc) in {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}.items()
                    if linked((r, c), (r + dr, c + dc))
                }
                left = "─" if "L" in dirs else " "
                right = "─" if "R" in dirs else " "
                if (r, c) in board.numbers:
                    k = str(board.numbers[(r, c)])
                    out.append(left + k + (right if len(k) == 1 else ""), style="bold black on #eaeaeb")
                else:
                    out.append(left + _GLYPHS.get(frozenset(dirs), "·") + right, style="#e8743b")
                if c < n - 1:
                    if frozenset({(r, c), (r, c + 1)}) in board.walls:
                        out.append("┃", style="bold red")
                    else:
                        out.append("─" if "R" in dirs else " ", style="#e8743b")
            out.append("\n")
            if r < n - 1:
                for c in range(n):
                    if frozenset({(r, c), (r + 1, c)}) in board.walls:
                        out.append("━━━", style="bold red")
                    else:
                        out.append(" │ " if linked((r, c), (r + 1, c)) else "   ", style="#e8743b")
                    if c < n - 1:
                        out.append(" ")
                out.append("\n")
        return out
