"""Screenshot helpers shared by grid-based games."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Run = tuple[int, int]  # inclusive (start, end) pixel range of one grid line

LINE_FRACTION = 0.5  # a grid line covers at least this much of the scan width


@dataclass(frozen=True)
class Grid:
    rows: list[Run]  # horizontal lines, top to bottom
    cols: list[Run]  # vertical lines, left to right

    @property
    def n(self) -> int:
        return len(self.rows) - 1

    @property
    def row_spans(self) -> list[tuple[int, int]]:
        """Cell interiors as [start, end) y ranges, between consecutive lines."""
        return [(a[1] + 1, b[0]) for a, b in zip(self.rows, self.rows[1:])]

    @property
    def col_spans(self) -> list[tuple[int, int]]:
        return [(a[1] + 1, b[0]) for a, b in zip(self.cols, self.cols[1:])]

    @property
    def ys(self) -> list[int]:
        return [(a + b) // 2 for a, b in self.row_spans]

    @property
    def xs(self) -> list[int]:
        return [(a + b) // 2 for a, b in self.col_spans]

    @property
    def cell_size(self) -> int:
        return min(b - a for a, b in self.row_spans + self.col_spans)


def line_runs(counts: np.ndarray, threshold: float) -> list[Run]:
    """Group consecutive indices with counts >= threshold into (start, end) runs."""
    runs, start = [], None
    for i, hit in enumerate(counts >= threshold):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(counts) - 1))
    return runs


def find_grid(line: np.ndarray, min_n: int = 4, fraction: float = LINE_FRACTION) -> Grid | None:
    """Find a square board drawn with grid lines, given a mask of line-coloured pixels.

    Other bands of the same colour (a black navigation bar, dark wallpaper) can
    also look like lines, so try every pair of candidate rows as the board's top
    and bottom and keep the biggest one that forms a square grid of evenly sized
    cells whose lines stop at the board's edge.

    `fraction` is how much of the width a line must cover; lower it for dashed lines.
    """
    w = line.shape[1]
    rows = line_runs(line.sum(axis=1), fraction * w)
    best: Grid | None = None
    for i in range(len(rows)):
        for j in range(i + min_n, len(rows)):
            if best and j - i <= best.n:
                continue
            sub = rows[i : j + 1]
            top, bottom = sub[0][0], sub[-1][1]
            cols = line_runs(line[top : bottom + 1].sum(axis=0), fraction * (bottom - top))
            if len(cols) != len(sub):
                continue
            left, right = cols[0][0], cols[-1][1]
            if abs((bottom - top) - (right - left)) > 0.05 * (right - left):
                continue
            # Grid lines stop at the board's edge; a bar across the screen runs past it.
            mids = [(a + b) // 2 for a, b in sub]
            if (left > 2 and line[mids, left - 3].any()) or (right < w - 3 and line[mids, right + 3].any()):
                continue
            grid = Grid(sub, cols)
            sizes = np.array([b - a for a, b in grid.row_spans + grid.col_spans])
            if sizes.min() >= 0.7 * sizes.max():
                best = grid
    return best
