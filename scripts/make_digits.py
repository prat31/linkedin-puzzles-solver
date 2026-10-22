"""Cut 0-9 digit templates (LinkedIn's game font) out of the Zip example screenshot.

    uv run python scripts/make_digits.py

The numbers below are what the example board shows; each dot's glyphs are
paired with the digits of its number. Writes src/linpuzs/util/digits.npz.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from linpuzs.solvers.zip import LINE_RGB, LINE_TOLERANCE, dot_ink
from linpuzs.util.digits import split_glyphs
from linpuzs.util.vision import find_grid

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "media" / "example" / "zip.png"
OUT = ROOT / "src" / "linpuzs" / "util" / "digits.npz"

# (row, col) -> number, as shown in media/example/zip.png
KNOWN = {
    (1, 0): 11, (1, 1): 9, (1, 4): 4, (1, 5): 12,
    (2, 1): 10, (2, 5): 5,
    (3, 0): 8, (3, 4): 3,
    (4, 0): 7, (4, 1): 1, (4, 4): 2, (4, 5): 6,
}  # fmt: skip


def main() -> None:
    a = np.asarray(Image.open(EXAMPLE).convert("RGB"))
    grid = find_grid(np.abs(a.astype(int) - LINE_RGB).max(axis=2) <= LINE_TOLERANCE)
    assert grid is not None
    templates: dict[str, np.ndarray] = {}
    for (r, c), number in KNOWN.items():
        (y0, y1), (x0, x1) = grid.row_spans[r], grid.col_spans[c]
        ink = dot_ink(a[y0:y1, x0:x1])
        glyphs = split_glyphs(ink) if ink is not None else None
        digits = str(number)
        assert glyphs is not None and len(glyphs) == len(digits), (r, c, number)
        for d, g in zip(digits, glyphs):
            templates.setdefault(d, g)
    assert sorted(templates) == list("0123456789"), sorted(templates)
    np.savez_compressed(OUT, **templates)
    print(f"wrote {OUT.relative_to(ROOT)}: " + ", ".join(f"{d} {t.shape[1]}×{t.shape[0]}" for d, t in sorted(templates.items())))


if __name__ == "__main__":
    main()
