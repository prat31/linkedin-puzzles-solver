"""Reading numbers drawn in LinkedIn's game font.

Digits are matched against 0-9 templates (digits.npz) cut from the numbered
dots in media/example/zip.png. The same font is used by several games, with dark
digits on light (Zip) or light digits on colour (Patches); callers pass a
boolean ink mask, so colour doesn't matter here.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import numpy as np
from PIL import Image, ImageFilter

GLYPH_SIZE = (20, 30)
GLYPH_BLUR = 1.0
MIN_SCORE, MIN_MARGIN = 0.6, 0.07  # how sure a digit match has to be


def normalise(ink: np.ndarray) -> np.ndarray:
    crop = Image.fromarray(ink.astype(np.uint8) * 255)
    small = crop.resize(GLYPH_SIZE, Image.BILINEAR).filter(ImageFilter.GaussianBlur(GLYPH_BLUR))
    return np.asarray(small, dtype=float) / 255


def similarity(u: np.ndarray, v: np.ndarray) -> float:
    u, v = u - u.mean(), v - v.mean()
    return float((u * v).sum() / (np.sqrt((u * u).sum() * (v * v).sum()) or 1))


def split_glyphs(ink: np.ndarray) -> list[np.ndarray]:
    """Split an ink mask into tightly cropped glyphs, left to right, at empty columns."""
    cols = np.nonzero(ink.any(axis=0))[0]
    if len(cols) == 0:
        return []
    spans, start = [], cols[0]
    for a, b in zip(cols, cols[1:]):
        if b > a + 1:
            spans.append((start, a + 1))
            start = b
    spans.append((start, cols[-1] + 1))
    glyphs = []
    for x0, x1 in spans:
        g = ink[:, x0:x1]
        rows = np.nonzero(g.any(axis=1))[0]
        glyphs.append(g[rows[0] : rows[-1] + 1])
    return glyphs


@cache
def _templates() -> dict[int, tuple[np.ndarray, float]]:
    """digit -> (normalised glyph, width / height)."""
    with resources.files("linpuzs.util").joinpath("digits.npz").open("rb") as f:
        data = np.load(f)
        return {int(k): (normalise(data[k]), data[k].shape[1] / data[k].shape[0]) for k in data.files}


def read_digit(ink: np.ndarray) -> int | None:
    """Best-matching digit, or None if no template matches clearly."""
    glyph, aspect = normalise(ink), ink.shape[1] / ink.shape[0]
    # Squashing every glyph to one size hides how narrow a "1" is, so add the
    # aspect-ratio difference back as a penalty.
    scores = sorted(
        ((similarity(glyph, t) - abs(aspect - ta), d) for d, (t, ta) in _templates().items()), reverse=True
    )
    (best, digit), (second, _) = scores[:2]
    return digit if best >= MIN_SCORE and best - second >= MIN_MARGIN else None


def read_number(ink: np.ndarray) -> int | None:
    """Read a whole (possibly multi-digit) number, or None if any digit is unclear."""
    glyphs = split_glyphs(ink)
    digits = [read_digit(g) for g in glyphs]
    if not digits or None in digits:
        return None
    return int("".join(map(str, digits)))
