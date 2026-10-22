"""Touch actions a solver asks the device to perform, in screen pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


@dataclass(frozen=True)
class Swipe:
    x1: int
    y1: int
    x2: int
    y2: int
    duration_ms: int = 100


@dataclass(frozen=True)
class Drag:
    """A single continuous stroke through a list of points (e.g. drawing a path)."""

    points: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Drag needs at least two points")


@dataclass(frozen=True)
class Wait:
    ms: int


Action = Tap | Swipe | Drag | Wait
