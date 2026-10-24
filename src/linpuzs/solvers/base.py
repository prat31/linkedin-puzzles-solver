"""Base class every game solver implements.

A solve runs as a pipeline of four stages, each timed by the runner:

    screenshot ──parse──▶ puzzle ──solve──▶ solution ──plan──▶ actions ──perform──▶ device

Subclasses fill in `parse`, `solve` and `plan`. The puzzle and solution types are
whatever suits the game (a grid, a path, a list of words...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from PIL import Image
from rich.console import RenderableType

from linpuzs.actions import Action

P = TypeVar("P")  # parsed puzzle state
S = TypeVar("S")  # solution


@dataclass(frozen=True)
class GameInfo:
    slug: str
    name: str
    tagline: str
    icon: str = "🧩"
    key: str | None = None  # single-key shortcut in the game menu


class SolverError(Exception):
    """Raised when a stage can't continue (unrecognised screen, unsolvable puzzle...)."""


class Solver(ABC, Generic[P, S]):
    game: ClassVar[GameInfo]

    def __init__(self) -> None:
        # Free-form counters a solver can bump during solve (nodes explored, backtracks...).
        # Shown in the stats table after the run.
        self.metrics: dict[str, Any] = {}

    @abstractmethod
    def parse(self, screenshot: Image.Image) -> P:
        """Read the puzzle state out of a full-screen screenshot."""

    @abstractmethod
    def solve(self, puzzle: P) -> S:
        """Compute a solution. Raise SolverError if there is none."""

    @abstractmethod
    def plan(self, puzzle: P, solution: S) -> list[Action]:
        """Turn the solution into touches, in screenshot pixel coordinates."""

    def render(self, puzzle: P, solution: S | None = None) -> RenderableType | None:
        """Optional pretty preview of the puzzle (and solution) for the terminal."""
        return None
