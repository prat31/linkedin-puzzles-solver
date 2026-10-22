"""Game catalog and solver registry.

To add a solver, create a module in this package and decorate the class:

    @register("queens")
    class QueensSolver(Solver[Board, list[Cell]]):
        ...

Modules here are imported automatically, so nothing else needs wiring up.
"""

from __future__ import annotations

import importlib
import pkgutil

from linpuzs.solvers.base import GameInfo, Solver, SolverError

__all__ = ["GAMES", "GameInfo", "Solver", "SolverError", "get_solver", "register", "solver_for"]

# Every LinkedIn game we know about, in menu order. Ones without a registered
# solver show up as "coming soon".
GAMES: dict[str, GameInfo] = {
    g.slug: g
    for g in [
        GameInfo("queens", "Queens", "One crown per row, column and colour region", "👑"),
        GameInfo("tango", "Tango", "Balance suns and moons across the grid", "🌗"),
        GameInfo("zip", "Zip", "Draw one path through every cell, in order", "〰️"),
        GameInfo("mini-sudoku", "Mini Sudoku", "6×6 sudoku", "🔢"),
        GameInfo("patches", "Patches", "Split the grid into shaped, sized patches", "🧵"),
    ]
}

_REGISTRY: dict[str, type[Solver]] = {}


def register(slug: str):
    if slug not in GAMES:
        raise KeyError(f"Unknown game {slug!r}; add it to GAMES first")

    def deco(cls: type[Solver]) -> type[Solver]:
        cls.game = GAMES[slug]
        _REGISTRY[slug] = cls
        return cls

    return deco


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name != "base":
            importlib.import_module(f"{__name__}.{mod.name}")


def solver_for(slug: str) -> type[Solver] | None:
    return _REGISTRY.get(slug)


def get_solver(slug: str) -> Solver:
    cls = solver_for(slug)
    if cls is None:
        raise KeyError(f"No solver for {slug!r} yet")
    return cls()


_discover()
