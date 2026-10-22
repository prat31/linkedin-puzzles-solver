"""Runs a solver end to end against a device, timing each stage."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from linpuzs.actions import Action
from linpuzs.solvers import Solver
from linpuzs.util.adb import AdbDevice

STAGES = ("screenshot", "parse", "solve", "plan", "perform")


@dataclass
class RunResult:
    solver: Solver
    timings: dict[str, float] = field(default_factory=dict)
    puzzle: Any = None
    solution: Any = None
    actions: list[Action] = field(default_factory=list)
    dry_run: bool = False

    @property
    def total(self) -> float:
        return sum(self.timings.values())


def run(
    solver: Solver,
    device: AdbDevice,
    *,
    dry_run: bool = False,
    on_stage: Callable[[str], None] = lambda _: None,
) -> RunResult:
    result = RunResult(solver=solver, dry_run=dry_run)

    @contextmanager
    def stage(name: str):
        on_stage(name)
        t0 = time.perf_counter()
        yield
        result.timings[name] = time.perf_counter() - t0

    with stage("screenshot"):
        img = device.screenshot()

    with stage("parse"):
        result.puzzle = solver.parse(img)
    with stage("solve"):
        result.solution = solver.solve(result.puzzle)
    with stage("plan"):
        result.actions = solver.plan(result.puzzle, result.solution)
    if not dry_run:
        with stage("perform"):
            device.perform(result.actions)

    return result
