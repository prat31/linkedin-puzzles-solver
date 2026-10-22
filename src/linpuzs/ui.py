"""Terminal presentation: banner, device/game tables, run stats."""

from __future__ import annotations

from collections import Counter

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from linpuzs import __version__
from linpuzs.runner import STAGES, RunResult
from linpuzs.solvers import GAMES, solver_for
from linpuzs.util.adb import Device

console = Console(highlight=False)

ACCENT = "#0a66c2"  # LinkedIn blue

_LOGO = r"""
██╗     ██╗███╗   ██╗██████╗ ██╗   ██╗███████╗███████╗
██║     ██║████╗  ██║██╔══██╗██║   ██║╚══███╔╝██╔════╝
██║     ██║██╔██╗ ██║██████╔╝██║   ██║  ███╔╝ ███████╗
██║     ██║██║╚██╗██║██╔═══╝ ██║   ██║ ███╔╝  ╚════██║
███████╗██║██║ ╚████║██║     ╚██████╔╝███████╗███████║
╚══════╝╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚══════╝╚══════╝
""".strip("\n")

_GRADIENT = ["#70b5f9", "#4a9eef", "#2b88e0", "#0a66c2", "#0a55a3", "#084585"]


def banner() -> None:
    logo = Text()
    for line, color in zip(_LOGO.splitlines(), _GRADIENT):
        logo.append(line + "\n", style=f"bold {color}")
    subtitle = Text.assemble(
        ("linkedin puzzle solver", "bold white"),
        ("  ·  ", "dim"),
        (f"v{__version__}", "dim"),
    )
    console.print()
    console.print(Align.center(logo))
    console.print(Align.center(subtitle))
    console.print()


def device_table(devices: list[Device]) -> Table:
    t = Table(title="ADB devices", title_style="bold", border_style=ACCENT, header_style="bold")
    t.add_column("Serial")
    t.add_column("Model")
    t.add_column("State")
    for d in devices:
        state = Text(d.state, style="green" if d.is_ready else "yellow")
        t.add_row(d.serial, d.model, state)
    return t


def games_table() -> Table:
    t = Table(title="Games", title_style="bold", border_style=ACCENT, header_style="bold")
    t.add_column("")
    t.add_column("Game", style="bold")
    t.add_column("Description")
    t.add_column("Status")
    for g in GAMES.values():
        ready = solver_for(g.slug) is not None
        status = Text("● ready", style="green") if ready else Text("○ coming soon", style="dim")
        t.add_row(g.icon, g.name, g.tagline, status)
    return t


def _fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:,.0f} ms" if ms >= 10 else f"{ms:.2f} ms"


def stats_panel(result: RunResult) -> Panel:
    game = result.solver.game
    total = result.total or 1e-9

    timings = Table.grid(padding=(0, 2))
    timings.add_column(style="bold")
    timings.add_column(justify="right")
    timings.add_column()
    for name in STAGES:
        if name not in result.timings:
            timings.add_row(name, Text("skipped", style="dim"), "")
            continue
        secs = result.timings[name]
        bar = "█" * max(1, round(secs / total * 24))
        timings.add_row(name, _fmt_ms(secs), Text(bar, style=ACCENT))
    timings.add_row(Text("total", style="bold green"), Text(_fmt_ms(result.total), style="bold green"), "")

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column()
    kinds = Counter(type(a).__name__.lower() for a in result.actions)
    facts.add_row("touches", ", ".join(f"{n} {k}" for k, n in kinds.items()) or "none")
    for k, v in result.solver.metrics.items():
        facts.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    if result.dry_run:
        facts.add_row("mode", Text("dry run: no touches sent", style="yellow"))

    parts = [timings, Text(""), facts]
    preview = result.solver.render(result.puzzle, result.solution)
    if preview is not None:
        parts = [preview, Text("")] + parts

    return Panel(
        Group(*parts),
        title=f"{game.icon}  {game.name} solved",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    )


def session_summary(results: list[RunResult]) -> None:
    if not results:
        return
    t = Table(title="Session", title_style="bold", border_style=ACCENT, header_style="bold")
    t.add_column("Game")
    t.add_column("Touches", justify="right")
    t.add_column("Time", justify="right")
    for r in results:
        t.add_row(f"{r.solver.game.icon} {r.solver.game.name}", str(len(r.actions)), _fmt_ms(r.total))
    console.print(t)
