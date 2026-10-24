"""`linpuzs` entry point."""

from __future__ import annotations

import argparse

import questionary
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from questionary import Choice, Style
from questionary.prompts.common import InquirerControl

from linpuzs import __version__, runner
from linpuzs.solvers import GAMES, SolverError, get_solver, solver_for
from linpuzs.ui import ACCENT, banner, console, device_table, games_table, session_summary, stats_panel
from linpuzs.util.adb import AdbDevice, AdbError, Device, adb_path, list_devices

PROMPT_STYLE = Style(
    [
        ("qmark", f"fg:{ACCENT} bold"),
        ("pointer", f"fg:{ACCENT} bold"),
        ("highlighted", f"fg:{ACCENT} bold"),
        ("selected", "fg:green"),
        ("disabled", "fg:#666666 italic"),
    ]
)

EXIT = "__exit__"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="linpuzs", description="Solve LinkedIn daily puzzles on your phone over ADB.")
    p.add_argument("-s", "--serial", help="ADB device serial to use (skips the device prompt)")
    p.add_argument("-g", "--game", choices=list(GAMES), help="Game to solve (skips the game menu)")
    p.add_argument("-n", "--dry-run", action="store_true", help="Solve but don't send any touches")
    p.add_argument("--list", action="store_true", help="List games and exit")
    p.add_argument("--no-banner", action="store_true", help="Skip the ASCII banner")
    p.add_argument("-V", "--version", action="version", version=f"linpuzs {__version__}")
    return p.parse_args(argv)


def pick_device(serial: str | None) -> AdbDevice | None:
    if adb_path() is None:
        console.print("[red]✗[/] [bold]adb[/] not found on PATH. Install Android platform-tools and try again.")
        return None

    with console.status("Looking for ADB devices…"):
        devices = list_devices()

    ready = [d for d in devices if d.is_ready]
    not_ready = [d for d in devices if not d.is_ready]
    for d in not_ready:
        hint = "accept the USB debugging prompt on the phone" if d.state == "unauthorized" else "reconnect it"
        console.print(f"[yellow]![/] {d.serial} is [yellow]{d.state}[/]: {hint}.")

    if serial:
        match = next((d for d in ready if d.serial == serial), None)
        if match is None:
            console.print(f"[red]✗[/] Device [bold]{serial}[/] is not connected and ready.")
            return None
        return _connected(match)

    if not ready:
        console.print("[red]✗[/] No ADB device found. Plug in your phone with USB debugging enabled.")
        return None
    if len(ready) == 1:
        return _connected(ready[0])

    console.print(device_table(ready))
    chosen = questionary.select(
        "Multiple devices connected. Which one?",
        choices=[Choice(d.label(), value=d) for d in ready],
        style=PROMPT_STYLE,
    ).ask()
    return _connected(chosen) if chosen else None


def _connected(d: Device) -> AdbDevice:
    console.print(f"[green]✓[/] Connected to [bold]{d.model}[/] [dim]({d.serial})[/]")
    return AdbDevice(d)


def pick_game(**prompt_kwargs) -> str | None:
    """Game menu. Each game's key (q, t, z, ...) or e for Exit picks it straight away.

    `prompt_kwargs` go to the underlying prompt (tests pass input/output here).
    """
    choices = []
    for g in GAMES.values():
        ready = solver_for(g.slug) is not None
        choices.append(
            Choice(
                f"{g.icon}  {g.name:<12} {g.tagline}",
                value=g.slug,
                disabled=None if ready else "coming soon",
                shortcut_key=g.key or True,
            )
        )
    choices.append(Choice("👋  Exit", value=EXIT, shortcut_key="e"))
    question = questionary.select(
        "Which puzzle?", choices=choices, style=PROMPT_STYLE, use_shortcuts=True, **prompt_kwargs
    )

    # questionary's shortcuts only move the highlight; make a key press pick the
    # item right away, finishing the same way Enter does (so the answer is echoed).
    app = question.application
    menu = next(c for c in app.layout.find_all_controls() if isinstance(c, InquirerControl))

    def pick(index: int):
        def handler(event):
            menu.pointed_at = index
            menu.is_answered = True
            event.app.exit(result=menu.get_pointed_at().value)

        return handler

    instant = KeyBindings()
    for i, c in enumerate(menu.choices):
        if c.disabled or not isinstance(c.shortcut_key, str):
            continue
        for key in {c.shortcut_key.lower(), c.shortcut_key.upper()}:
            instant.add(key, eager=True)(pick(i))
    app.key_bindings = merge_key_bindings([app.key_bindings, instant])
    return question.ask()


def solve_once(slug: str, device: AdbDevice, args: argparse.Namespace) -> runner.RunResult | None:
    game = GAMES[slug]
    if not args.game:
        ok = questionary.confirm(
            f"Open {game.name} on your phone so the board is visible. Ready?",
            default=True,
            style=PROMPT_STYLE,
        ).ask()
        if not ok:
            return None

    solver = get_solver(slug)
    with console.status(f"{game.icon}  Solving {game.name}…", spinner="dots") as status:
        try:
            result = runner.run(
                solver,
                device,
                dry_run=args.dry_run,
                on_stage=lambda s: status.update(f"{game.icon}  {game.name}: {s}…"),
            )
        except (SolverError, AdbError) as e:
            console.print(f"[red]✗[/] {game.name} failed: {e}")
            return None

    console.print(stats_panel(result))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.no_banner:
        banner()

    if args.list:
        console.print(games_table())
        return 0

    try:
        device = pick_device(args.serial)
    except AdbError as e:
        console.print(f"[red]✗[/] {e}")
        return 1
    if device is None:
        return 1

    if args.game and solver_for(args.game) is None:
        console.print(f"[yellow]![/] No solver for {GAMES[args.game].name} yet.")
        return 1
    if not args.game and not any(solver_for(s) for s in GAMES):
        console.print(games_table())
        console.print("[yellow]![/] No solvers are implemented yet. Check back soon.")
        return 0

    results: list[runner.RunResult] = []
    try:
        if args.game:
            if r := solve_once(args.game, device, args):
                results.append(r)
        else:
            console.print()
            while (slug := pick_game()) not in (None, EXIT):
                if r := solve_once(slug, device, args):
                    results.append(r)
                console.print()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")

    session_summary(results)
    console.print(f"[{ACCENT}]Bye![/]")
    return 0 if results or not args.game else 1
