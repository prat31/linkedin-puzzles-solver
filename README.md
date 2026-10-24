# linpuzs: LinkedIn Puzzles Solver

Plug in your Android phone, open today's LinkedIn puzzle and run `linpuzs`. It grabs a screenshot over ADB, figures out the board, solves it, and taps the answer in for you, in a few seconds.

![linpuzs solving Queens, Tango, Mini Sudoku and Patches on a phone](https://raw.githubusercontent.com/prat31/linkedin-puzzles-solver/main/media/demo.gif)

## Supported games

| Game | Goal |
| --- | --- |
| 👑 Queens | Place one crown in every row, column and colour region, with no two touching |
| 🌗 Tango | Fill the grid with suns and moons: balanced rows and columns, no three in a row, respecting `=` and `×` |
| 〰️ Zip | Draw one path through every cell, passing the numbered dots in order |
| 🔢 Mini Sudoku | Fill every row, column and 2×3 box with 1–6 |
| 🧵 Patches | Split the grid into rectangles matching each clue's shape and size |

## Install

You'll need `adb` (Android platform-tools) on your PATH and USB debugging turned on in your phone's developer options.

With [pipx](https://pipx.pypa.io/) (recommended; needs Python 3.10+):

```bash
pipx install linpuzs
```

With pip (Python 3.10+):

```bash
pip install linpuzs
```

With Homebrew (macOS/Linux):

```bash
brew install prat31/tap/linpuzs
```

To update, run `pipx upgrade linpuzs`, `pip install -U linpuzs` or `brew upgrade linpuzs`.

## Usage

Open a puzzle on your phone, then:

```bash
linpuzs               # pick a device (if several), pick a game, solve it
linpuzs -g queens     # solve one game, skipping the menu
linpuzs -s SERIAL     # use a specific device
linpuzs -n            # dry run: solve and print the answer, but don't touch the phone
linpuzs --list        # list games and whether they're supported
```

In the game menu, press a game's key to start it straight away: `q` Queens, `t` Tango, `z` Zip, `m` Mini Sudoku, `p` Patches, or `e` to exit. Arrow keys and Enter work too.

A dry run (`-n` or `--dry-run`) reads the board from your phone, solves it and prints the solved board in the terminal, without tapping anything.

## Development

You'll need [uv](https://docs.astral.sh/uv/) and `adb` on your PATH.

```bash
uv sync               # install dependencies
uv run linpuzs        # run from source
```

### Tests

No phone needed. Each game has two test files:

- `tests/test_<game>.py` checks the real screenshots in `media/example/`.
- `tests/test_<game>_fuzz.py` draws random boards the way the app does, solves them, and replays the touches.

```bash
uv run pytest -q                     # everything (~1-2 min)
uv run pytest -q tests/test_zip.py   # one game's screenshot tests
```
