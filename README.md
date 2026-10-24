# linpuzs: Linkedin Puzzle Solver

For when you want the streak but not the 4 minutes of staring at coloured squares.

Plug in your Android phone, open today's LinkedIn puzzle and run `linpuzs`. It grabs a screenshot over ADB, figures out the board, solves it, and taps the answer in for you, usually in a few seconds.

![linpuzs solving Queens, Tango, Mini Sudoku and Patches on a phone](https://raw.githubusercontent.com/prat31/linkedin-puzzles-solver/main/media/demo.gif)

### Supported games

| Game | Goal |
| --- | --- |
| 👑 Queens | One crown in every row, column and colour region, none touching |
| 🌗 Tango | Fill with suns and moons: balanced rows and columns, no three in a row, respect `=` and `×` |
| 〰️ Zip | One path through every cell, passing the numbered dots in order |
| 🔢 Mini Sudoku | 1-6 once in every row, column and 2×3 box |
| 🧵 Patches | Split the grid into rectangles matching each clue's shape and size |

## Install

You'll need Python 3.10+, `adb` (Android platform-tools) on your PATH, and USB debugging turned on in your phone's developer options.

With [pipx](https://pipx.pypa.io/) (recommended, keeps it in its own environment):

```bash
pipx install linpuzs
```

Or with pip:

```bash
pip install linpuzs
```

Or with Homebrew (macOS/Linux):

```bash
brew install prat31/tap/linpuzs
```

Then plug in your phone and run `linpuzs`. To update later, use `pipx upgrade linpuzs`, `pip install -U linpuzs` or `brew upgrade linpuzs`.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and `adb` on your PATH.

```bash
uv sync
uv run linpuzs
```

## Usage

```
linpuzs                      interactive: pick device (if several) → pick game → solve
linpuzs -g queens            solve one game without the menu
linpuzs -s SERIAL            use a specific device
linpuzs -n                   dry run: solve, but don't touch the screen
linpuzs --list               show games and solver status
```

In the game menu, press a game's key to start it straight away: `q` Queens, `t` Tango, `z` Zip, `m` Mini Sudoku, `p` Patches, or `e` to exit. Arrow keys and Enter work too.

**Dry run:** `linpuzs -n` (or `--dry-run`) reads the board from your phone, solves it and prints the solved board in the terminal, without tapping anything.

## Tests

No phone needed. Each game has two test files:

- `tests/test_<game>.py` checks the real screenshot in `media/example/`.
- `tests/test_<game>_fuzz.py` draws random boards the way the app does, solves them, and replays the touches.

```bash
uv run pytest -q                                      # everything (~1-2 min)
uv run pytest -q tests/test_zip.py                    # one game's screenshot tests
```
