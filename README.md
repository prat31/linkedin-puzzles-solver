# linpuzs: Linkedin Puzzle Solver

For when you want the streak but not the 4 minutes of staring at coloured squares.

Plug in your Android phone, open today's LinkedIn puzzle and run `linpuzs`. It grabs a screenshot over ADB, figures out the board, solves it, and taps the answer in for you, usually in a few seconds.

### Supported games

| Game | Goal | How the answer is entered |
| --- | --- | --- |
| 👑 Queens | One crown in every row, column and colour region, none touching | Double-tap each crown cell |
| 🌗 Tango | Fill with suns and moons: balanced rows and columns, no three in a row, respect `=` and `×` | Tap each empty cell (once for sun, twice for moon) |
| 〰️ Zip | One path through every cell, passing the numbered dots in order | One continuous drag |
| 🔢 Mini Sudoku | 1-6 once in every row, column and 2×3 box | Tap the cell, then the number pad |
| 🧵 Patches | Split the grid into rectangles matching each clue's shape and size | Drag across each patch |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `adb` (Android platform-tools) on your PATH, with USB debugging enabled on the phone.

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

**Dry run:** `linpuzs -n` (or `--dry-run`) reads the board from your phone, solves it and prints the solved board in the terminal, without tapping anything.

## Tests

No phone needed. Each game has two test files:

- `tests/test_<game>.py` checks the real screenshot in `media/example/`.
- `tests/test_<game>_fuzz.py` draws random boards the way the app does, solves them, and replays the touches.

```bash
uv run pytest -q                                      # everything (~1-2 min)
uv run pytest -q tests/test_zip.py                    # one game's screenshot tests
```
