import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from linpuzs.cli import EXIT, pick_game
from linpuzs.solvers import GAMES

DOWN, ENTER = "\x1b[B", "\r"


def menu(keys: str) -> str | None:
    with create_pipe_input() as inp:
        inp.send_text(keys)
        return pick_game(input=inp, output=DummyOutput())


@pytest.mark.parametrize("slug", list(GAMES))
def test_game_key_picks_it_straight_away(slug):
    assert menu(GAMES[slug].key) == slug


def test_keys_are_case_insensitive():
    assert menu("T") == "tango"


@pytest.mark.parametrize("key", ["e", "E"])
def test_e_exits(key):
    assert menu(key) == EXIT


def test_arrow_keys_and_enter_still_work():
    assert menu(DOWN + ENTER) == list(GAMES)[1]


def test_every_game_has_a_unique_key():
    keys = [g.key for g in GAMES.values()]
    assert all(keys) and len(set(keys)) == len(keys) and "e" not in keys
