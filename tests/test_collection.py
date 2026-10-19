import re

import pandas as pd
import pytest

from balatro_analysis import collection, metrics
from balatro_analysis.game_text import (
    UNKNOWN_HTML,
    GameText,
    _Evaluator,
    _Obj,
    format_line,
    parse_loc_vars,
    starting_ability,
)
from balatro_analysis.names import Names
from balatro_analysis.sprites import Sprites


def _text(lines: list[str]) -> str:
    return " / ".join(re.sub(r"<[^>]+>", "", line) for line in lines)


# --- evaluator / parsing ---------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("self.ability.mult", 4),
        ("self.ability.extra.odds * 2 + 1", 2001),
        ("''..(G.GAME and G.GAME.probabilities.normal or 1)", "1"),  # Lua and/or
        ("0 and 5 or 7", 5),  # 0 is truthy in Lua
        ("nil or 3", 3),
        ("'a'..1.0", "a1"),
        ("math.max(2, self.ability.x_mult)", 2),
    ],
)
def test_evaluator_lua_semantics(expr, expected):
    ability = starting_ability({"config": {"mult": 4, "extra": {"odds": 1000}}})
    names = {"self": _Obj(ability=ability), "G": _Obj(GAME=None), "math": _Obj(max=max)}
    assert _Evaluator(names).run(expr) == expected


def test_evaluator_refuses_unknown_names_and_calls():
    ev = _Evaluator({"self": _Obj(ability=_Obj())})
    for expr in ("self.ability.missing", "os.execute('x')", "__import__('os')", "G.GAME.x"):
        with pytest.raises(Exception):  # noqa: B017 - any refusal is fine
            ev.run(expr)


def test_parse_loc_vars_branches():
    from conftest import FAKE_CARD_LUA

    lv = parse_loc_vars(FAKE_CARD_LUA)
    assert lv["Joker"] == ["self.ability.mult"]
    assert lv["Lusty Joker"] == lv["Greedy Joker"]  # 'or' branches share expressions
    assert len(lv["Cavendish"]) == 3
    assert len(lv["Castle"]) == 2  # keyed 'colours = {...}' dropped


def test_format_line_colours_badges_and_escaping():
    html = format_line("{X:mult,C:white} X#1# {} Mult {C:chips}<b>#2#{}", [3, None])
    assert "background:#FE5F55" in html and "color:#ffffff" in html
    assert " X3 " in html
    assert "&lt;b&gt;" in html  # text is escaped
    assert UNKNOWN_HTML in html  # unknown value


# --- game text from an archive --------------------------------------------


def test_game_text(fake_game):
    gt = GameText.load(fake_game / "Balatro.exe")
    assert _text(gt.joker("j_joker").lines) == "+4 Mult"
    assert _text(gt.joker("j_greedy_joker").lines) == "Played Diamond cards / give +3 Mult"
    cav = gt.joker("j_cavendish")
    assert _text(cav.lines) == " X3  Mult / 1 in 1000 chance &lt;gone&gt;"  # HTML-escaped
    assert cav.unresolved == 0
    castle = gt.joker("j_castle")
    assert castle.unresolved == 1  # suit depends on the live round
    assert gt.joker("j_misprint").lines == []
    assert gt.joker("j_blueprint") is None  # not in the fake file


def test_item_text(fake_game):
    """Consumables, vouchers and packs: generate_card_ui's values, planets at level 1."""
    gt = GameText.load(fake_game / "Balatro.exe")
    assert _text(gt.item("c_hermit").lines) == "Doubles money / (Max of $20)"  # not the unlock
    assert _text(gt.item("c_magician").lines) == "Enhances 2 selected / cards to Lucky Cards"
    assert _text(gt.item("p_arcana_normal_1").lines) == "Choose 1 of up to / 3 Tarot cards"
    mercury = gt.item("c_mercury")  # the Planet set's own loc_vars
    assert _text(mercury.lines) == "(lvl.1) Level up / Pair / +1 Mult and / +15 chips"
    assert mercury.unresolved == 0
    assert gt.item("c_fool") is None  # not in the fake file
    assert gt.item("v_telescope").lines == ["Celestial packs"]  # no #n# values


def test_parse_ui_loc_vars_skips_other_sections():
    from conftest import FAKE_COMMON_EVENTS

    from balatro_analysis.game_text import parse_ui_loc_vars

    ui = parse_ui_loc_vars(FAKE_COMMON_EVENTS)
    assert ui["The Hermit"].exprs == ["_c.config.extra"]  # Tarot section, not the unlock
    assert ui["The Empress"] == ui["The Magician"]  # "or" branches share loc_vars
    assert ui["Arcana Pack"].desc == "p_arcana_normal"
    assert len(ui["Planet"].exprs) == 4  # colours = {...} left out
    assert parse_ui_loc_vars("no such function") == {}


# --- stats and cards --------------------------------------------------------


def test_joker_stats(run):
    s = metrics.joker_stats([run])
    assert s.loc["j_joker", "bought"] == 2
    assert s.loc["j_joker", "episodes_bought"] == 2
    assert s.loc["j_joker", "held_at_end"] == 1
    assert s.loc["j_greedy_joker", "first_ante"] == 2
    assert s.loc["j_banner", "mean_ante_held"] == 1
    assert metrics.joker_stats([]).empty


def test_build_cards_without_game(run):
    cards = collection.build_cards(metrics.joker_stats([run]))
    assert len(cards) == 150
    assert [c.key for c in cards[:2]] == ["j_joker", "j_greedy_joker"]  # collection order
    joker = cards[0]
    assert (joker.rarity_name, joker.cost, joker.bought, joker.icon) == ("Common", 2, 2, None)


def test_build_cards_with_game(run, fake_game):
    archive = fake_game / "Balatro.exe"
    cards = collection.build_cards(
        metrics.joker_stats([run]), GameText.load(archive), Sprites(archive, scale=1)
    )
    joker = next(c for c in cards if c.key == "j_joker")
    assert joker.icon.startswith("data:image/png;base64,")
    assert joker.plain_text == "+4 Mult"


def test_select_filters_and_sorts(run):
    cards = collection.build_cards(metrics.joker_stats([run]))
    used = collection.select(cards, only_used=True, sort="Times bought")
    assert [c.key for c in used] == ["j_joker", "j_greedy_joker", "j_banner"]
    assert [c.key for c in collection.select(cards, query="greedy")] == ["j_greedy_joker"]
    legendary = collection.select(cards, rarities=[4])
    assert len(legendary) == 5 and {c.rarity_name for c in legendary} == {"Legendary"}
    assert collection.select(cards, query="j_caino")[0].name == "Canio"


def test_grid_html_naming_modes(run):
    cards = collection.select(collection.build_cards(metrics.joker_stats([run])), query="greedy")
    pretty = collection.grid_html(cards, Names(pretty=True))
    keys = collection.grid_html(cards, Names(pretty=False))
    assert '<div class="jk-name">Greedy Joker</div>' in pretty
    assert '<div class="jk-name">j_greedy_joker</div>' in keys
    assert "Bought <b>1&times;</b>" in pretty
    assert "Suit Mult" in pretty  # effect category stands in for the game text


def test_export_includes_joker_stats(run):
    import io
    import zipfile

    from balatro_analysis import export

    z = zipfile.ZipFile(io.BytesIO(export.export_zip([run], {})))
    stats = pd.read_csv(io.BytesIO(z.read("tables/joker_stats.csv")))
    assert {"joker", "joker_name", "bought", "held_at_end"} <= set(stats.columns)
