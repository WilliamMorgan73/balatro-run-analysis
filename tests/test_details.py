import dataclasses
import io
import json
import re

import pandas as pd
from conftest import BLIND_COLOUR
from PIL import Image

from balatro_analysis import collection, details, metrics
from balatro_analysis.game_text import UNKNOWN_HTML, GameText, parse_descriptions
from balatro_analysis.names import Names
from balatro_analysis.sprites import Sprites


def _text(lines: list[str]) -> str:
    return " / ".join(re.sub(r"<[^>]+>", "", line) for line in lines)


def _with_bosses(run):
    """run_small plus two Hook rounds (one lost) and a beaten Wall."""
    extra = pd.DataFrame(
        [
            {"episode": 0, "ante": 1, "round": 3, "blind_key": "bl_hook", "target": 600,
             "chips": 900, "beaten": True},
            {"episode": 1, "ante": 2, "round": 6, "blind_key": "bl_hook", "target": 1600,
             "chips": 400, "beaten": False},
            {"episode": 2, "ante": 3, "round": 9, "blind_key": "bl_wall", "target": 8000,
             "chips": 8000, "beaten": True},
        ]
    )  # fmt: skip
    return dataclasses.replace(run, rounds=pd.concat([run.rounds, extra], ignore_index=True))


# --- game text -------------------------------------------------------------


def test_deck_and_blind_text(fake_game):
    gt = GameText.load(fake_game / "Balatro.exe")
    assert _text(gt.deck("b_red").lines) == "+1 discard / every round"  # config.discards
    assert _text(gt.deck("b_black").lines) == "+1 Joker slot / -1 hand"  # -(-1)
    assert _text(gt.deck("b_nebula").lines) == "Start run with the / Telescope voucher"
    assert gt.deck("b_blue") is None  # not in the fake file
    assert _text(gt.blind("bl_hook").lines) == "Discards 2 random / cards per hand played"
    ox = gt.blind("bl_ox")
    assert ox.unresolved == 1 and UNKNOWN_HTML in ox.lines[0]  # the hand depends on the run


def test_parse_descriptions_prefix():
    loc = 'Joker={ j_a={ name="A", text={"x"} } } Back={ b_a={ name="B", text={"y"} } }'
    assert list(parse_descriptions(loc)) == ["j_a"]
    assert parse_descriptions(loc, "b_") == {"b_a": {"name": "B", "text": ["y"]}}


def test_blind_chip_sprite(fake_game):
    s = Sprites(fake_game / "Balatro.exe")
    chip = Image.open(io.BytesIO(s.png("bl_hook")))
    assert chip.size == (68, 68)
    assert chip.convert("RGBA").getpixel((34, 34)) == BLIND_COLOUR


# --- metrics ---------------------------------------------------------------


def test_win_rate_and_held_by_run(run):
    s = metrics.joker_stats([run])
    assert s.loc["j_joker", "win_rate_held"] == 0
    held = metrics.joker_held_by_run([run])
    row = held[held["joker"] == "j_joker"].iloc[0]
    assert (row["run"], row["held_at_end"], row["episodes"]) == ("run_small", 1, 3)


def test_deck_top_jokers(run):
    top = metrics.deck_top_jokers([run], "b_red")
    assert top.to_dict() == {"j_greedy_joker": 1, "j_joker": 1}
    assert metrics.deck_top_jokers([run], "b_plasma").empty


def test_boss_stats(run):
    assert metrics.boss_stats([run]).empty  # no bosses in run_small
    b = metrics.boss_stats([_with_bosses(run)])
    assert list(b.index) == ["bl_hook", "bl_wall"]  # most faced first
    hook = b.loc["bl_hook"]
    assert (hook["faced"], hook["beaten"], hook["losses"]) == (2, 1, 1)
    assert hook["beat_rate"] == 0.5
    assert hook["mean_ratio"] == (1.5 + 0.25) / 2
    assert hook["antes"] == [1, 2]


# --- HTML --------------------------------------------------------------------


def _dialog_ids(markup: str) -> tuple[list[str], list[str]]:
    return re.findall(r'data-dialog="([^"]+)"', markup), re.findall(
        r'<dialog class="dt-dialog" id="([^"]+)"', markup
    )


def test_joker_grid_has_a_dialog_per_card(run, fake_game):
    archive = fake_game / "Balatro.exe"
    cards = collection.build_cards(
        metrics.joker_stats([run]),
        GameText.load(archive),
        Sprites(archive, scale=1),
        metrics.joker_held_by_run([run]),
    )
    used = collection.select(cards, only_used=True)
    markup = collection.grid_html(used, Names())
    tiles, dialogs = _dialog_ids(markup)
    assert tiles == dialogs == [f"jk-{c.key}" for c in used]
    assert "showModal" in markup
    greedy = markup[markup.index('id="jk-j_greedy_joker"') :]
    assert "Win rate when held at end" in greedy and "Played Diamond cards" in _text([greedy])
    unused = collection.grid_html(collection.select(cards, query="cavendish"), Names())
    assert "Not bought or held" in unused


def test_mini_grid_allows_repeated_jokers(run):
    cards = collection.select(collection.build_cards(metrics.joker_stats([run])), query="greedy")
    markup = collection.mini_grid_html(cards * 2, Names(), "ex", ["a", "b"])
    tiles, dialogs = _dialog_ids(markup)
    assert len(set(tiles)) == 2 and tiles == dialogs


def test_deck_dialog(run, fake_game):
    gt = GameText.load(fake_game / "Balatro.exe")
    deck_table = metrics.by_deck([run])
    red = details.deck_info(
        "b_red", deck_table, gt, lambda k: None, metrics.deck_top_jokers([run], "b_red")
    )
    assert red.episodes == 2 and red.max_ante == 2
    markup = details.deck_grid_html([red], Names())
    assert _dialog_ids(markup) == (["deck-b_red"], ["deck-b_red"])
    assert "+1 discard" in _text([markup])
    assert "Most held at the end" in markup and "Greedy Joker" in markup
    plasma = details.deck_info("b_plasma", deck_table, None, lambda k: None)
    assert "Not played" in details.deck_grid_html([plasma], Names(pretty=False))


def test_blind_dialog(run, fake_game):
    gt = GameText.load(fake_game / "Balatro.exe")
    infos = details.blind_infos(metrics.boss_stats([_with_bosses(run)]), gt, lambda k: None)
    markup = details.blind_grid_html(infos, Names())
    assert _dialog_ids(markup)[1] == ["boss-bl_hook", "boss-bl_wall"]
    hook = markup[markup.index('id="boss-bl_hook"') : markup.index('id="boss-bl_wall"')]
    assert "Discards 2 random" in hook and "From ante 1" in hook and "$5 reward" in hook
    assert "1 (50%)" in hook  # beaten
    assert "background:#a84024" in markup  # boss colour stands in for missing art


def test_blinds_json_has_chip_data():
    from balatro_analysis import catalog

    hook = catalog.blinds()["bl_hook"]
    assert hook["pos"] == {"x": 0, "y": 7} and hook["colour"].startswith("#")
    assert catalog.blinds()["bl_final_bell"]["showdown"] is True
    assert json.dumps(catalog.centers()["b_red"]["config"]) == '{"discards": 1}'


# --- tables whose item icons open dialogs -------------------------------------


def _item_table(pretty=True, icon=lambda k: "data:image/png;base64,AAAA"):
    df = pd.DataFrame(
        {
            "step": [0, 1],
            "action_name": ["buy_card_0", "skip"],
            "item_key": ["j_joker", None],
            "jokers_before": [[], ["j_joker", "j_greedy_joker"]],
            "note": ["<b>x</b>", 1.0],
            "beaten": [True, False],
        }
    )
    calls = []

    def dialog(key, did):
        calls.append(key)
        return f'<dialog id="{did}">{key}</dialog>'

    out = details.item_table_html(
        df, ["item_key", "jokers_before"], Names(pretty), "t", dialog, icon
    )
    return out, calls


def test_item_table_tiles_and_dialogs():
    out, calls = _item_table()
    assert out.count('data-dialog="t-j_joker"') == 2  # one tile per occurrence
    assert out.count('<dialog id="t-j_joker">') == 1  # one dialog per item
    assert calls == ["j_joker", "j_greedy_joker"]
    assert "&lt;b&gt;x&lt;/b&gt;" in out and "<b>x</b>" not in out  # cells escaped
    assert "Buy card 1" in out and '<td class="dt-num">1</td>' in out  # 1.0 shown as 1
    assert '<td class="dt-num">0</td>' in out  # numpy ints are numbers too
    assert "<td>✓</td>" in out and "<td>✗</td>" in out  # numpy bools


def test_item_table_without_art_and_in_key_mode():
    out, _ = _item_table(pretty=False, icon=lambda k: None)
    assert '<span class="dt-chipname">j_greedy_joker</span>' in out  # still clickable
    assert "buy_card_0" in out and "<img" not in out


def test_item_dialog(fake_game):
    gt = GameText.load(fake_game / "Balatro.exe")
    out = details.item_dialog(
        "c_hermit", "d1", Names(), gt.item("c_hermit"), None, [("Times bought", "3")]
    )
    assert "The Hermit" in out and "Doubles money" in out
    assert "Tarot" in out and "Times bought" in out
    assert "needs the game files" in details.item_dialog("c_hermit", "d2", Names(), None, None)
