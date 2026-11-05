"""The eval hand log (log_version 7): cards, hand metrics, the replay views and
playing-card art."""

import io
import json
import shutil

import pytest
from conftest import FACE_COLOUR, HANDS_DIR, RUNS_DIR
from PIL import Image

from balatro_analysis import cards, hands, metrics, replay
from balatro_analysis.loader import load_run
from balatro_analysis.names import Names
from balatro_analysis.sprites import Sprites, game_archive


# Through the module, not imported names: the app tests reload the package, which
# replaces these classes (a card from a reloaded module isn't == an old PlayingCard)
def PlayingCard(*args, **kwargs):
    return cards.PlayingCard(*args, **kwargs)


def parse_card(v):
    return cards.parse_card(v)


def parse_cards(v):
    return cards.parse_cards(v)


@pytest.fixture
def eval_run(tmp_path):
    """run_small with an eval/ folder holding the hand log fixture."""
    d = tmp_path / "run"
    shutil.copytree(RUNS_DIR / "run_small", d)
    shutil.copytree(RUNS_DIR / "run_small", d / "eval", ignore=shutil.ignore_patterns("config*"))
    for name in ("hands.jsonl", "decks.jsonl"):
        shutil.copy(HANDS_DIR / name, d / "eval" / name)
    return load_run(d).eval


def test_parse_card_both_forms():
    assert parse_card("H_A") == PlayingCard("H", "A")
    glass = parse_card({"c": "S_T", "en": "m_glass", "ed": "foil", "seal": "Red", "debuff": True})
    assert glass == PlayingCard("S", "T", "m_glass", "foil", "Red", True)
    assert glass.label == "10♠"
    assert glass.name == "10 of Spades (Glass, Foil, Red Seal, debuffed)"
    assert parse_card("D_K").name == "King of Diamonds"
    assert parse_cards(None) == []


def test_loader_reads_hand_log(eval_run):
    assert len(eval_run.hands) == 8 and len(eval_run.decks) == 4
    assert hands.has_hands(eval_run)
    assert load_run(RUNS_DIR / "run_small").hands is None  # training never has one


def test_hand_type_stats(eval_run):
    s = hands.hand_type_stats([eval_run]).set_index("hand_type")
    assert s.index.tolist() == ["Flush", "Pair"]  # game order, best first
    assert s.at["Pair", "plays"] == 4 and s.at["Flush", "plays"] == 2
    assert s["share"].sum() == pytest.approx(1)
    assert s.at["Flush", "mean_score_vs_target"] == pytest.approx(340 / 300)
    assert hands.most_played_hand(eval_run) == "Pair"


def test_blind_play_and_deck(eval_run):
    b = hands.blind_play([eval_run]).set_index(["episode", "round"])
    assert b.at[(0, 1), "hands_played"] == 2 and b.at[(0, 1), "discards_used"] == 1
    assert b.at[(0, 2), "blind type"] == "Big"
    d = hands.deck_composition([eval_run])
    assert d["size"].tolist() == [52] and d["enhanced"].tolist() == [1]
    assert d["edition"].tolist() == [1] and d["sealed"].tolist() == [1]
    decks = hands.episode_decks(eval_run, 0)
    assert decks.loc[2, "hand_levels"] == {"Pair": 2}


def test_episode_hands_in_order(eval_run):
    h = hands.episode_hands(eval_run, 3)
    assert list(zip(h["round"], h["hand_no"], strict=True)) == [(1, 0), (1, 1), (1, 2), (2, 0)]
    assert h.loc[0, "cards"][2] == PlayingCard("H", "2", "m_glass", seal="Red")


def test_blind_html_shows_plays_and_discards(eval_run):
    acts = hands.episode_hands(eval_run, 0)
    page = replay.blind_html(
        acts[acts["round"] == 1], Names(), lambda c: None, lambda k: None, True
    )
    assert page.count('class="rp-act"') == 3
    assert page.count("picked discard") == 3 and page.count("picked play") == 7
    assert ">Flush<" in page and ">340<" in page and ">300<" in page  # score and target
    assert "Small Blind" in page and "Beaten" in page
    assert ">A♠<" in page  # text cards without the game


def test_deck_html_groups_by_suit():
    deck = parse_cards(["H_A", "H_2", "S_K", {"c": "C_5", "en": "m_stone"}])
    page = replay.deck_html(deck, lambda c: None)
    assert "♠ 1" in page and "♥ 2" in page and "Stone 1" in page and "♣" not in page


def test_shop_html_marks_bought(run):
    from balatro_analysis import economy, metrics

    one = metrics.filter_episodes(run, 0, 0)
    page = replay.shop_html(economy.offers(one), one.shops, one.steps, Names(), lambda k: None)
    bought = economy.offers(one)["bought"].sum()
    assert page.count("rp-slot card bought") + page.count("rp-slot pack bought") == bought
    assert "Shop after round" in page


def test_run_sheet_html():
    page = replay.run_sheet_html(
        "qlearn_s0",
        {"shop_agent": "QLearningShopAgent", "play_agent": "StrongPlayAgent"},
        3.4,
        (3.0, 3.8),
        0.25,
        6,
        40,
        ["b_red"],
        Names(),
        lambda k: None,
        "Two Pair",
        best=True,
    )
    assert "QLearning shop, Strong play" in page and "3.40" in page and "25%" in page
    assert "Two Pair" in page and "Best mean ante" in page and "Red Deck" in page


def test_card_art_from_the_game(fake_game):
    s = Sprites(game_archive(fake_game))
    png = s.card_png(PlayingCard("H", "A"))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    assert img.size == (142, 190)
    assert img.getpixel((71, 95)) == FACE_COLOUR
    stone = Image.open(io.BytesIO(s.card_png(PlayingCard("H", "A", "m_stone")))).convert("RGBA")
    assert stone.getpixel((71, 95)) != FACE_COLOUR  # a Stone card has no rank or suit
    assert s.font_uri().startswith("data:font/ttf;base64,")


def test_hand_fixture_is_valid_jsonl():
    for name in ("hands.jsonl", "decks.jsonl"):
        rows = [json.loads(line) for line in (HANDS_DIR / name).open()]
        assert rows and all("episode" in r for r in rows)


def test_level_and_jokers_logged(eval_run):
    h = hands.episode_hands(eval_run, 0)
    plays = h[h["kind"] == "play"]
    assert plays["level"].tolist() == [1, 1, 2] and plays["level_logged"].all()
    assert h.loc[0, "jokers_held"] == ["j_joker"]
    assert h.loc[2, "jokers_held"] == ["j_joker", "j_greedy_joker"]
    assert hands.blind_jokers(eval_run, 0, 1) == (["j_joker"], True, [None])

    page = replay.blind_html(h[h["round"] == 2], Names(), lambda c: None, lambda k: None)
    assert 'title="Hand level when played">lvl.2<' in page
    page = replay.blind_html(h[h["round"] == 1], Names(), lambda c: None, lambda k: None)
    assert page.count('class="rp-jchange"') == 1 and "gained <b>Greedy Joker</b>" in page


def test_level_and_jokers_estimated_without_them(eval_run):
    """Runs from before jokers/hand_level were logged: level at the start of the
    blind from decks.jsonl, jokers from the shop log."""
    eval_run.hands = eval_run.hands.drop(columns=["jokers", "hand_level"])
    eval_run._memo.clear()
    h = hands.episode_hands(eval_run, 0)
    plays = h[h["kind"] == "play"]
    assert plays["level"].tolist() == [1, 1, 2]  # Pair is level 2 at round 2's start
    assert not plays["level_logged"].any()
    assert h["jokers_held"].isna().all()
    page = replay.blind_html(h[h["round"] == 2], Names(), lambda c: None, lambda k: None)
    assert 'class="rp-lvl est"' in page and 'class="rp-jchange"' not in page
    jokers, logged, states = hands.blind_jokers(eval_run, 0, 2)
    assert not logged and jokers == ["j_joker"]  # held leaving the shop after round 1
    assert states == [None]  # before log_version 15


def test_parse_joker():
    assert cards.parse_joker("j_joker") == ("j_joker", None)
    assert cards.parse_joker({"k": "j_joker", "ed": "foil"}) == ("j_joker", "foil")
    assert cards.joker_keys(None) is None


# --- log_version 15: scaling jokers' current values ----------------------------


def test_joker_state_and_labels():
    assert cards.joker_state({"k": "j_trousers", "s": {"mult": 6}}) == {"mult": 6}
    assert cards.joker_state({"k": "j_joker", "ed": "e_foil"}) is None
    assert cards.joker_state("j_joker") is None
    # a steps.jsonl joker_state item read by Arrow: every path any joker had
    assert cards.joker_state({"mult": None, "x_mult": 1.91, "extra.chips": None}) == {
        "x_mult": 1.91
    }
    assert cards.joker_state({"mult": None}) is None
    assert cards.joker_states(None) is None
    assert cards.state_label("j_trousers", {"mult": 6}) == "+6 Mult"
    assert cards.state_label("j_ramen", {"x_mult": 1.9134}) == "X1.91 Mult"
    assert cards.state_label("j_ramen", {"x_mult": 2.0}) == "X2 Mult"
    assert cards.state_label("j_square", {"extra.chips": 12}) == "+12 Chips"
    assert cards.state_label("j_selzer", {"extra": 7}) == "7 hands left"
    assert cards.state_label("j_new", {"odd.path": 3}) == "odd.path 3"  # unknown: as logged
    assert cards.state_label("j_joker", None) == ""
    held, states = ["j_joker", "j_trousers"], [None, {"mult": 6}]
    assert cards.state_summary(held, states, str) == "j_trousers +6 Mult"
    assert cards.state_summary(held, None, str) == ""


def _v15(eval_run):
    """``eval_run`` as a log_version 15 run: Spare Trousers scaling across a blind."""
    h = eval_run.hands.copy()
    jokers = []
    for i in range(len(h)):
        held = list(h.at[i, "jokers"]) if isinstance(h.at[i, "jokers"], list) else []
        jokers.append([{"k": "j_trousers", "s": {"mult": 2 * i}}, *held])
    h["jokers"] = jokers
    eval_run.hands = h
    eval_run._memo.clear()
    return eval_run


def test_scaling_jokers_in_the_replay(eval_run):
    run = _v15(eval_run)
    h = hands.episode_hands(run, 0)
    assert h.loc[1, "joker_states"][0] == {"mult": 2}
    keys, logged, states = hands.blind_jokers(run, 0, 1)
    assert logged and keys[0] == "j_trousers" and states[0] == {"mult": 0}
    page = replay.blind_html(h[h["round"] == 1], Names(pretty=True), lambda c: None, lambda k: None)
    assert "<b>Spare Trousers</b> +0 Mult → +2 Mult" in page


def test_joker_state_from_the_shop_log(eval_run):
    """Without a hand log, a blind's jokers and values come from the shop before it."""
    s = eval_run.steps.copy()
    s["joker_state"] = [[None] * len(j) for j in s["jokers_before"]]
    last = s[s["episode"] == 0].sort_values("step").groupby("round").tail(1).index
    s.at[last[0], "joker_state"] = [{"mult": 8}] + [None] * (
        len(s.at[last[0], "jokers_before"]) - 1
    )
    eval_run.steps, eval_run.hands = s, None
    eval_run._memo.clear()
    _, logged, states = hands.blind_jokers(eval_run, 0, 2)
    assert not logged and states[0] == {"mult": 8}
    rounds = metrics.episode_rounds(eval_run, 0)
    assert "joker_state" in rounds
