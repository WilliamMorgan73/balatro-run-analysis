"""balatro-shop-rl log_version 17: editions and stickers, the shop per decision,
endless mode and replays.  The fixture is a real 3-episode run (stakes 4 and 7,
endless) with its 2 test games replayed as ``retest``."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_app import TABS
from test_test_split import _app

from balatro_analysis import cards
from balatro_analysis.loader import find_runs, load_run

V17 = Path(__file__).parent / "fixtures" / "v17"
APP = str(Path(__file__).parent.parent / "app.py")


def test_v17_run_loads():
    assert [p.name for p in find_runs(V17)] == ["run_v17"]
    run = load_run(V17 / "run_v17", V17)
    assert run.config["log_version"] == 17 and run.config["endless"] is True
    assert run.replays.tags() == ["retest"]
    retest = run.split("replay/retest")
    assert retest.replay["stakes"] == [4, 7] and retest.replay["games"] == 2
    pd.testing.assert_frame_equal(
        retest.episodes.drop(columns="seconds"), run.test.episodes.drop(columns="seconds")
    )


def test_v17_steps_fields():
    steps = load_run(V17 / "run_v17").steps
    for col in ("jokers_extra", "joker_slots", "shop", "item_edition", "item_stickers"):
        assert col in steps
    held = [
        cards.held_jokers(k, x)
        for k, x in zip(steps["jokers_before"], steps["jokers_extra"], strict=True)
    ]
    perishable = [j for js in held for j in js if "perishable" in j.st]
    assert perishable and all(j.perish is not None for j in perishable)
    assert {"card", "voucher", "pack"} <= set(steps["shop"].iloc[0])


def test_v17_shop_items_carry_stickers():
    items = [i for row in load_run(V17 / "run_v17").shops["items"] for i in row]
    extras = [cards.item_extra(i) for i in items]
    assert ("foil", ("eternal",)) in extras
    assert any(st == ("eternal",) for _, st in extras)
    assert cards.item_extra({"key": "j_joker"}) == (None, ())


def test_parse_joker():
    assert cards.parse_joker("j_joker") == ("j_joker", None, (), None)
    j = cards.parse_joker({"k": "j_egg", "ed": "holo", "st": ["perishable"], "perish": 0})
    assert j.key == "j_egg" and j.ed == "holo" and j.expired
    assert not cards.parse_joker({"k": "j_egg", "st": ["perishable"], "perish": 2}).expired
    # Arrow-read structs hold every key, null where unset
    assert cards.parse_joker({"k": "j_egg", "ed": None, "st": None, "perish": np.nan}) == (
        "j_egg",
        None,
        (),
        None,
    )


def test_held_jokers_without_extras():
    assert cards.held_jokers(["j_joker"], None) == [cards.HeldJoker("j_joker")]
    assert cards.held_jokers(None, None) == []
    held = cards.held_jokers(["a", "b"], [None, {"st": ["eternal"]}])
    assert held == [cards.HeldJoker("a"), cards.HeldJoker("b", st=("eternal",))]


def test_older_runs_still_load(runs_dir):
    run = load_run(runs_dir / "run_small")
    assert "jokers_extra" not in run.steps and run.replays.tags() == []


@pytest.mark.parametrize("tab", TABS)
@pytest.mark.parametrize("phase", ["Test", "Replay: retest", "Training"])
def test_every_tab_renders_on_a_v17_run(tmp_path, phase, tab):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = _app(root, phase, tab)
    assert not at.exception


SUB_TABS = (
    [
        ("Actions", "actions_tab", t)
        for t in (
            "Overview",
            "Context",
            "Sequences",
            "Over training",
            "Availability",
            "Actions vs outcome",
        )
    ]
    + [
        ("Economy", "econ_tab", t)
        for t in (
            "Money flow",
            "Spending",
            "Interest & saving",
            "Rerolls & prices",
            "Economy vs outcome",
            "Cash-out income",
            "Shop offers",
            "Shop purchases",
        )
    ]
    + [("Episode explorer", "explorer_view", t) for t in ("Replay", "Shop", "Money & blinds")]
)


@pytest.mark.parametrize(("tab", "key", "sub"), SUB_TABS)
@pytest.mark.parametrize("phase", ["Test", "Replay: retest", "Training"])
def test_every_sub_tab_renders_on_a_v17_run(tmp_path, phase, tab, key, sub):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    at.session_state["phase"] = phase
    at.session_state["main_tab"] = tab
    at.session_state[key] = sub
    at.run()
    assert not at.exception
