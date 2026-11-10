"""Eternal and Perishable jokers (log_version 17), as balatro-shop-rl's
scripts/stickers.py ``behaviour``; checked against it on generated logs."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import cards, hands, stickers
from balatro_analysis.loader import Run, load_run

V17 = Path(__file__).parent / "fixtures" / "v17"
APP = str(Path(__file__).parent.parent / "app.py")
ETERNAL = {"st": ["eternal"]}


def _expired():
    return {"st": ["perishable"], "perish": 0}


def _run(steps: list[tuple], stake: int = 7, name: str = "r") -> Run:
    """``steps``: (held keys, jokers_extra, action) per decision of episode 0."""
    rows = [
        {
            "episode": 0,
            "step": i,
            "jokers_before": held,
            "jokers_extra": extra,
            "joker_slots": 2,
            "action_name": action,
        }
        for i, (held, extra, action) in enumerate(steps)
    ]
    e = pd.DataFrame({"episode": [0], "ante": [1], "won": [0], "stake": [stake]})
    return Run(name, Path(name), e, {"log_version": 17}, steps=pd.DataFrame(rows))


def test_behaviour():
    run = _run(
        [
            (["j_a", "j_b"], [ETERNAL, _expired()], "skip"),
            (["j_a", "j_b"], [ETERNAL, _expired()], "sell_joker_1"),  # sold after 2 decisions
            (["j_a", "j_c"], [ETERNAL, _expired()], "sell_joker_0"),  # an Eternal "sold"
        ]
    )
    b = stickers.behaviour(run).set_index("stake").loc[7]
    assert b["decisions"] == 3
    assert b["eternal_held_per_decision"] == 1.0
    assert b["full_share_with_eternal"] == 1.0
    assert b["p_sell_expired"] == pytest.approx(1 / 3)
    assert b["decisions_before_sold"] == 2.0
    assert b["expired_never_sold"] == 1  # j_c
    assert b["sell_eternal"] == 1


def test_no_stickers_held():
    b = stickers.behaviour(_run([(["j_a"], [None], "skip")])).set_index("stake").loc[7]
    assert b["eternal_held_per_decision"] == 0 and np.isnan(b["p_sell_expired"])
    assert b["full_share_with_eternal"] == 0  # not full


def test_behaviour_table_over_seeds():
    a = _run([(["j_a"], [ETERNAL], "skip")], name="a")
    b = _run([(["j_a", "j_b"], [None, _expired()], "skip")], name="b")
    t = stickers.behaviour_table({"g": [a, b]}).set_index("metric")
    assert t.at["Eternals held (per shop decision)", "g"] == 0.5  # mean of 1 and 0
    assert t.at["expired Perishables never sold (total)", "g"] == 1  # summed
    # a has no expired Perishable (NaN, dropped), b never sold one
    assert t.at["expired Perishable held: P(sold this decision)", "g"] == 0


def test_blind_held_reads_stickers():
    run = load_run(V17 / "run_v17")
    steps = run.steps
    row = steps[steps["jokers_extra"].map(len) > 0].iloc[0]
    later = steps[(steps["episode"] == row["episode"]) & (steps["round"] > row["round"])]
    if later.empty:
        pytest.skip("no later blind in the fixture's episode")
    last = steps[(steps["episode"] == row["episode"]) & (steps["round"] < later["round"].min() + 1)]
    keys = list(last.sort_values("step").iloc[-1]["jokers_before"])
    held = hands.blind_held(run, row["episode"], int(later["round"].min()) + 1, keys)
    assert [j.key for j in held] == keys
    assert any(j.st for j in held)


def test_extra_badges():
    html = cards.extra_badges("negative", ("eternal", "perishable"), 0)
    assert "Negative" in html and "Eternal" in html and "Perishable · expired" in html
    assert "Perishable 3" in cards.extra_badges(None, ("perishable",), 3)
    assert cards.extra_badges(None) == ""


def _jokers_tab(root: Path, phase: str | None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    if phase:
        at.session_state["phase"] = phase
    at.session_state["main_tab"] = "Jokers"
    at.run()
    return at


@pytest.mark.parametrize("phase", ["Test", "Training"])
def test_app_sticker_panel(tmp_path, phase):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = _jokers_tab(root, phase)
    assert not at.exception
    assert any("stickers.py" in c.value for c in at.caption)


def test_app_sticker_panel_on_old_run(runs_dir):
    at = _jokers_tab(runs_dir, None)
    assert not at.exception
    assert any("log_version 17" in i.value for i in at.info)


def test_app_explorer_on_v17(tmp_path):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    at.session_state["phase"] = "Training"
    at.session_state["main_tab"] = "Episode explorer"
    at.run()
    assert not at.exception
