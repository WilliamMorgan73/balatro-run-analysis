"""Shop purchase rates, as balatro-shop-rl's scripts/special_buys.py."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import purchases
from balatro_analysis.loader import Run

V17 = Path(__file__).parent / "fixtures" / "v17"
APP = str(Path(__file__).parent.parent / "app.py")


def _mask(*actions: int) -> int:
    return sum(1 << a for a in actions)


def _step(step, legal, action="skip", key=None, ed=None, held=("j_joker",) * 5, shop=None):
    return {
        "episode": 0,
        "step": step,
        "jokers_before": list(held),
        "joker_slots": 5,
        "shop": shop if shop is not None else {"card": ["j_egg"], "voucher": [], "pack": []},
        "legal_mask": _mask(*legal),
        "action_name": action,
        "item_key": key,
        "item_edition": ed,
    }


def _run(shops: list[dict], steps: list[dict], config=None) -> Run:
    e = pd.DataFrame({"episode": [0], "ante": [1], "won": [0], "deck": ["b_red"], "stake": [1]})
    return Run(
        "r",
        Path("r"),
        e,
        config if config is not None else {"log_version": 17},
        shops=pd.DataFrame(shops),
        steps=pd.DataFrame(steps),
    )


def _counts(run: Run) -> dict[str, tuple[int, int, int]]:
    c = purchases.run_counts(run)
    return {r.category: (r.offered, r.legal, r.bought) for r in c.itertuples()}


def test_legal_only_after_a_sell():
    """A Negative Egg too dear at the listing (slots full): the agent sells a joker,
    then buys it.  Legal at a later decision of the listing's window counts."""
    shops = [
        {
            "episode": 0,
            "step": 0,
            "items": [{"kind": "card", "key": "j_egg", "cost": 9, "ed": "negative"}],
        }
    ]
    steps = [
        _step(0, legal=[5], action="sell_joker_0"),  # can't buy the Egg yet
        _step(1, legal=[0], action="buy_card_0", key="j_egg", ed="negative", held=["j_joker"] * 4),
    ]
    got = _counts(_run(shops, steps))
    assert got["Joker: negative"] == (1, 1, 1)
    assert got["Joker: negative, slots full"] == (1, 1, 1)  # full at the listing
    assert got["Joker: Common"] == (1, 1, 1)  # the Egg's rarity
    assert got["Joker: no sticker"] == (1, 1, 1)


def test_window_ends_at_the_next_listing():
    shops = [
        {"episode": 0, "step": 0, "items": [{"kind": "card", "key": "j_egg", "cost": 4}]},
        {"episode": 0, "step": 1, "items": [{"kind": "card", "key": "j_egg", "cost": 4}]},
    ]
    steps = [
        _step(0, legal=[], action="reroll"),
        _step(1, legal=[0], action="buy_card_0", key="j_egg"),
    ]
    assert _counts(_run(shops, steps))["Joker"] == (2, 1, 1)  # the reroll's listing only


def test_edition_must_match_to_count_as_bought():
    shops = [
        {
            "episode": 0,
            "step": 0,
            "items": [
                {"kind": "card", "key": "j_egg", "cost": 4},
                {"kind": "card", "key": "j_egg", "cost": 7, "ed": "foil", "st": ["eternal"]},
            ],
        }
    ]
    shop = {"card": ["j_egg", "j_egg"], "voucher": [], "pack": []}
    steps = [_step(0, legal=[0, 1], action="buy_card_1", key="j_egg", ed="foil", shop=shop)]
    got = _counts(_run(shops, steps))
    assert got["Joker: foil"] == (1, 1, 1)
    assert got["Joker: no edition"] == (1, 1, 0)
    assert got["Joker: eternal"] == (1, 1, 1)
    assert got["Joker"] == (2, 2, 1)


def test_matched_by_key_not_slot():
    """As the script: two unedited copies of a key share bought (and legal)."""
    shops = [
        {
            "episode": 0,
            "step": 0,
            "items": [
                {"kind": "card", "key": "j_egg", "cost": 4},
                {"kind": "card", "key": "j_egg", "cost": 4},
            ],
        }
    ]
    shop = {"card": ["j_egg", "j_egg"], "voucher": [], "pack": []}
    steps = [_step(0, legal=[1], action="buy_card_1", key="j_egg", shop=shop)]
    assert _counts(_run(shops, steps))["Joker"] == (2, 2, 2)


def test_hidden_items_and_packs_and_vouchers():
    shops = [
        {
            "episode": 0,
            "step": 0,
            "items": [
                {"kind": "card", "key": "j_egg", "cost": 4, "hidden": True},
                {"kind": "voucher", "key": "v_overstock_norm", "cost": 10},
                {"kind": "pack", "key": "p_buffoon_jumbo_1", "cost": 6},
                {"kind": "card", "key": "c_fool", "cost": 3},
            ],
        }
    ]
    shop = {
        "card": ["j_egg", "c_fool"],
        "voucher": ["v_overstock_norm"],
        "pack": ["p_buffoon_jumbo_1"],
    }
    steps = [_step(0, legal=[1, 2, 4], action="buy_pack_0", key="p_buffoon_jumbo_1", shop=shop)]
    got = _counts(_run(shops, steps))
    assert "Joker" not in got
    assert got["Voucher"] == (1, 1, 0)
    assert got["Pack: buffoon"] == got["Pack: jumbo"] == got["Pack"] == (1, 1, 1)
    assert got["Tarot"] == (1, 1, 0)


def test_before_v17_legal_at_the_listing_only():
    shops = [{"episode": 0, "step": 0, "items": [{"kind": "card", "key": "j_egg", "cost": 9}]}]
    steps = [_step(0, legal=[5], action="sell_joker_0"), _step(1, legal=[0], action="skip")]
    for s in steps:
        del s["shop"], s["joker_slots"], s["item_edition"]
    run = _run(shops, steps, {"log_version": 16})
    assert not purchases.has_editions(run)
    got = _counts(run)
    assert got["Joker"] == (1, 0, 0)  # legal only after the sell: not seen
    assert "Joker: no edition, slots full" not in got  # slots unknown


def test_rates_and_table():
    def run(bought: bool) -> Run:
        shops = [{"episode": 0, "step": 0, "items": [{"kind": "card", "key": "j_egg", "cost": 4}]}]
        action = ("buy_card_0", "j_egg") if bought else ("skip", None)
        return _run(shops, [_step(0, legal=[0], action=action[0], key=action[1])])

    rates = purchases.buy_rates({"g": [run(True), run(False)], "h": [run(True)]})
    joker = rates.set_index(["group", "category"]).loc[("g", "Joker")]
    assert joker["rate"] == 0.5 and joker["sd"] == pytest.approx(0.7071, abs=1e-4)
    table = purchases.rate_table(rates, "all", min_legal=0).set_index("item")
    assert table.at["Joker", "g"] == "0.50 ± 0.71 (2)"
    assert table.at["Joker", "h"] == "1.00 ± 0.00 (1)"
    assert purchases.rate_table(rates, "all", min_legal=3).empty


def test_v17_fixture_counts():
    from balatro_analysis.loader import load_run

    run = load_run(V17 / "run_v17")
    got = purchases.run_counts(run.test, "stake")
    assert set(got["part"].astype(str)) == {"4", "7"}
    assert purchases.has_editions(run.test)


def _purchases_tab(root: Path, phase: str | None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    if phase:
        at.session_state["phase"] = phase
    at.session_state["main_tab"] = "Economy"
    at.session_state["econ_tab"] = "Shop purchases"
    at.session_state["buys_min_legal"] = 0  # the fixture is a few games
    at.run()
    return at


@pytest.mark.parametrize("phase", ["Test", "Replay: retest", "Training"])
def test_app_shop_purchases(tmp_path, phase):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = _purchases_tab(root, phase)
    assert not at.exception
    assert any("special_buys.py" in c.value for c in at.caption)
    assert any("Joker" in set(d.value.iloc[:, 0]) for d in at.dataframe)


def test_app_purchases_prefers_retest(tmp_path):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = _purchases_tab(root, "Test")
    assert any("**retest** replay" in c.value for c in at.caption)


def test_app_purchases_on_old_run(runs_dir):
    at = _purchases_tab(runs_dir, None)
    assert not at.exception
    assert any("log_version 17" in i.value for i in at.info)
