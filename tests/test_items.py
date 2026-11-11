"""Per-item shop rates (Shop items tab) and counting a voucher or pack once per offer."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import items, purchases
from balatro_analysis.loader import Run

V17 = Path(__file__).parent / "fixtures" / "v17"
APP = str(Path(__file__).parent.parent / "app.py")
VOUCHER = {"kind": "voucher", "key": "v_overstock_norm", "cost": 10}
PACK = {"kind": "pack", "key": "p_buffoon_normal_1", "cost": 4}


def _mask(*actions: int) -> int:
    return sum(1 << a for a in actions)


def _step(step, legal, action="skip", key=None, ante=1, shop=None):
    return {
        "episode": 0,
        "step": step,
        "ante": ante,
        "jokers_before": [],
        "joker_slots": 5,
        "shop": shop or {"card": [], "voucher": ["v_overstock_norm"], "pack": []},
        "legal_mask": _mask(*legal),
        "action_name": action,
        "item_key": key,
        "item_edition": None,
    }


def _listing(step, reason, ante=1, *its):
    return {"episode": 0, "step": step, "ante": ante, "reason": reason, "items": list(its)}


def _run(shops, steps, won=1) -> Run:
    e = pd.DataFrame({"episode": [0], "ante": [2], "won": [won], "deck": ["b_red"], "stake": [1]})
    return Run(
        "r",
        Path("r"),
        e,
        {"log_version": 17},
        shops=pd.DataFrame(shops),
        steps=pd.DataFrame(steps),
    )


def _voucher_run() -> Run:
    """One voucher, listed on entering two shops of ante 1 and on a reroll; bought
    at the third listing.  The next ante lists another copy, never bought."""
    shops = [
        _listing(0, "enter", 1, VOUCHER),
        _listing(1, "enter", 1, VOUCHER),
        _listing(2, "reroll", 1, VOUCHER),
        _listing(4, "enter", 2, VOUCHER),
    ]
    steps = [
        _step(0, legal=[4]),
        _step(1, legal=[4], action="reroll"),
        _step(2, legal=[4], action="buy_voucher", key="v_overstock_norm"),
        _step(3, legal=[]),
        _step(4, legal=[4], ante=2),
    ]
    return _run(shops, steps)


def _voucher(run, per_offer):
    c = purchases.run_counts(run, per_offer=per_offer).set_index("category")
    return tuple(c.loc["Voucher", ["offered", "legal", "bought"]])


def test_voucher_once_per_ante():
    run = _voucher_run()
    assert _voucher(run, per_offer=False) == (4, 4, 1)  # per listing, as the script
    assert _voucher(run, per_offer=True) == (2, 2, 1)  # ante 1's (bought) and ante 2's


def test_pack_once_per_visit():
    """A pack carried over a reroll is one offer; the next shop's is another."""
    shop = {"card": [], "voucher": [], "pack": ["p_buffoon_normal_1"]}
    shops = [
        _listing(0, "enter", 1, PACK),
        _listing(1, "reroll", 1, PACK),
        _listing(3, "enter", 1, PACK),
    ]
    steps = [
        _step(0, legal=[], action="reroll", shop=shop),  # too dear at first
        _step(1, legal=[2], action="buy_pack_0", key="p_buffoon_normal_1", shop=shop),
        _step(2, legal=[], shop={"card": [], "voucher": [], "pack": []}),
        _step(3, legal=[2], shop=shop),
    ]
    run = _run(shops, steps)
    o = purchases.offers(run, per_offer=True)
    assert len(o) == 2
    assert list(o["legal"]) == [True, True] and list(o["bought"]) == [True, False]
    assert len(purchases.offers(run, per_offer=False)) == 3


def test_cards_are_never_merged():
    egg = {"kind": "card", "key": "j_egg", "cost": 4}
    shop = {"card": ["j_egg"], "voucher": [], "pack": []}
    shops = [_listing(0, "enter", 1, egg), _listing(1, "reroll", 1, egg)]
    steps = [_step(0, legal=[0], action="reroll", shop=shop), _step(1, legal=[0], shop=shop)]
    assert len(purchases.offers(_run(shops, steps), per_offer=True)) == 2


def test_item_rates_sold_used_and_wins():
    shop = {"card": ["c_fool", "j_egg"], "voucher": [], "pack": []}
    fool = {"kind": "card", "key": "c_fool", "cost": 3}
    egg = {"kind": "card", "key": "j_egg", "cost": 4}
    shops = [_listing(0, "enter", 1, fool, egg)]
    steps = [
        _step(0, legal=[0, 1], action="buy_card_0", key="c_fool", shop=shop),
        _step(1, legal=[13], action="use_cons_0", key="c_fool", shop=shop),
        _step(2, legal=[1], action="skip", shop=shop),
    ]
    rates = items.item_rates({"g": [_run(shops, steps, won=1)]}).set_index("key")
    f = rates.loc["c_fool"]
    assert (f["set"], f["legal"], f["bought"], f["used"]) == ("Tarot", 1, 1, 1)
    assert f["rate"] == 1 and f["use_rate"] == 1 and f["win_rate_bought"] == 1
    assert f["first_ante"] == 1 and f["mean_cost"] == 3
    e = rates.loc["j_egg"]
    assert (e["legal"], e["bought"], e["skipped"], e["rate"]) == (1, 0, 1, 0)
    assert pd.isna(e["use_rate"])
    assert list(items.section(rates.reset_index(), "Tarots")["key"]) == ["c_fool"]


def test_item_rates_mean_over_runs_and_ante_split():
    def run(bought: bool) -> Run:
        return _run(
            [_listing(0, "enter", 1, VOUCHER)],
            [
                _step(
                    0,
                    legal=[4],
                    **({"action": "buy_voucher", "key": VOUCHER["key"]} if bought else {}),
                )
            ],
        )

    rates = items.item_rates({"g": [run(True), run(False)]}, by="ante")
    row = rates.set_index("key").loc[VOUCHER["key"]]
    assert row["part"] == "1-2"
    assert row["rate"] == 0.5 and row["sd"] == pytest.approx(0.7071, abs=1e-4)
    shown = items.wide_table(rates, "1-2", min_legal=3)
    assert shown.empty


def test_v17_fixture_items():
    from balatro_analysis.loader import load_run

    run = load_run(V17 / "run_v17")
    rates = items.item_rates({"g": [run.test]}, by="stake")
    assert set(rates["part"].astype(str)) == {"4", "7"}
    # per offer never counts more than per listing
    per_listing = purchases.offers(run.test, per_offer=False)
    assert len(purchases.offers(run.test, per_offer=True)) <= len(per_listing)


@pytest.mark.parametrize("phase", ["Test", "Training"])
def test_app_shop_items(tmp_path, phase):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    at.session_state["phase"] = phase
    at.session_state["main_tab"] = "Shop items"
    at.session_state["items_min_legal"] = 0
    for tab in ("Vouchers", "Jokers", "Packs"):
        at.session_state["items_tab"] = tab
        at.run()
        assert not at.exception, tab
        assert any("Buy rate = bought" in c.value for c in at.caption)


def test_app_shop_items_on_old_run(runs_dir):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(runs_dir)
    at.session_state["main_tab"] = "Shop items"
    at.run()
    assert not at.exception
