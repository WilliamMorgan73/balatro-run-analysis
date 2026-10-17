"""Economy metrics on the fixture run, checked by hand.

Fixture shop steps (money before -> after):
  ep0 round1: buy j_joker 10->5, skip            (entry 10, exit 5, spent 5)
  ep0 round3: buy j_greedy_joker 12->6, skip     (entry 12, exit 6, spent 6)
  ep1 round1: buy j_joker 9->4, sell joker 4->6, buy j_banner 6->2, skip
                                                 (entry 9, exit 2, spent 9, sold 2)
"""

import pytest

from balatro_analysis import economy
from balatro_analysis.loader import load_run


def test_category():
    assert economy.category("reroll", None) == "Rerolls"
    assert economy.category("buy_card_0", "j_joker") == "Jokers"
    assert economy.category("buy_card_1", "c_jupiter") == "Planet"
    assert economy.category("buy_card_1", "c_death") == "Tarot"
    assert economy.category("buy_pack_0", "p_buffoon_normal_1") == "Packs"
    assert economy.category("buy_voucher", "v_overstock_norm") == "Vouchers"
    assert economy.category("sell_joker_2", "j_joker") == "Selling jokers"
    assert economy.category("skip", None) is None


def test_shop_visits(run):
    v = economy.shop_visits([run]).set_index(["episode", "round"])
    assert v.loc[(0, 1), ["entry", "exit", "spent", "sold"]].tolist() == [10, 5, 5, 0]
    assert v.loc[(0, 3), ["ante", "entry", "exit", "spent"]].tolist() == [2, 12, 6, 6]
    assert v.loc[(1, 1), ["entry", "exit", "spent", "sold", "buys", "decisions"]].tolist() == [
        9, 2, 9, 2, 2, 4,
    ]  # fmt: skip
    assert v.loc[(0, 1), "est_interest"] == 1  # $5 left -> $1
    assert v.loc[(1, 1), "est_interest"] == 0
    assert not v["at_cap"].any()


def test_spend_and_income(run):
    s = economy.spend_by_category([run]).set_index("category")
    assert s.loc["Jokers", "spent"] == 20
    assert s.loc["Jokers", "share"] == pytest.approx(1)
    inc = economy.income_by_source([run]).set_index("source")
    assert inc.loc["Selling jokers", "per_episode"] == pytest.approx(2 / 3)
    assert inc.loc["Blind reward", "per_episode"] == pytest.approx(7 / 3)
    assert inc.loc["Interest", "kind"] == "cash-out"


def test_money_by_round(run):
    m = economy.money_by_round([run])
    r1 = m[(m["round"] == 1) & (m["when"] == "entry")].iloc[0]
    assert r1["median"] == 9.5 and r1["visits"] == 2


def test_income_breakdown_and_interest(run):
    b = economy.income_breakdown([run]).set_index("component")["value"]
    assert b["Blind reward"] == 3.5
    assert b["Interest"] == 0.5
    eff = economy.interest_efficiency([run]).iloc[0]
    assert eff["max_interest"] == 5 and eff["efficiency"] == pytest.approx(0.1)


def test_offer_conversion(run):
    o = economy.offers(run)
    assert len(o) == 8
    first = o[o["step"].eq(0) & o["episode"].eq(0)].set_index("key")
    assert first.loc["j_joker", "bought"] and not first.loc["c_moon", "bought"]
    assert first.loc["v_overstock_norm", "affordable"]  # $10 with $10
    c = economy.offer_conversion([run]).set_index("category")
    assert c.loc["Jokers", ["offered", "affordable", "bought"]].tolist() == [5, 5, 4]
    s = economy.skipped_jokers([run]).set_index("key")
    assert s.loc["j_banner", "skipped"] == 1


def test_outcome(run):
    e = economy.episode_economy([run]).set_index("episode")
    assert e.loc[0, "total_spent"] == 11
    assert e.loc[1, "joker_share"] == 1
    assert e.loc[2, ["ante", "final_dollars"]].tolist() == [1, 4]
    assert e.loc[0, "spent_per_visit"] == 5.5


def test_kpis(run):
    k = economy.kpis([run]).iloc[0]
    assert k["spent_per_episode"] == pytest.approx(20 / 3)
    assert k["mean_exit"] == pytest.approx(13 / 3)


def test_old_logs_have_no_new_views(episodes_only_dir):
    old = load_run(episodes_only_dir / "old_run")
    for f in (economy.income_breakdown, economy.offer_conversion, economy.shop_visits):
        assert f([old]).empty
    assert economy.interest_cap(old) == 25


def test_v1_boss_ante_corrected(tmp_path):
    """Log v1 filed beaten bosses under the next ante; the loader moves them back."""
    import json

    d = tmp_path / "v1"
    d.mkdir()
    (d / "episodes.csv").write_text(
        "episode,ante,won,return,shop_steps,dollars_spent,seconds\n0,2,0,1,0,0,0\n"
    )
    common = {"episode": 0, "ante": 2, "target": 600, "chips": 0}
    rows = [
        {**common, "round": 3, "blind_key": "bl_hook", "beaten": True},
        {**common, "round": 4, "blind_key": "bl_small", "beaten": False},
    ]
    (d / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_run(d).rounds["ante"].tolist() == [1, 2]
    (d / "config.json").write_text('{"log_version": 2}')
    assert load_run(d).rounds["ante"].tolist() == [2, 2]


def test_tag_income(run):
    """log_version 5 logs money paid by tags at cash-out; older logs have no column."""
    assert "Tags" not in set(economy.income_by_source([run])["source"])
    run.rounds["earn_tags"] = [25 if i == 1 else 0 for i in range(len(run.rounds))]
    run.rounds.loc[run.rounds["earn_total"].isna(), "earn_tags"] = None
    inc = economy.income_by_source([run]).set_index("source")
    assert inc.loc["Tags", "per_episode"] == pytest.approx(25 / 3)
    assert "Tags" in set(economy.income_breakdown([run])["component"])


def test_pack_income(run):
    """log_version 5 logs money once a bought pack closes; gains count as income."""
    assert "Pack contents" not in set(economy.income_by_source([run])["source"])
    s = run.steps
    s["dollars_after_pack"] = None
    first = s.index[0]
    s.loc[first, "dollars_after_pack"] = s.loc[first, "dollars_after"] + 6
    inc = economy.income_by_source([run]).set_index("source")
    assert inc.loc["Pack contents", "per_episode"] == pytest.approx(6 / 3)
    v = economy.shop_visits([run])
    assert v["other_income"].sum() >= 6


def test_pack_loss(run):
    """Money lost while a pack is open (Wraith sets it to $0) is kept, not clipped."""
    s = run.steps
    s["dollars_after_pack"] = None
    first = s.index[0]
    s.loc[first, "dollars_after_pack"] = s.loc[first, "dollars_after"] - 3
    s.loc[s.index[1], ["dollars_before", "dollars_after"]] -= 3
    inc = economy.income_by_source([run]).set_index("source")
    assert inc.loc["Lost in packs", "per_episode"] == pytest.approx(-3 / 3)
    assert "Pack contents" not in inc.index
    v = economy.shop_visits([run]).set_index(["episode", "round"])
    assert v.loc[(0, 1), "pack_loss"] == 3
    assert v["pack_loss"].sum() == 3


def test_interest_cap_per_visit(copied_runs):
    """Seed Money / Money Tree raise the cap mid-run: each visit uses the cap at the
    next cash-out (the next blind's row), falling back to its own round's row."""
    import json

    path = copied_runs / "run_small" / "rounds.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        if (r["episode"], r["round"]) == (0, 2):
            r["interest_cap"] = 50  # Seed Money bought in the round-1 shop
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    v = economy.shop_visits([load_run(copied_runs / "run_small")]).set_index(["episode", "round"])
    assert v.loc[(0, 1), "interest_cap"] == 50
    assert v.loc[(0, 3), "interest_cap"] == 25  # next row (round 4) still 25
    assert v.loc[(1, 1), "interest_cap"] == 25


def test_rent_is_not_cash_out(copied_runs):
    import json

    path = copied_runs / "run_small" / "rounds.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["earn_rent"] = 3
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    inc = economy.income_by_source([load_run(copied_runs / "run_small")]).set_index("source")
    assert inc.loc["Rent", "kind"] == "end of round"
    assert inc.loc["Rent", "per_episode"] == pytest.approx(-1)
    assert inc.loc["Blind reward", "kind"] == "cash-out"
