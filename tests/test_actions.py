"""Shop-behaviour metrics on the fixture run.

Fixture visits: ep0 r1 (buy_card_0, skip), ep0 r3 (buy_card_1, skip),
ep1 r1 (buy_card_0, sell_joker_0, buy_card_0, skip).  Legal sets per step, in
order: [0,1,2,16] [5,15,16] [0,1,5,15,16] [5,6,16] [0,16] [0,5,16] [0,16] [5,16].
"""

import pytest

from balatro_analysis import actions
from balatro_analysis.loader import load_run


def test_family():
    assert actions.family("buy_card_1") == "Buy card"
    assert actions.family("buy_pack_0") == "Buy pack"
    assert actions.family("buy_voucher") == "Buy voucher"
    assert actions.family("sell_joker_5") == "Sell joker"
    assert actions.family("sell_cons_0") == "Sell consumable"
    assert actions.family("use_cons_1") == "Use consumable"
    assert actions.family("reroll") == "Reroll"
    assert actions.family("skip") == "Leave shop"


def test_steps_with_context(run):
    s = actions.steps_with_context(run)
    assert s["position"].tolist() == [1, 2, 1, 2, 1, 2, 3, 4]
    assert s["visit_len"].tolist() == [2, 2, 2, 2, 4, 4, 4, 4]
    # $10, 5, 12, 6, 9, 4, 6, 2 before each decision
    assert s["money_bracket"].astype(str).tolist() == [
        "$10-14", "$5-9", "$10-14", "$5-9", "$5-9", "$0-4", "$5-9", "$0-4",
    ]  # fmt: skip
    assert s["jokers_held"].astype(str).tolist() == ["0", "1", "1", "2", "0", "1", "0", "1"]
    assert set(s["next_blind"].astype(str)) == {"Big", "Small"}


def test_family_share_and_visits(run):
    fam = actions.family_share([run]).set_index("family")
    assert fam.loc["Buy card", "share"] == pytest.approx(0.5)
    assert fam.loc["Leave shop", "count"] == 3
    assert actions.decisions_per_visit([run])["decisions"].tolist() == [2, 2, 4]
    v = actions.visit_summary([run]).iloc[0]
    assert v["visits"] == 3
    assert v["decisions_per_visit"] == pytest.approx(8 / 3)
    assert v["leave_at_once"] == 0
    assert v["with_buy"] == 1
    assert v["with_sell"] == pytest.approx(1 / 3)
    assert v["top_first_action"] == "buy_card_0"
    assert v["top_first_share"] == pytest.approx(2 / 3)


def test_share_by(run):
    money = actions.share_by(run, "Money")
    assert list(money.columns) == ["$0-4", "$5-9", "$10-14"]
    assert money.sum().tolist() == pytest.approx([1, 1, 1])
    assert money.loc["sell_joker_0", "$0-4"] == pytest.approx(0.5)
    pos = actions.share_by(run, "Position in visit")
    assert pos.loc["skip", "2"] == pytest.approx(2 / 3)
    assert list(actions.share_by(run, "Ante").index) == [
        "buy_card_0", "buy_card_1", "sell_joker_0", "skip",
    ]  # fmt: skip


def test_transitions(run):
    t = actions.transitions(run).set_index(["from", "to"])
    assert t.loc[("START", "buy_card_0"), "count"] == 2
    assert t.loc[("START", "buy_card_0"), "rate"] == pytest.approx(2 / 3)
    assert t.loc[("buy_card_0", "skip"), "rate"] == pytest.approx(2 / 3)
    assert "skip" not in t.index.get_level_values("from")  # leaving ends the visit
    assert "END" not in t.index.get_level_values("to")
    m = actions.transition_matrix(run)
    assert m.index[0] == "START"
    assert m.sum(axis=1).tolist() == pytest.approx([1] * len(m))


def test_top_sequences(run):
    seqs = actions.top_sequences(run)
    assert seqs["count"].tolist() == [1, 1, 1]
    assert seqs["share"].sum() == pytest.approx(1)
    assert seqs.iloc[-1]["sequence"] == "buy_card_0 → sell_joker_0 → buy_card_0 → skip"


def test_family_over_training(run):
    over = actions.family_over_training([run], bins=2)
    assert over.groupby("episode")["share"].sum().tolist() == pytest.approx([1, 1])


def test_availability(run):
    a = actions.availability([run]).set_index("action")
    buy = a.loc["buy_card_0"]
    assert buy["legal_rate"] == pytest.approx(5 / 8)
    assert buy["chosen_when_legal"] == pytest.approx(3 / 5)
    assert buy["uniform_expected"] == pytest.approx((1 / 4 + 1 / 5 + 1 / 2 + 1 / 3 + 1 / 2) / 5)
    assert buy["preference"] == pytest.approx(0.6 / buy["uniform_expected"])
    skip = a.loc["skip"]
    assert skip["legal_rate"] == 1
    assert skip["chosen_when_legal"] == pytest.approx(3 / 8)
    assert a.loc["reroll", "preference"] == 0  # legal twice, never chosen
    assert a.loc["buy_voucher", "legal_count"] == 0
    assert a["preference"].isna()["buy_voucher"]
    counts = actions.legal_counts([run]).set_index("n_legal")["count"]
    assert counts.to_dict() == {2: 3, 3: 3, 4: 1, 5: 1}


def test_old_logs_have_no_availability(copied_runs):
    steps = copied_runs / "run_small" / "steps.jsonl"
    import json

    rows = [json.loads(line) for line in steps.read_text().splitlines()]
    steps.write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "legal"}) + "\n" for r in rows)
    )
    old = load_run(copied_runs / "run_small", copied_runs)
    assert actions.availability([old]).empty
    assert actions.legal_counts([old]).empty
    assert not actions.transitions(old).empty  # the rest still works


def test_no_steps(episodes_only_dir):
    old = load_run(episodes_only_dir / "old_run")
    for f in (actions.decisions_per_visit, actions.visit_summary, actions.availability):
        assert f([old]).empty
    assert actions.share_by(old, "Ante").empty
    assert actions.top_sequences(old).empty
    assert not actions.family_share([old]).empty  # from episodes.csv counts
