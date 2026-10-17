"""Invariants a train_random_shop run (log_version 5) must satisfy, checked on a
real run (tests/fixtures/v5/run_v5: 12 smart-play episodes plus 2 eval rounds).

The dashboard relies on these; regenerate the fixture with the jackdaw script
when the log format changes, and these tests catch a harness that drifts:

    uv run python scripts/train_random_shop.py --episodes 12 --play smart \\
        --decks b_red,b_yellow --eval-every 6 --eval-episodes 3 --run-name run_v5
"""

from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from balatro_analysis.loader import Run, load_run

RUN_DIR = Path(__file__).parent / "fixtures" / "v5" / "run_v5"
EARN_PARTS = ["earn_blind", "earn_hands", "earn_discards", "earn_interest", "earn_jokers"]


@pytest.fixture(params=["train", "eval"])
def run(request) -> Run:
    run = load_run(RUN_DIR)
    return run if request.param == "train" else run.eval


def test_fixture_is_v5(run):
    assert run.config["log_version"] == 5
    assert len(run.steps) and len(run.rounds) and len(run.shops)


def test_cash_out_breakdown(run):
    paid = run.rounds.dropna(subset=["earn_total"])
    assert len(paid)
    assert (paid[EARN_PARTS].sum(axis=1) == paid["earn_total"]).all()
    assert ((paid["earn_tags"] >= 0) & (paid["earn_tags"] % 25 == 0)).all()  # Investment
    assert paid["beaten"].all()
    assert run.rounds.loc[~run.rounds["beaten"], "earn_total"].isna().all()


def test_one_loss_per_lost_episode_and_it_is_last(run):
    for ep, g in run.rounds.groupby("episode"):
        assert g["round"].diff().dropna().eq(1).all(), ep
        lost = g[~g["beaten"]]
        assert len(lost) <= 1
        if len(lost):
            assert lost.index[0] == g.index[-1]
    e = run.episodes.set_index("episode")
    last = run.rounds.groupby("episode").tail(1).set_index("episode")
    assert (last["round"] == e.loc[last.index, "final_round"]).all()


def test_money_is_continuous_within_a_visit(run):
    """The next decision starts with the money the last one ended with (after the
    pack it opened, if any): no money changes go unlogged."""
    for _, g in run.steps.sort_values(["episode", "step"]).groupby("episode"):
        rows = g.to_dict("records")
        for a, b in pairwise(rows):
            if a["round"] != b["round"]:
                assert a["action_name"] == "skip"  # a visit ends by leaving the shop
                continue
            after = a["dollars_after_pack"]
            after = a["dollars_after"] if pd.isna(after) else after
            assert after == b["dollars_before"], (a, b)


def test_episode_totals_match_steps(run):
    e = run.episodes.set_index("episode")
    s = run.steps
    g = s.groupby("episode")
    assert (g.size().reindex(e.index, fill_value=0) == e["shop_steps"]).all()
    spent = (s["dollars_before"] - s["dollars_after"]).clip(lower=0).groupby(s["episode"]).sum()
    assert (spent.reindex(e.index, fill_value=0) == e["dollars_spent"]).all()
    counts = g["action_name"].value_counts().unstack(fill_value=0)
    for name in run.config["action_names"]:
        col = counts[name] if name in counts else pd.Series(dtype=int)
        assert (col.reindex(e.index, fill_value=0) == e[name]).all(), name
    visits = g["round"].nunique().reindex(e.index, fill_value=0)
    assert (visits == e["shops_visited"]).all()
    entered = run.shops[run.shops["reason"] == "enter"].groupby("episode").size()
    assert (entered.reindex(e.index, fill_value=0) == e["shops_visited"]).all()
    rewards = g["reward"].sum().reindex(e.index, fill_value=0)
    assert (rewards == e["return"]).all()


def test_actions_are_legal_and_named(run):
    names = run.config["action_names"]
    s = run.steps
    assert all(a in legal for a, legal in zip(s["action"], s["legal"], strict=True))
    assert all(names.index("skip") in legal for legal in s["legal"])
    assert (s["action"].map(names.__getitem__) == s["action_name"]).all()
    for _, g in s.groupby("episode"):
        assert g["step"].tolist() == list(range(len(g)))


def test_buys_pay_the_listed_price(run):
    buys = run.steps[run.steps["action_name"].str.startswith("buy")]
    assert len(buys)
    assert buys["item_key"].notna().all()
    assert (buys["dollars_before"] - buys["dollars_after"] == buys["item_cost"]).all()


def test_eval_games_are_the_fixed_set():
    run = load_run(RUN_DIR)
    games = run.config["eval_set"]
    e = run.eval.episodes
    assert e.groupby("eval_round").size().eq(len(games)).all()
    for i, game in enumerate(games):
        g = e[e["episode"] % len(games) == i]
        assert (g["seed"] == game["seed"]).all()
        assert (g["deck"] == game["deck"]).all()
        assert (g["stake"].astype(str) == str(game["stake"])).all()
    assert e.groupby("eval_round")["train_episodes"].first().tolist() == [0, 6, 12]
