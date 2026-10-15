import pytest

from balatro_analysis import metrics
from balatro_analysis.loader import load_runs


def test_summary(run):
    s = metrics.summary([run]).iloc[0]
    assert s["episodes"] == 3
    assert s["mean ante"] == pytest.approx(4 / 3)
    assert s["max ante"] == 2
    assert s["win rate"] == 0


def test_filter_episodes(run):
    cut = metrics.filter_episodes(run, 1, 2)
    assert cut.episodes["episode"].tolist() == [1, 2]
    assert set(cut.steps["episode"]) == {1}
    assert set(cut.rounds["episode"]) == {1, 2}


def test_learning_curve_rolling(run):
    c = metrics.learning_curve([run], window=2)
    assert c["mean ante"].tolist() == pytest.approx([2, 1.5, 1])


def test_ante_distribution(run):
    d = metrics.ante_distribution([run]).set_index("ante")["fraction"]
    assert d[1] == pytest.approx(2 / 3)
    assert d[2] == pytest.approx(1 / 3)


def test_action_frequency(run):
    f = metrics.action_frequency([run]).set_index("action")
    assert f.loc["skip", "count"] == 3
    assert f.loc["buy_card_0", "count"] == 3
    assert f["share"].sum() == pytest.approx(1)


def test_action_by_ante(run):
    t = metrics.action_by_ante(run)
    assert t[1].sum() == pytest.approx(1)
    assert t.loc["skip", 2] == pytest.approx(0.5)


def test_joker_purchases_and_first_ante(run):
    assert len(metrics.joker_purchases(run)) == 4
    first = metrics.first_bought_ante([run])
    row = first[(first["episode"] == 0) & (first["joker"] == "j_greedy_joker")]
    assert row["ante"].item() == 2


def test_most_bought(run):
    top = metrics.most_bought([run]).set_index("joker")["bought"]
    assert top["j_joker"] == 2


def test_cooccurrence(run):
    m = metrics.joker_cooccurrence(run)
    assert m.loc["j_joker", "j_greedy_joker"] == 1
    assert m.loc["j_greedy_joker", "j_joker"] == 1
    assert m.loc["j_banner", "j_banner"] == 1
    assert m.loc["j_banner", "j_joker"] == 0


def test_sell_then_buy(run):
    pairs = metrics.sell_then_buy(run)
    assert pairs.to_dict("records") == [{"sold": "j_joker", "bought": "j_banner", "count": 1}]


def test_score_ratio_and_deaths(run):
    r = metrics.score_ratio([run])
    assert r.loc[0, "ratio"] == pytest.approx(1.5)
    assert set(r["blind type"]) == {"Small", "Big"}
    d = metrics.deaths([run])
    assert sorted(d["episode"]) == [0, 1, 2]


def test_episode_views(run):
    t = metrics.episode_timeline(run, 0)
    assert t["action_name"].tolist() == ["buy_card_0", "skip", "buy_card_1", "skip"]
    assert t.iloc[3]["jokers_before"] == ["j_joker", "j_greedy_joker"]
    assert len(metrics.episode_rounds(run, 0)) == 3


def test_metrics_without_step_logs(episodes_only_dir):
    runs = load_runs(episodes_only_dir)
    run = runs[0]
    assert metrics.action_by_ante(run).empty
    assert metrics.money_by_ante(runs).empty
    assert metrics.score_ratio(runs).empty
    assert metrics.sell_then_buy(run).empty
    assert not metrics.action_frequency(runs).empty  # falls back to episodes.csv columns


def test_filter_by_deck_and_stake(run):
    blue = metrics.filter_episodes(run, 0, 99, decks=["b_blue"])
    assert blue.episodes["episode"].tolist() == [1]
    assert set(blue.steps["episode"]) == {1}
    red_white = metrics.filter_episodes(run, 0, 99, decks=["b_red"], stakes=["1"])
    assert red_white.episodes["episode"].tolist() == [0]
    assert metrics.filter_episodes(run, 0, 99, decks=[]).episodes.empty


def test_by_deck(run):
    d = metrics.by_deck([run]).set_index(["deck", "stake"])
    assert d.loc[("b_red", "1"), "mean_ante"] == 2
    assert d["episodes"].sum() == 3


def test_summary_lists_decks(run):
    s = metrics.summary([run]).iloc[0]
    assert s["decks"] == "b_blue;b_red"
    assert s["stakes"] == "1;2"


def test_metrics_are_key_only(run):
    assert list(metrics.most_bought([run]).columns) == ["run", "joker", "bought"]
    assert "deck name" not in metrics.by_deck([run])
