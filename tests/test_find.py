import pandas as pd
import pytest

from balatro_analysis import find


def test_episode_table(run):
    t = find.episode_table(run)
    assert t["episode"].tolist() == [0, 1, 2]
    assert t["final_jokers"].tolist() == [["j_joker", "j_greedy_joker"], ["j_banner"], []]
    # lost blinds: 600/1200, 225/450, 100/300
    assert t["score_vs_target"].round(3).tolist() == [0.5, 0.5, 0.333]
    assert t["lost_to"].tolist() == ["bl_big", "bl_big", "bl_small"]


def test_filter_episode_table(run):
    t = find.episode_table(run)
    eps = lambda **kw: find.filter_episode_table(t, **kw)["episode"].tolist()  # noqa: E731
    assert eps() == [0, 1, 2]
    assert eps(decks=["b_red"]) == [0, 2]
    assert eps(stakes=["2"]) == [2]
    assert eps(antes=(2, 8)) == [0]
    assert eps(outcome="Won") == []
    assert eps(outcome="Lost") == [0, 1, 2]
    assert eps(jokers=["j_joker"]) == [0]
    assert eps(jokers=["j_joker", "j_banner"]) == []


def _table(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "episode": list(range(n)),
        "ante": [1] * n,
        "won": [0] * n,
        "return": [0.0] * n,
        "final_dollars": [0] * n,
        "dollars_spent": [0] * n,
        "shop_steps": [0] * n,
        "deck": ["b_red"] * n,
        "stake": ["1"] * n,
        "seed": [f"S_{i}" for i in range(n)],
        "final_jokers": [[] for _ in range(n)],
    }
    return pd.DataFrame(base | cols)


def _by_label(table):
    return {s.label: s for s in find.suggested_episodes(table)}


def test_suggestions_pick_the_right_episodes():
    t = _table(
        ante=[2, 3, 3, 1],
        **{"return": [0.0, 1.0, 3.0, 0.0]},
        final_dollars=[30, 5, 1, 0],
        dollars_spent=[1, 50, 2, 0],
        shop_steps=[9, 3, 2, 0],
        final_jokers=[["j_joker"], ["j_joker", "j_cavendish"], ["j_joker"], []],
        score_vs_target=[0.4, 0.9, None, 0.2],
        lost_to=["bl_big", "bl_hook", None, "bl_small"],
    )
    s = _by_label(t)
    assert s["Best"].episode == 2  # ante 3, higher return than episode 1
    assert s["Closest loss"].episode == 1
    assert "bl_hook" in s["Closest loss"].reason and "90%" in s["Closest loss"].reason
    assert s["Richest finish"].episode == 0 and "$30" in s["Richest finish"].reason
    assert s["Biggest spender"].episode == 1
    assert s["Most shop decisions"].episode == 0
    assert s["Rarest build"].episode == 1  # Cavendish appears once
    assert s["Biggest outlier"].episode in (1, 2)  # +0.75 antes above the mean (2.25)


def test_suggestions_skip_what_has_no_answer():
    s = _by_label(_table(ante=[1, 1]))
    assert set(s) == {"Best"}  # no losses logged, no money, no jokers, no spread
    assert find.suggested_episodes(_table(ante=[]).iloc[:0]) == []


def test_most_improved_eval_game():
    t = _table(
        ante=[1, 1, 3, 1],
        seed=["EVAL_0", "EVAL_1", "EVAL_0", "EVAL_1"],
        eval_round=[0, 0, 1, 1],
    )
    s = _by_label(t)["Most improved eval game"]
    assert s.episode == 2
    assert "EVAL_0" in s.reason and "1" in s.reason and "3" in s.reason


@pytest.fixture
def index():
    return pd.DataFrame(
        {
            "run": ["b_s0", "a_s0", "a_s1", "solo", "training"],
            "group": ["b", "a", "a", "", ""],
            "shop_agent": ["Q", "R", "R", "R", "?"],
            "play_agent": ["S", "S", "S", "R", "?"],
            "started": ["2026-10-03", "2026-10-01", "2026-10-02", "2026-09-30", ""],
            "episodes": [10, 20, 30, 5, 0],
            "mean_ante": [1.5, 1.2, 1.4, 1.0, float("nan")],
            "win_rate": [0.0, 0.1, 0.0, 0.0, float("nan")],
        }
    )


def test_find_runs_sort_and_filter(index):
    assert find.find_runs_in(index) == ["a_s0", "a_s1", "b_s0", "solo", "training"]
    assert find.find_runs_in(index, sort="Mean ante") == [
        "b_s0",
        "a_s1",
        "a_s0",
        "solo",
        "training",  # nothing to sort by: last
    ]
    assert find.find_runs_in(index, sort="Newest")[0] == "b_s0"
    assert find.find_runs_in(index, search="S0") == ["a_s0", "b_s0"]
    assert find.find_runs_in(index, shop_agents=["Q"]) == ["b_s0"]
    assert find.find_runs_in(index, groups=["a"], sort="Episodes") == ["a_s1", "a_s0"]


def test_best_per_group(index):
    assert find.best_per_group(index) == ["a_s1", "b_s0"]
    ungrouped = index.assign(group="")
    assert find.best_per_group(ungrouped) == ["b_s0"]
    assert find.best_per_group(index.assign(mean_ante=float("nan"))) == []


def test_run_index(run):
    idx = find.run_index({"run_small": run, "training": None})
    assert idx["run"].tolist() == ["run_small", "training"]
    assert idx.loc[0, "episodes"] == 3 and idx.loc[0, "mean_ante"] == pytest.approx(4 / 3)
    assert idx.loc[0, "shop_agent"] == "RandomShopAgent"
    assert idx.loc[1, "episodes"] == 0 and pd.isna(idx.loc[1, "mean_ante"])
