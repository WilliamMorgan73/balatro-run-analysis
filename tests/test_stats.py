import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from balatro_analysis import economy, metrics, stats
from balatro_analysis.loader import Run


def _run(name: str, antes: list[int], won: list[int] | None = None, seeds=None) -> Run:
    n = len(antes)
    e = pd.DataFrame(
        {
            "episode": range(n),
            "ante": antes,
            "won": won if won is not None else [0] * n,
            "return": [a - 1.0 for a in antes],
            "seed": seeds if seeds is not None else [f"S_{i}" for i in range(n)],
            "deck": "b_red",
            "stake": 1,
        }
    )
    return Run(name=name, path=Path(name), episodes=e)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.arange(1, 9, dtype=float)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert stats.bootstrap_ci(x) == (lo, hi)


def test_bootstrap_ci_too_few_values():
    assert all(np.isnan(stats.bootstrap_ci([3.0])))


def test_bootstrap_diff_ci_drops_nans():
    a, b = np.arange(1, 9, dtype=float), np.arange(3, 11, dtype=float)
    expected = stats.bootstrap_diff_ci(a, b)
    assert all(np.isfinite(expected))
    assert stats.bootstrap_diff_ci(np.append(a, np.nan), np.append(b, np.nan)) == expected
    assert all(np.isnan(stats.bootstrap_diff_ci([1.0, np.nan], b)))


def test_wilson_ci_no_wins_is_not_degenerate():
    lo, hi = stats.wilson_ci(0, 50)
    assert lo == pytest.approx(0, abs=1e-12)
    assert 0.05 < hi < 0.1  # ~0.071 for 0/50


def test_summary_ci():
    s = stats.summary_ci([_run("a", [1, 2, 3, 4])]).set_index("outcome")
    assert s.loc["ante reached", "mean"] == pytest.approx(2.5)
    assert s.loc["ante reached", "median"] == pytest.approx(2.5)
    assert s.loc["ante reached", "ci_low"] < 2.5 < s.loc["ante reached", "ci_high"]
    assert s.loc["win rate", "ci_low"] == pytest.approx(0, abs=1e-12)


def test_compare_identical_runs_shows_no_difference():
    a = _run("a", [1, 2, 3, 2, 1, 4])
    b = _run("b", [1, 2, 3, 2, 1, 4])
    c = stats.compare([a, b], "a").iloc[0]
    assert c["run"] == "b"
    assert c["mean_ante_diff"] == 0
    assert c["rank_biserial"] == pytest.approx(0)
    assert c["mann_whitney_p"] == pytest.approx(1)
    assert c["paired_episodes"] == 6
    assert c["wilcoxon_p"] == 1
    assert c["fisher_p"] == 1


def test_compare_clearly_better_run():
    base = _run("base", [1] * 20)
    better = _run("better", [3] * 20)
    c = stats.compare([base, better], "base").iloc[0]
    assert c["mean_ante_diff"] == 2
    assert c["rank_biserial"] == pytest.approx(1)
    assert c["mann_whitney_p"] < 0.001
    assert c["wilcoxon_p"] < 0.001
    assert c["diff_ci_low"] == c["diff_ci_high"] == 2


def test_paired_only_matches_same_seed_and_deck():
    a = _run("a", [1, 2, 3], seeds=["X", "Y", "Z"])
    b = _run("b", [2, 2, 2], seeds=["X", "Y", "W"])
    b.episodes.loc[1, "deck"] = "b_blue"  # same seed, different deck -> not a pair
    pairs = stats.paired_ante(b, a)
    assert pairs["seed"].tolist() == ["X"]
    assert stats.compare([a, b], "a").iloc[0]["paired_episodes"] == 1


def test_compare_unknown_baseline():
    assert stats.compare([_run("a", [1, 2])], "nope").empty


def test_compare_empty_baseline():
    """A baseline the filters left empty: no difference, and no empty-mean warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        c = stats.compare([_run("base", []), _run("a", [1, 2])], "base").iloc[0]
    assert np.isnan(c["mean_ante_diff"])


def test_bh_adjust():
    q = stats.bh_adjust(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert q.tolist()[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(q.iloc[3])


def test_spearman_constant_input():
    assert all(np.isnan(stats.spearman(pd.Series([1, 1, 1]), pd.Series([1, 2, 3]))))


def test_learning_curve_band():
    c = metrics.learning_curve([_run("a", [1, 3, 1, 3])], window=2)
    assert c["mean ante low"].iloc[0] == c["mean ante high"].iloc[0] == 1  # one episode
    assert c["mean ante low"].iloc[1] < c["mean ante"].iloc[1] < c["mean ante high"].iloc[1]


def test_outcome_correlations_have_q_values():
    econ = pd.DataFrame(
        {"run": "a", "ante": [1, 2, 3, 4, 5, 6], "final_dollars": [1, 2, 3, 4, 5, 6]}
    )
    corr = economy.outcome_correlations(econ).iloc[0]
    assert corr["spearman_rho"] == pytest.approx(1)
    assert corr["p_value"] < 0.05
    assert corr["q_value"] == pytest.approx(corr["p_value"])  # only one test in the run
    assert corr["significant"]


# --- IQM and stratified bootstrap ------------------------------------------


def test_iqm_hand_computed():
    assert stats.iqm(np.arange(1, 9)) == 4.5  # mean of 3, 4, 5, 6
    assert stats.iqm([np.nan, 1, 2, 3, 4]) == 2.5
    assert np.isnan(stats.iqm([]))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 50, 101])
def test_iqm_from_counts_matches_trim_mean(n):
    from scipy.stats import trim_mean

    rng = np.random.default_rng(n)
    for x in (rng.integers(1, 9, n).astype(float), rng.normal(size=n)):
        values, counts = np.unique(x, return_counts=True)
        got = stats._iqm_from_counts(values, counts[None, :], n)[0]
        assert got == pytest.approx(trim_mean(x, 0.25))


def _rliable_reference(matrix, n_boot, seed):
    """rliable's StratifiedBootstrap + aggregate_iqm, written as loops."""
    from scipy.stats import trim_mean

    rng = np.random.default_rng(seed)
    r, t = matrix.shape
    idx = rng.integers(0, r, size=(n_boot, r, t))
    iqms = []
    for b in range(n_boot):
        sample = np.empty_like(matrix)
        for task in range(t):
            for row in range(r):
                sample[row, task] = matrix[idx[b, row, task], task]
        iqms.append(trim_mean(sample, 0.25, axis=None))
    return np.quantile(iqms, 0.025), np.quantile(iqms, 0.975)


def test_stratified_bootstrap_matches_rliable_procedure():
    m = np.random.default_rng(1).integers(1, 9, size=(3, 7)).astype(float)
    got = stats.stratified_bootstrap_ci(m, n_boot=300, seed=4)
    assert got == pytest.approx(_rliable_reference(m, 300, 4))
    assert got == stats.stratified_bootstrap_ci(m, n_boot=300, seed=4)  # deterministic
    continuous = np.random.default_rng(2).normal(size=(3, 7))  # sorted, not counted
    got = stats.stratified_bootstrap_ci(continuous, n_boot=300, seed=4)
    assert got == pytest.approx(_rliable_reference(continuous, 300, 4))


def test_stratified_bootstrap_resamples_within_tasks():
    same = np.tile([1.0, 4.0, 6.0, 2.0], (3, 1))  # every run scored the same per task
    lo, hi = stats.stratified_bootstrap_ci(same)
    assert lo == hi == stats.iqm(same)
    # task 0 only ever 1-2, task 1 only ever 10-11: a sample can't mix them up
    m = np.array([[1.0, 10.0], [2.0, 11.0]])
    lo, hi = stats.stratified_bootstrap_ci(m)
    assert 5.5 <= lo <= hi <= 6.5


def _eval_run(name, antes, seeds, group=None, rounds=None):
    r = _run(name, antes, seeds=seeds)
    r.episodes["eval_round"] = rounds if rounds is not None else 0
    r.config = {"group": group} if group else {}
    return r


def test_score_matrix_keeps_shared_games_once():
    a = _eval_run("a", [1, 2, 3, 9], ["S1", "S2", "S3", "S1"])  # S1 replayed
    b = _eval_run("b", [4, 5, 6], ["S1", "S2", "S3"])
    m = stats.score_matrix([a, b])
    assert m.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert stats.score_matrix([a]) is None
    c = _eval_run("c", [1, 2, 3], ["X1", "X2", "S3"])  # one game in common
    assert stats.score_matrix([a, c]) is None
    a.episodes = a.episodes.drop(columns="seed")
    assert stats.score_matrix([a, b]) is None


def test_iqm_summary_methods():
    seeds = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    shared = [_eval_run(f"g_s{i}", rng.integers(1, 6, 20), seeds, "g") for i in range(3)]
    t = stats.iqm_summary(shared, by_group=True)
    ante = t[t["outcome"] == "ante reached"].iloc[0]
    assert ante["group"] == "g" and ante["method"] == stats.SHARED_GAMES
    assert ante["games"] == 20 and ante["runs"] == 3
    assert ante["iqm_low"] <= ante["iqm"] <= ante["iqm_high"]
    m = stats.score_matrix(shared)
    assert ante["iqm"] == pytest.approx(stats.iqm(m))
    # different games per run: each run's episodes are a stratum
    apart = [
        _run(f"h_s{i}", list(rng.integers(1, 6, 20)), seeds=[f"T{i}_{j}" for j in range(20)])
        for i in range(2)
    ]
    for r in apart:
        r.config = {"group": "h"}
    h = stats.iqm_summary(apart, by_group=True)
    assert set(h["method"]) == {stats.WITHIN_RUNS} and h["games"].iloc[0] == 40
    runs_table = stats.iqm_summary(apart)
    assert list(runs_table["run"].unique()) == ["h_s0", "h_s1"]
    assert "win rate" not in set(runs_table["outcome"])


def test_episodes_iqm_ci_stratified():
    lo, hi = stats._episodes_iqm_ci([np.full(10, 3.0), np.full(30, 5.0)])
    assert lo == hi == stats.iqm(np.r_[np.full(10, 3.0), np.full(30, 5.0)])  # strata fixed
    lo, hi = stats._episodes_iqm_ci([np.arange(100.0)])
    assert lo < stats.iqm(np.arange(100.0)) < hi


# --- Power ------------------------------------------------------------------


def test_power_textbook_sample_sizes():
    # Cohen's d = 0.5, alpha 0.05, 80% power: 64 per arm two-sample, 34 pairs paired
    assert stats._n_needed(0.5, lambda n: np.sqrt(2 / n), lambda n: 2 * n - 2) == 64
    assert stats._n_needed(0.5, lambda n: 1 / np.sqrt(n), lambda n: n - 1) == 34
    mdes = [stats._mde(np.sqrt(2 / n), 2 * n - 2) for n in (10, 50, 200)]
    assert mdes == sorted(mdes, reverse=True)
    assert stats._mde(np.sqrt(2 / 64), 126) == pytest.approx(0.5, abs=0.01)


def test_power_runs_paired_and_unpaired():
    rng = np.random.default_rng(2)
    seeds = [f"S{i}" for i in range(50)]
    base = _run("base", list(rng.integers(1, 6, 50)), seeds=seeds)
    other = _run("other", list(rng.integers(1, 6, 50)), seeds=seeds)
    fresh = _run("fresh", list(rng.integers(1, 6, 50)), seeds=[f"F{i}" for i in range(50)])
    p = stats.power_runs([base, other, fresh], "base", target=0.25).set_index("run")
    assert p.loc["other", "paired"] and not p.loc["fresh", "paired"]
    assert p.loc["fresh", "mde"] > 0.25  # 50 games can't see a quarter of an ante
    assert p.loc["fresh", "games_needed"] > 50
    assert stats.power_runs([base], "nope").empty


def test_variance_components_and_seeds_needed():
    rng = np.random.default_rng(3)
    # three seeds whose true means differ by a whole ante, little game noise
    members = []
    for i, mu in enumerate((2.0, 3.0, 4.0)):
        r = _run(f"g_s{i}", list(mu + rng.normal(0, 0.1, 400)))
        r.config = {"group": "g"}
        members.append(r)
    vc = stats.variance_components(members)
    assert vc["seeds"] == 3 and vc["games"] == 400
    assert vc["var_game"] == pytest.approx(0.01, rel=0.2)
    assert vc["var_seed"] == pytest.approx(1.0, rel=0.1)
    row = stats.power_groups(members, target=0.25).iloc[0]
    assert row["seeds_needed"] > 3 and row["mde"] > 0.25
    # with seed variation dominating, unlimited games barely help
    assert row["seeds_needed_any_games"] <= row["seeds_needed"] <= row["seeds_needed_any_games"] + 1
    flat = [_run(f"f_s{i}", [1, 5] * 20) for i in range(3)]  # no seed variation
    for r in flat:
        r.config = {"group": "f"}
    row = stats.power_groups(flat).iloc[0]
    assert row["sd_seed"] == 0 and row["seeds_needed_any_games"] is None
    single = [_run("solo", [1, 2, 3])]
    assert stats.power_groups(single).empty


# --- Performance profiles and probability of improvement ----------------------


def test_performance_profile_hand_computed():
    p = stats.performance_profile([_run("a", [1, 2, 2, 3])])
    assert p["ante"].tolist() == [1, 2, 3]
    assert p["fraction"].tolist() == [1.0, 0.75, 0.25]
    assert (p["fraction low"] <= p["fraction"]).all() and (
        p["fraction"] <= p["fraction high"]
    ).all()
    assert p.loc[0, "fraction low"] == p.loc[0, "fraction high"] == 1.0  # everyone reaches ante 1


def test_performance_profile_groups_use_shared_games():
    seeds = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(5)
    members = [_eval_run(f"g_s{i}", rng.integers(1, 9, 30), seeds, "g") for i in range(3)]
    p = stats.performance_profile(members, by_group=True)
    assert set(p["group"]) == {"g"} and set(p["method"]) == {stats.SHARED_GAMES}
    assert p["games"].iloc[0] == 30 and p["runs"].iloc[0] == 3
    cells = stats.score_matrix(members).ravel()
    assert p["fraction"].tolist() == pytest.approx([(cells >= k).mean() for k in p["ante"]])
    assert p["fraction"].is_monotonic_decreasing


def test_improvement_unpaired_matches_mann_whitney():
    a = _run("a", [1, 2, 3, 3, 4, 5], seeds=[f"A{i}" for i in range(6)])
    b = _run("b", [1, 1, 2, 3, 3], seeds=[f"B{i}" for i in range(5)])
    row = stats.improvement([b, a], "b").iloc[0]
    _, rank_biserial = stats.mann_whitney(
        a.episodes["ante"].to_numpy(), b.episodes["ante"].to_numpy()
    )
    assert row["method"] == stats.WITHIN_RUNS
    assert row["p_improvement"] == pytest.approx((rank_biserial + 1) / 2)
    assert row["p_low"] <= row["p_improvement"] <= row["p_high"]
    back = stats.improvement([a, b], "a").iloc[0]
    assert back["p_improvement"] == pytest.approx(1 - row["p_improvement"])


def test_improvement_paired_runs():
    seeds = ["S1", "S2", "S3", "S4"]
    a = _run("a", [2, 3, 3, 5], seeds=seeds)
    b = _run("b", [1, 3, 4, 5], seeds=seeds)
    row = stats.improvement([b, a], "b").iloc[0]
    assert row["method"] == "paired games"
    assert row["p_improvement"] == pytest.approx((1 + 0.5 + 0 + 0.5) / 4)
    same = stats.improvement([b, _run("b2", [1, 3, 4, 5], seeds=seeds)], "b").iloc[0]
    assert same["p_improvement"] == same["p_low"] == same["p_high"] == 0.5


def test_improvement_groups_match_rliable():
    seeds = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(7)
    a = [_eval_run(f"a_s{i}", rng.integers(1, 9, 6), seeds, "a") for i in range(3)]
    b = [_eval_run(f"b_s{i}", rng.integers(1, 9, 6), seeds, "b") for i in range(2)]
    row = stats.improvement([*b, *a], "b", by_group=True).iloc[0]
    assert row["group"] == "a" and row["method"] == stats.SHARED_GAMES and row["games"] == 6
    ma, mb = stats.score_matrix(a), stats.score_matrix(b)

    def p_of(xa, xb):  # rliable: per task, share of seed pairs where a beats b
        per_task = [
            np.mean([(x > y) + 0.5 * (x == y) for x in xa[:, t] for y in xb[:, t]])
            for t in range(xa.shape[1])
        ]
        return np.mean(per_task)

    assert row["p_improvement"] == pytest.approx(p_of(ma, mb))
    ra = np.random.default_rng(stats.SEED).integers(0, 3, size=(stats.N_BOOT, 3, 6))
    rb = np.random.default_rng(stats.SEED + 1).integers(0, 2, size=(stats.N_BOOT, 2, 6))
    t = np.arange(6)
    boot = [p_of(ma[ra[i], t], mb[rb[i], t]) for i in range(stats.N_BOOT)]
    assert (row["p_low"], row["p_high"]) == pytest.approx(
        (np.quantile(boot, 0.025), np.quantile(boot, 0.975))
    )
    assert stats.improvement(a, "nope").empty
