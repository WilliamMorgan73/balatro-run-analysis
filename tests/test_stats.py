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
