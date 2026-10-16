"""Groups: one config trained with several seeds (``--group``)."""

import numpy as np
import pytest

from balatro_analysis import export, stats
from balatro_analysis.loader import load_runs

RUN_SMALL_MEAN_ANTE = (2 + 1 + 1) / 3  # run_small's episodes reach antes 2, 1, 1


def test_group_of_defaults_to_run_name(run):
    assert stats.group_of(run) == run.name


def test_groups_in_order(group_runs_dir):
    g = stats.groups(load_runs(group_runs_dir))
    assert list(g) == ["a", "b"]
    assert [len(v) for v in g.values()] == [3, 2]


def test_t_ci():
    lo, hi = stats.t_ci([1.0, 2.0, 3.0])
    assert (lo + hi) / 2 == pytest.approx(2.0)
    assert hi - 2.0 == pytest.approx(4.302653 / np.sqrt(3), rel=1e-5)
    assert np.isnan(stats.t_ci([1.0])[0])


def test_group_summary_is_across_runs(group_runs_dir):
    s = stats.group_summary(load_runs(group_runs_dir))
    ante = s[s["outcome"] == "ante reached"].set_index("group")
    assert ante.loc["a", "runs"] == 3
    assert ante.loc["a", "episodes"] == 9
    assert ante.loc["a", "mean"] == pytest.approx(RUN_SMALL_MEAN_ANTE + 1)
    assert ante.loc["a", "std_across_runs"] == pytest.approx(1.0)
    lo, hi = stats.t_ci(RUN_SMALL_MEAN_ANTE + np.array([0, 1, 2]))
    assert ante.loc["a", "ci_low"] == pytest.approx(lo)
    assert ante.loc["a", "ci_high"] == pytest.approx(hi)


def test_compare_groups(group_runs_dir):
    c = stats.compare_groups(load_runs(group_runs_dir), "a").iloc[0]
    assert c["group"] == "b"
    assert (c["runs"], c["baseline_runs"]) == (2, 3)
    assert c["ante_diff"] == pytest.approx(3.5 - 1)
    assert c["ante_ci_low"] < c["ante_diff"] < c["ante_ci_high"]
    assert 0 < c["ante_welch_p"] < 0.05
    assert np.isnan(c["win_rate_welch_p"])  # no wins anywhere: no variance to test


def test_compare_groups_unknown_baseline(group_runs_dir):
    assert stats.compare_groups(load_runs(group_runs_dir), "nope").empty


def test_group_learning_curve(group_runs_dir):
    curve = stats.group_learning_curve(load_runs(group_runs_dir), window=1)
    assert set(curve["run"]) == {"a", "b"}
    a0 = curve[(curve["run"] == "a") & (curve["episode"] == 0)].iloc[0]
    assert a0["runs"] == 3
    assert a0["mean ante"] == pytest.approx(2 + 1)  # episode 0 reaches ante 2, shifts 0..2
    assert a0["mean ante low"] < a0["mean ante"] < a0["mean ante high"]


def test_group_eval_curve(eval_runs_dir):
    runs = [r.eval for r in load_runs(eval_runs_dir)]
    for r in runs:
        r.config = r.config | {"group": "g"}
    curve = stats.group_eval_curve(runs)
    assert list(curve["eval_round"]) == [0, 1]
    assert list(curve["train_episodes"]) == [0, 10]
    assert list(curve["runs"]) == [2, 2]
    assert list(curve["games"]) == [6, 6]
    per_run = stats.eval_curve(runs)
    first = per_run[per_run["eval_round"] == 0]["mean ante"].mean()
    assert curve["mean ante"].iloc[0] == pytest.approx(first)


def test_export_has_group_tables(group_runs_dir):
    tables = export.computed_tables(load_runs(group_runs_dir))
    assert len(tables["group_summary"]) == 6  # 2 groups x 3 outcomes
    assert list(tables["group_comparison"]["group"]) == ["b"]
