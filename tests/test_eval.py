"""Eval episodes (greedy, fixed seeds) written to <run>/eval/ by --eval-every."""

import numpy as np

from balatro_analysis import export, metrics, stats
from balatro_analysis.loader import find_runs, load_run, load_runs


def test_eval_dir_is_not_a_run(eval_runs_dir):
    assert [p.name for p in find_runs(eval_runs_dir)] == ["run_a", "run_b"]


def test_eval_dir_given_as_root_still_loads(eval_runs_dir):
    run = load_run(eval_runs_dir / "run_a" / "eval")
    assert len(run.episodes) == 6
    assert run.config  # the parent run's config.json


def test_eval_loaded_with_run(eval_runs_dir):
    run = load_run(eval_runs_dir / "run_a", eval_runs_dir)
    assert run.eval is not None
    assert run.eval.name == run.name == "run_a"
    assert run.eval.config == run.config
    assert set(run.eval.episodes["eval_round"]) == {0, 1}
    assert run.eval.steps is not None


def test_run_without_eval(run):
    assert run.eval is None


def test_filter_keeps_eval_and_filters_rounds(eval_runs_dir):
    run = load_run(eval_runs_dir / "run_a", eval_runs_dir)
    cut = metrics.filter_episodes(run, 0, 0, decks=["b_red"])
    assert len(cut.episodes) == 1
    assert set(cut.eval.episodes["deck"]) == {"b_red"}
    assert set(cut.eval.episodes["eval_round"]) == {0, 1}  # range is for training only
    last = metrics.filter_episodes(run.eval, 0, np.inf, eval_rounds=[1])
    assert set(last.episodes["eval_round"]) == {1}


def test_pairing_includes_eval_round(eval_runs_dir):
    a, b = (r.eval for r in load_runs(eval_runs_dir))
    pairs = stats.paired_ante(b, a)
    assert "eval_round" in pairs
    assert len(pairs) == 6  # 3 seeds x 2 rounds, none dropped as duplicates
    assert (pairs["ante"] - pairs["ante_baseline"] == 1).all()


def test_eval_curve(eval_runs_dir):
    runs = [r.eval for r in load_runs(eval_runs_dir)]
    curve = stats.eval_curve(runs)
    assert len(curve) == 4  # 2 runs x 2 rounds
    a = curve[curve["run"] == "run_a"].sort_values("eval_round")
    assert list(a["train_episodes"]) == [0, 10]
    assert list(a["games"]) == [3, 3]
    e = runs[0].episodes
    assert a["mean ante"].iloc[0] == e.loc[e["eval_round"] == 0, "ante"].mean()
    assert (a["mean ante low"] <= a["mean ante"]).all()
    assert (a["mean ante"] <= a["mean ante high"]).all()
    assert (a["win rate low"] <= a["win rate high"]).all()


def test_eval_curve_skips_training_runs(run):
    assert stats.eval_curve([run]).empty


def test_export_has_eval_curve(eval_runs_dir):
    tables = export.computed_tables(load_runs(eval_runs_dir))
    assert len(tables["eval_curve"]) == 4
