"""The test set (<run>/test/, balatro-shop-rl log_version 14): the final policy on
fixed games, loaded as Run.test and the dashboard's default view."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_app import TABS

from balatro_analysis import export, find, metrics, runcache, stats
from balatro_analysis.loader import find_runs, load_run, load_runs, log_files

APP = str(Path(__file__).parent.parent / "app.py")


def test_test_dir_is_not_a_run(test_runs_dir):
    assert [p.name for p in find_runs(test_runs_dir)] == ["run_a", "run_b"]


def test_test_loaded_with_run(test_runs_dir):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    assert run.test is not None and run.eval is not None
    assert run.test.name == run.name == "run_a"
    assert run.test.config == run.config
    assert list(run.test.episodes["seed"]) == ["TEST_0", "TEST_1", "TEST_2"]
    assert run.test.hands is not None
    assert any(f.parent.name == "test" for f in log_files(run.path))


def test_test_dir_given_as_root_still_loads(test_runs_dir):
    run = load_run(test_runs_dir / "run_a" / "test")
    assert len(run.episodes) == 3
    assert run.config  # the parent run's config.json


def test_run_without_test(run):
    assert run.test is None


def test_filter_keeps_test(test_runs_dir):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    cut = metrics.filter_episodes(run, 0, 0, decks=["b_red"])
    assert set(cut.test.episodes["deck"]) == {"b_red"}
    assert len(cut.test.episodes) == (run.test.episodes["deck"] == "b_red").sum()
    assert metrics.filter_episodes(run, 0, np.inf) is run


def test_test_games_pair_across_runs(test_runs_dir):
    a, b = (r.test for r in load_runs(test_runs_dir))
    pairs = stats.paired_ante(b, a)
    assert len(pairs) == 3
    assert (pairs["ante"] - pairs["ante_baseline"] == 1).all()


def test_cache_round_trip_keeps_test(test_runs_dir):
    path = test_runs_dir / "run_a"
    folder = runcache.cache_dir()
    first = runcache.load_run_cached(path, test_runs_dir, folder)
    second = runcache.load_run_cached(path, test_runs_dir, folder)
    assert second.test is not None
    assert second.test.episodes.equals(first.test.episodes)


def test_cache_sees_a_new_test_set(test_runs_dir, tmp_path):
    path = tmp_path / "run_a"
    shutil.copytree(test_runs_dir / "run_a", path)
    shutil.rmtree(path / "test")
    folder = runcache.cache_dir()
    assert runcache.load_run_cached(path, tmp_path, folder).test is None
    shutil.copytree(test_runs_dir / "run_a" / "test", path / "test")
    assert runcache.load_run_cached(path, tmp_path, folder).test is not None


def test_run_index_scores_on_test(test_runs_dir, run):
    summaries = {p.name: find.run_summary(p, p.name) for p in find_runs(test_runs_dir)}
    summaries["train_only"] = find.run_summary(run.path, "train_only")
    index = find.run_index(summaries).set_index("run")
    a = load_run(test_runs_dir / "run_a")
    assert index.at["run_a", "scored_on"] == "test"
    assert index.at["run_a", "mean_ante"] == a.test.episodes["ante"].mean()
    assert index.at["run_a", "episodes"] == len(a.episodes)  # the number trained
    assert index.at["train_only", "scored_on"] == "train"
    assert find.best_per_group(index.reset_index()) == ["run_b"]


def test_export_eval_curve_from_test_runs(test_runs_dir):
    runs = load_runs(test_runs_dir)
    tables = export.computed_tables([r.test for r in runs], eval_runs=[r.eval for r in runs])
    assert len(tables["eval_curve"]) == 4  # 2 runs x 2 eval rounds, not the test games


# --- app ---------------------------------------------------------------------


def _app(root: Path, phase: str | None = None, tab: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    if phase:
        at.session_state["phase"] = phase
    if tab:
        at.session_state["main_tab"] = tab
    at.run()
    return at


def test_app_defaults_to_test(test_runs_dir):
    at = _app(test_runs_dir)
    assert not at.exception
    assert at.sidebar.segmented_control(key="phase").value == "Test"
    assert at.sidebar.segmented_control(key="phase").options == ["Test", "Evaluation", "Training"]
    assert any("**test games**" in i.value for i in at.info)
    assert not any(s.label == "Eval round" for s in at.sidebar.selectbox)
    assert any(r.label == "Eval metric" for r in at.radio)  # the learning curve stays


def test_app_without_test_defaults_to_training(eval_runs_dir):
    at = _app(eval_runs_dir)
    assert not at.exception
    assert at.sidebar.segmented_control(key="phase").value == "Training"
    assert at.sidebar.segmented_control(key="phase").options == ["Evaluation", "Training"]


@pytest.mark.parametrize("tab", TABS)
@pytest.mark.parametrize("phase", ["Test", "Evaluation", "Training"])
def test_every_tab_renders_in_every_phase(test_runs_dir, phase, tab):
    at = _app(test_runs_dir, phase, tab)
    assert not at.exception


def _add_replay(run_dir: Path, tag: str = "retest") -> Path:
    """A replay of ``run_dir``'s test games (balatro-shop-rl's replay_test.py)."""
    out = run_dir / "replay" / tag
    shutil.copytree(run_dir / "test", out)
    return out


def test_replay_dir_is_not_a_run(test_runs_dir):
    _add_replay(test_runs_dir / "run_a")
    assert [p.name for p in find_runs(test_runs_dir)] == ["run_a", "run_b"]


def test_replay_dir_given_as_root_reads_the_runs_config(test_runs_dir):
    replay = _add_replay(test_runs_dir / "run_a")
    assert load_run(replay).config == load_run(test_runs_dir / "run_a").config


def test_adding_a_replay_keeps_the_cached_run(test_runs_dir):
    path = test_runs_dir / "run_a"
    before = runcache.fingerprint(path)
    _add_replay(path)
    assert runcache.fingerprint(path) == before
