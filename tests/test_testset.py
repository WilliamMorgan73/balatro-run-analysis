"""The held-out test set written to <run>/test/ by --test-episodes (log_version 14)."""

import io
import zipfile

import numpy as np

from balatro_analysis import export, metrics, runcache, stats
from balatro_analysis.loader import find_runs, load_run, load_runs, log_files


def test_test_dir_is_not_a_run(test_runs_dir):
    assert [p.name for p in find_runs(test_runs_dir)] == ["run_a", "run_b", "run_c"]


def test_test_loaded_with_run(test_runs_dir):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    assert run.test is not None and run.eval is not None
    assert run.test.name == run.name == "run_a"
    assert run.test.config == run.config
    assert run.test.path == test_runs_dir / "run_a" / "test"
    assert list(run.test.episodes["seed"]) == ["TEST_0", "TEST_1", "TEST_2"]
    assert run.test.hands is not None
    assert load_run(test_runs_dir / "run_c").test is None


def test_test_dir_given_as_root_still_loads(test_runs_dir):
    run = load_run(test_runs_dir / "run_a" / "test")
    assert len(run.episodes) == 3
    assert run.config  # the parent run's config.json


def test_test_logs_count_as_the_runs(test_runs_dir):
    run_dir = test_runs_dir / "run_a"
    assert run_dir / "test" / "steps.jsonl" in log_files(run_dir)
    before = runcache.fingerprint(run_dir)
    e = run_dir / "test" / "episodes.csv"
    e.write_text(e.read_text() + e.read_text().splitlines(keepends=True)[-1])
    assert runcache.fingerprint(run_dir) != before


def test_cached_run_keeps_its_test_set(test_runs_dir, tmp_path):
    run_dir, cache = test_runs_dir / "run_a", tmp_path / "cache"
    runcache.load_run_cached(run_dir, test_runs_dir, cache)
    again = runcache.load_run_cached(run_dir, test_runs_dir, cache)
    assert len(again.test.episodes) == 3


def test_test_set_appearing_is_read_on(test_runs_dir, monkeypatch):
    """Training ends, then the test games are played: only test/ is new."""
    run_dir = test_runs_dir / "run_a"
    held = test_runs_dir.parent / "held"
    (run_dir / "test").rename(held)
    early = runcache.load_run_cached(run_dir, test_runs_dir)
    assert early.test is None
    held.rename(run_dir / "test")
    monkeypatch.setattr(runcache, "load_run", lambda *a, **k: 1 / 0)
    later = runcache.load_run_cached(run_dir, test_runs_dir, previous=early)
    assert list(later.test.episodes["seed"]) == ["TEST_0", "TEST_1", "TEST_2"]


def test_filter_applies_to_test_set(test_runs_dir):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    assert metrics.filter_episodes(run, 0, np.inf) is run  # nothing filtered: no copy
    cut = metrics.filter_episodes(run, 0, 0, decks=["b_red"])
    assert set(cut.test.episodes["deck"]) == {"b_red"}
    assert len(cut.test.episodes) == (run.test.episodes["deck"] == "b_red").sum()


def test_test_games_pair_across_runs(test_runs_dir):
    a, b = (r.test for r in load_runs(test_runs_dir) if r.test is not None)
    pairs = stats.paired_ante(b, a)
    assert len(pairs) == 3
    assert (pairs["ante"] - pairs["ante_baseline"] == 2).all()


def test_export_names_test_tables(test_runs_dir):
    runs = [r.test for r in load_runs(test_runs_dir) if r.test is not None]
    data = export.export_zip(runs, {"phase": "Test"}, phase="Test")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        readme = z.read("README.txt").decode()
    tables = [n for n in names if n.startswith(("tables/", "latex/"))]
    assert tables and all(n.split("/")[1].startswith("test_") for n in tables)
    assert "held-out test set" in readme
    plain = export.export_zip(runs, {})
    with zipfile.ZipFile(io.BytesIO(plain)) as z:
        assert "tables/summary.csv" in z.namelist()
