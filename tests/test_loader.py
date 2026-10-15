from balatro_analysis.loader import find_runs, load_run, load_runs


def test_find_runs(runs_dir):
    assert [p.name for p in find_runs(runs_dir)] == ["run_small"]


def test_find_runs_missing_dir(tmp_path):
    assert find_runs(tmp_path / "nope") == []


def test_load_full_run(run):
    assert run.name == "run_small"
    assert len(run.episodes) == 3
    assert run.config["play_agent"] == "SmartPlayAgent"
    assert len(run.steps) == 8
    assert len(run.rounds) == 5
    assert run.steps.loc[2, "jokers_before"] == ["j_joker"]
    assert run.episodes.loc[2, "final_jokers"] == ""


def test_load_episodes_only(episodes_only_dir):
    (run,) = load_runs(episodes_only_dir)
    assert run.name == "old_run"
    assert run.config == {}
    assert run.steps is None
    assert run.rounds is None


def test_load_run_without_root(runs_dir):
    assert load_run(runs_dir / "run_small").name == "run_small"
