"""Tables (the JSONL logs) are read when first asked for, not when a run loads."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import loader, metrics, runcache
from balatro_analysis.loader import TABLES, TableBudgetError, load_run

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def reads(monkeypatch):
    """The JSONL files read whole, in order."""
    seen: list[Path] = []
    real = loader._read_jsonl

    def counting(path):
        if path.is_file():
            seen.append(path)
        return real(path)

    monkeypatch.setattr(loader, "_read_jsonl", counting)
    return seen


def test_loading_reads_no_logs(test_runs_dir, reads):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    assert reads == []
    assert run.loaded() == [] and run.test.loaded() == []
    assert run.has("steps") and run.test.has("hands") and not run.has("hands")
    assert reads == []  # has() doesn't read either


def test_a_table_is_read_once_when_asked_for(test_runs_dir, reads):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    steps = run.steps
    assert [p.name for p in reads] == ["steps.jsonl"]
    assert run.steps is steps and len(reads) == 1
    assert run.loaded() == ["steps"]
    assert run.read_bytes() == (run.path / "steps.jsonl").stat().st_size
    assert run.hands is None  # no log: None, like before
    _ = run.test.decks
    assert [p.parent.name for p in reads[1:]] == ["test"]


def test_filtered_run_cuts_its_parents_table_when_asked(test_runs_dir, reads):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    cut = metrics.filter_episodes(run, 0, 0)
    assert reads == []
    assert set(cut.steps["episode"]) <= {0}
    assert len(reads) == 1 and run.loaded() == ["steps"]  # read once, on the parent


def test_unload_frees_and_rereads(test_runs_dir, reads):
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    _ = run.steps, run.test.steps
    assert run.unload() > 0
    assert run.loaded() == [] and run.test.loaded() == [] and run.read_bytes() == 0
    _ = run.steps
    assert len(reads) == 3


def test_table_cache_skips_parsing_next_time(test_runs_dir, reads, tmp_path):
    folder = tmp_path / "cache"
    first = runcache.load_run_cached(test_runs_dir / "run_a", test_runs_dir, folder)
    steps = first.steps
    assert len(reads) == 1
    second = runcache.load_run_cached(test_runs_dir / "run_a", test_runs_dir, folder)
    assert second is not first and second.loaded() == []
    assert second.steps.equals(steps)
    assert len(reads) == 1  # from the table cache
    (test_runs_dir / "run_a" / "steps.jsonl").write_text(
        (test_runs_dir / "run_a" / "steps.jsonl").read_text()
    )  # rewritten: new mtime
    third = runcache.load_run_cached(test_runs_dir / "run_a", test_runs_dir, folder)
    _ = third.steps
    assert len(reads) == 2


def test_cached_run_holds_no_tables(test_runs_dir, tmp_path):
    folder = tmp_path / "cache"
    run = runcache.load_run_cached(test_runs_dir / "run_a", test_runs_dir, folder)
    _ = [getattr(run, t) for t in TABLES]
    skel = runcache.skeleton(run)
    assert skel.loaded() == [] and skel.test.loaded() == []
    assert skel.steps.equals(run.steps)  # read back through its reader


def test_hook_can_refuse_a_read(test_runs_dir, monkeypatch):
    def refuse(run, table, nbytes):
        raise TableBudgetError(run.name, table, nbytes, 1)

    monkeypatch.setattr(loader, "table_read_hook", refuse)
    run = load_run(test_runs_dir / "run_a", test_runs_dir)
    with pytest.raises(TableBudgetError):
        _ = run.steps
    assert run.loaded() == []


# --- app ---------------------------------------------------------------------


def _app(root: Path, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def test_overview_on_test_games_reads_no_logs(test_runs_dir, reads):
    at = _app(test_runs_dir)
    assert not at.exception
    assert at.sidebar.segmented_control(key="phase").value == "Test"
    assert [p for p in reads if p.name == "steps.jsonl"] == []


def test_a_tab_over_the_memory_budget_warns(test_runs_dir, monkeypatch):
    monkeypatch.setenv("BALATRO_ANALYSIS_RUN_MEMORY", "0.000001")  # ~1 KB
    at = _app(test_runs_dir, main_tab="Actions", phase="Training")
    assert not at.exception
    assert any("Not enough memory for this view" in w.value for w in at.warning)
