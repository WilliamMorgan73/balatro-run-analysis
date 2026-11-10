import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from balatro_analysis import loader, runcache
from balatro_analysis.loader import TABLES, Run, load_run

V5 = Path(__file__).parent / "fixtures" / "v5"
HANDS = Path(__file__).parent / "fixtures" / "hands"
RUNS = Path(__file__).parent / "fixtures" / "runs"


def _copy_run(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    shutil.copytree(V5, root)
    return root


def _assert_same(a: Run, b: Run) -> None:
    assert (a.name, a.path, a.config) == (b.name, b.path, b.config)
    for table in ("episodes", *TABLES):
        x, y = getattr(a, table), getattr(b, table)
        assert (x is None) == (y is None)
        if x is not None:
            pd.testing.assert_frame_equal(x, y)


def test_cached_run_matches_parsed(tmp_path):
    root = _copy_run(tmp_path)
    cache = tmp_path / "cache"
    first = runcache.load_run_cached(root / "run_v5", root, cache)
    assert len(list(cache.glob("*.pkl"))) == 1
    second = runcache.load_run_cached(root / "run_v5", root, cache)
    fresh = load_run(root / "run_v5", root)
    _assert_same(second, fresh)
    _assert_same(second.eval, fresh.eval)
    _assert_same(first, second)


def test_changed_file_is_parsed_again(tmp_path):
    root = _copy_run(tmp_path)
    cache = tmp_path / "cache"
    before = runcache.load_run_cached(root / "run_v5", root, cache)
    csv = root / "run_v5" / "episodes.csv"
    lines = csv.read_text().splitlines(keepends=True)
    csv.write_text("".join(lines[:-1]))  # one episode fewer
    after = runcache.load_run_cached(root / "run_v5", root, cache)
    assert len(after.episodes) == len(before.episodes) - 1


def test_corrupt_cache_falls_back_to_parsing(tmp_path):
    root = _copy_run(tmp_path)
    cache = tmp_path / "cache"
    runcache.load_run_cached(root / "run_v5", root, cache)
    (file,) = cache.glob("*.pkl")
    file.write_bytes(b"not a pickle")
    run = runcache.load_run_cached(root / "run_v5", root, cache)
    _assert_same(run, load_run(root / "run_v5", root))
    assert file.read_bytes() != b"not a pickle"  # rewritten


def test_cache_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(runcache.ENV, str(tmp_path / "here"))
    assert runcache.cache_dir() == tmp_path / "here"
    for off in ("", "0"):
        monkeypatch.setenv(runcache.ENV, off)
        assert runcache.cache_dir() is None
    monkeypatch.delenv(runcache.ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert runcache.cache_dir() == tmp_path / "xdg" / "balatro-analysis"


def test_no_cache_writes_nothing(tmp_path):
    root = _copy_run(tmp_path)
    run = runcache.load_run_cached(root / "run_v5", root, None)
    _assert_same(run, load_run(root / "run_v5", root))
    assert not (tmp_path / "cache").exists()


def test_stale_files_are_removed(tmp_path):
    import os
    import time

    root = _copy_run(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    old, recent = cache / "old.pkl", cache / "recent.pkl"
    for f in (old, recent):
        f.write_bytes(b"x")
    month = time.time() - (runcache.STALE_DAYS + 1) * 86400
    os.utime(old, (month, month))
    runcache.load_run_cached(root / "run_v5", root, cache)
    assert not old.exists() and recent.exists()
    assert len(list(cache.glob("*.pkl"))) == 2  # recent + the run just cached


def test_code_text_ignores_comments_and_docstrings():
    src = 'def f(x):\n    """Doc."""\n    return x + 1\n'
    edited = (
        '# a comment\ndef f(x):\n    """Other doc,\n    longer."""\n\n    return x + 1  # why\n'
    )
    changed = 'def f(x):\n    """Doc."""\n    return x + 2\n'
    assert runcache._code_text(src) == runcache._code_text(edited)
    assert runcache._code_text(src) != runcache._code_text(changed)
    assert runcache._code_text('def g():\n    "only a docstring"\n')  # still valid


def test_code_version_is_stable():
    assert runcache.code_version() == runcache.code_version()
    assert runcache.code_version() in runcache.fingerprint(V5 / "run_v5")


# --- Reading on from where a run was loaded (runs still training) ------------


def _logs(run_dir: Path) -> list[Path]:
    return [f for d in (run_dir, run_dir / "eval") for f in sorted(d.glob("*.jsonl"))]


def _cut(run_dir: Path, fraction: float) -> dict[Path, bytes]:
    """Truncate every JSONL log (and episodes.csv) to ``fraction`` of its lines,
    as if training were part-way; returns the full contents to put back."""
    full = {}
    for f in [*_logs(run_dir), *run_dir.rglob("episodes.csv")]:
        data = f.read_bytes()
        full[f] = data
        lines = data.splitlines(keepends=True)
        keep = max(int(len(lines) * fraction), 1 if f.name == "episodes.csv" else 0)
        f.write_bytes(b"".join(lines[:keep]))
    return full


def _no_full_parse(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("parsed the whole run again")

    monkeypatch.setattr(runcache, "load_run", refuse)


def _training_copy(tmp_path: Path, source: Path, with_hands: bool = False) -> Path:
    root = tmp_path / "runs"
    shutil.copytree(source, root / source.name)
    if with_hands:  # eval hands/decks don't read as Arrow: read again whole
        for name in ("hands.jsonl", "decks.jsonl"):
            shutil.copy(HANDS / name, root / source.name / "eval" / name)
    return root


@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("in_memory", [False, True])
def test_grown_run_reads_only_new_lines(tmp_path, monkeypatch, fraction, in_memory):
    root = _training_copy(tmp_path, V5 / "run_v5", with_hands=True)
    run_dir, cache = root / "run_v5", tmp_path / "cache"
    full = _cut(run_dir, fraction)
    early = runcache.load_run_cached(run_dir, root, cache)
    for f, data in full.items():
        f.write_bytes(data)
    fresh = load_run(run_dir, root)
    _no_full_parse(monkeypatch)
    later = runcache.load_run_cached(run_dir, root, cache, previous=early if in_memory else None)
    _assert_same(later, fresh)
    _assert_same(later.eval, fresh.eval)
    _assert_same(runcache.load_run_cached(run_dir, root, cache), fresh)  # as cached


def test_run_read_on_several_times(tmp_path, monkeypatch):
    root = _training_copy(tmp_path, RUNS / "run_small")
    run_dir = root / "run_small"
    full = _cut(run_dir, 0.0)
    run = runcache.load_run_cached(run_dir, root)
    _no_full_parse(monkeypatch)
    for f, data in full.items():  # one line at a time, the last one in two halves
        lines = data.splitlines(keepends=True)
        for i in range(len(lines)):
            f.write_bytes(b"".join(lines[: i + 1])[: -3 if i == len(lines) - 1 else None])
            run = runcache.load_run_cached(run_dir, root, previous=run)
        f.write_bytes(data)
        run = runcache.load_run_cached(run_dir, root, previous=run)
    _assert_same(run, load_run(run_dir, root))


def test_eval_folder_appearing_is_loaded(tmp_path, monkeypatch):
    root = _training_copy(tmp_path, V5 / "run_v5")
    run_dir = root / "run_v5"
    shutil.move(run_dir / "eval", tmp_path / "eval")
    early = runcache.load_run_cached(run_dir, root)
    assert early.eval is None
    shutil.move(tmp_path / "eval", run_dir / "eval")
    _no_full_parse(monkeypatch)
    later = runcache.load_run_cached(run_dir, root, previous=early)
    _assert_same(later.eval, load_run(run_dir, root).eval)


def test_rewritten_log_is_read_again_whole(tmp_path, monkeypatch):
    root = _training_copy(tmp_path, V5 / "run_v5")
    run_dir = root / "run_v5"
    early = runcache.load_run_cached(run_dir, root)
    steps = run_dir / "steps.jsonl"
    lines = steps.read_text().splitlines(keepends=True)
    row = json.loads(lines[0])
    row["dollars"] = 999
    steps.write_text(json.dumps(row) + "\n" + "".join(lines[1:]) + lines[1])
    _no_full_parse(monkeypatch)  # still no whole-run parse: just this log
    later = runcache.load_run_cached(run_dir, root, previous=early)
    _assert_same(later, load_run(run_dir, root))
    assert later.steps["dollars"].iloc[0] == 999


def test_shrunk_log_is_read_again_whole(tmp_path):
    root = _training_copy(tmp_path, V5 / "run_v5")
    run_dir = root / "run_v5"
    early = runcache.load_run_cached(run_dir, root)
    _cut(run_dir, 0.5)
    _assert_same(runcache.load_run_cached(run_dir, root, previous=early), load_run(run_dir, root))


def test_changed_config_parses_again(tmp_path, monkeypatch):
    root = _training_copy(tmp_path, V5 / "run_v5")
    run_dir = root / "run_v5"
    early = runcache.load_run_cached(run_dir, root)
    config = json.loads((run_dir / "config.json").read_text())
    config["group"] = "renamed"
    (run_dir / "config.json").write_text(json.dumps(config))
    calls = []
    real = runcache.load_run
    monkeypatch.setattr(runcache, "load_run", lambda *a, **k: calls.append(1) or real(*a, **k))
    later = runcache.load_run_cached(run_dir, root, previous=early)
    assert calls and later.config["group"] == "renamed"


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_joker_state_read_in_parts(tmp_path, monkeypatch, cut):
    """log_version 15 joker_state starts as only nulls (no scaling joker yet): the
    first part's schema can't type the rest, which is then read again whole."""
    run = tmp_path / "v15"
    run.mkdir()
    (run / "episodes.csv").write_text("episode,ante,won,return\n0,2,0,1.0\n")
    held = [
        (["j_jolly", "j_wily"], [None, None]),
        (["j_trousers", "j_jolly"], [{"mult": 6}, None]),
        (["j_ramen"], [{"x_mult": 1.91}]),
        ([], []),
    ]
    lines = [
        json.dumps({"episode": 0, "step": i, "jokers_before": k, "joker_state": s}) + "\n"
        for i, (k, s) in enumerate(held)
    ]
    (run / "steps.jsonl").write_text("".join(lines[:cut]))
    early = runcache.load_run_cached(run)
    (run / "steps.jsonl").write_text("".join(lines))
    _no_full_parse(monkeypatch)
    later = runcache.load_run_cached(run, previous=early)
    pd.testing.assert_frame_equal(later.steps, load_run(run).steps)


def test_extend_run_leaves_the_old_run_alone(tmp_path):
    root = _training_copy(tmp_path, RUNS / "run_small")
    run_dir = root / "run_small"
    full = _cut(run_dir, 0.5)
    early = load_run(run_dir, root)
    before = {t: getattr(early, t) for t in ("episodes", *TABLES)}
    lengths = {t: None if df is None else len(df) for t, df in before.items()}
    for f, data in full.items():
        f.write_bytes(data)
    later = loader.extend_run(early)
    assert {t: None if (df := getattr(early, t)) is None else len(df) for t in before} == lengths
    assert len(later.steps) > lengths["steps"]


def _read_all(run):
    """Read every table of ``run`` and its eval/test runs (tables are lazy), as a
    dashboard that has shown them would have: those are the ones read on."""
    for r in (run, *run.splits()):
        for t in TABLES:
            getattr(r, t)
    return run


@pytest.mark.parametrize("source", [V5 / "run_v5", RUNS / "run_small"])
def test_appended_logs_are_not_read_again(tmp_path, monkeypatch, source):
    root = _training_copy(tmp_path, source)
    run_dir = root / source.name
    full = _cut(run_dir, 0.5)
    early = _read_all(load_run(run_dir, root))
    for f, data in full.items():
        f.write_bytes(data)
    fresh = _read_all(load_run(run_dir, root))

    real = loader._read_jsonl

    def refuse(path):
        assert not path.is_file(), f"read {path} again whole"
        return real(path)

    monkeypatch.setattr(loader, "_read_jsonl", refuse)
    later = loader.extend_run(early)
    _assert_same(later, fresh)
    if fresh.eval is not None:
        _assert_same(later.eval, fresh.eval)


def test_unchanged_pandas_log_is_not_read_again(tmp_path, monkeypatch):
    root = _training_copy(tmp_path, V5 / "run_v5", with_hands=True)
    run_dir = root / "run_v5"
    full = _cut(run_dir, 0.5)
    for name in ("hands.jsonl", "decks.jsonl"):  # eval hands complete already
        f = run_dir / "eval" / name
        f.write_bytes(full.pop(f))
    early = _read_all(load_run(run_dir, root))
    for f, data in full.items():
        f.write_bytes(data)
    fresh = _read_all(load_run(run_dir, root))
    real = loader._read_jsonl

    def refuse(path):
        assert not path.is_file(), f"read {path} again whole"
        return real(path)

    monkeypatch.setattr(loader, "_read_jsonl", refuse)
    later = loader.extend_run(early)
    assert later.eval.hands is early.eval.hands
    _assert_same(later.eval, fresh.eval)
