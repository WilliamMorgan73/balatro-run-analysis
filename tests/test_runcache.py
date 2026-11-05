import shutil
from pathlib import Path

import pandas as pd

from balatro_analysis import runcache
from balatro_analysis.loader import Run, load_run

V5 = Path(__file__).parent / "fixtures" / "v5"


def _copy_run(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    shutil.copytree(V5, root)
    return root


def _assert_same(a: Run, b: Run) -> None:
    assert (a.name, a.path, a.config) == (b.name, b.path, b.config)
    for table in ("episodes", "steps", "rounds", "shops"):
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
