"""An on-disk cache of loaded runs, so the dashboard starts without re-parsing
every run's JSONL logs (seconds per run) when nothing has changed.

Each run is pickled to one file in the cache folder with a fingerprint of its
source files (size and mtime), of ``loader.py`` and of the pandas and pyarrow
versions; any mismatch, or a cache that can't be read, parses the run again and
rewrites it.  Whenever a run is written, cached files not used for ``STALE_DAYS``
(deleted or renamed runs, old fingerprints) are removed.

The folder is ``$BALATRO_ANALYSIS_CACHE``, else ``$XDG_CACHE_HOME/balatro-analysis``,
else ``~/.cache/balatro-analysis``.  ``BALATRO_ANALYSIS_CACHE=0`` (or empty)
turns the cache off.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from balatro_analysis import loader
from balatro_analysis.loader import EVAL_DIR, LOG_FILES, Progress, Run, load_run

ENV = "BALATRO_ANALYSIS_CACHE"
STALE_DAYS = 30  # cached runs not read or written for this long are deleted
SOURCES = LOG_FILES


def cache_dir() -> Path | None:
    """Where runs are cached, or None if the cache is turned off."""
    if ENV in os.environ:
        value = os.environ[ENV]
        return None if value in ("", "0") else Path(value).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "balatro-analysis"


def _code_hash() -> str:
    return hashlib.sha1(Path(loader.__file__).read_bytes()).hexdigest()


def fingerprint(path: Path) -> tuple:
    """What a cached copy of the run at ``path`` must match to be reused."""
    dirs = [path, path / EVAL_DIR]
    if loader.is_eval_dir(path):
        dirs.append(path.parent)  # an eval dir loaded on its own reads the run's config
    files = []
    for d in dirs:
        for name in SOURCES:
            f = d / name
            if f.is_file():
                st = f.stat()
                files.append((str(f), st.st_size, st.st_mtime_ns))
    return (_code_hash(), pd.__version__, pa.__version__, tuple(files))


def _cache_file(folder: Path, path: Path, root: str | Path | None) -> Path:
    key = f"{path.resolve()}\0{Path(root).resolve() if root else ''}"
    return folder / f"{hashlib.sha1(key.encode()).hexdigest()}.pkl"


def load_run_cached(
    path: str | Path,
    root: str | Path | None = None,
    folder: Path | None = None,
    progress: Progress | None = None,
) -> Run:
    """``loader.load_run``, from the cache in ``folder`` when the run hasn't
    changed since it was cached.  ``folder=None``: no cache.  ``progress`` as for
    ``load_run``; a cached copy reports once, as "cached copy"."""
    path = Path(path)
    if folder is None:
        return load_run(path, root, progress)
    file = _cache_file(folder, path, root)
    expected = fingerprint(path)
    try:
        with file.open("rb") as fh:
            if progress is not None:
                progress("cached copy", 0.0)
            cached = pickle.load(fh)
        if cached["fingerprint"] == expected:
            _touch(file)
            return cached["run"]
    except Exception:  # missing, stale format or corrupt: parse it again
        pass
    run = load_run(path, root, progress)
    if progress is not None:
        progress("saving to cache", 1.0)
    tmp = file.with_suffix(f".{os.getpid()}.tmp")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            pickle.dump({"fingerprint": expected, "run": run}, fh, protocol=5)
        os.replace(tmp, file)
    except OSError:  # read-only or full disk: run without the cache
        tmp.unlink(missing_ok=True)
    remove_stale(folder)
    return run


def _touch(file: Path) -> None:
    """Mark a cached run as used (its mtime), so ``remove_stale`` keeps it."""
    try:
        os.utime(file)
    except OSError:
        pass


def remove_stale(folder: Path, days: float = STALE_DAYS) -> None:
    """Delete cached runs (and leftover temp files) not used for ``days``."""
    cutoff = time.time() - days * 86400
    for f in [*folder.glob("*.pkl"), *folder.glob("*.tmp")]:
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:  # gone already, or not ours to delete
            pass
