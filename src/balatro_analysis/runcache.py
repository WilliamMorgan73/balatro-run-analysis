"""An on-disk cache of loaded runs, so the dashboard starts without re-parsing
every run's JSONL logs (seconds per run) when nothing has changed.

A run is cached in two layers.  The run itself (its episodes, config and eval and
test runs, no tables) is pickled to one small file with a fingerprint of its
source files (size and mtime), of the loading code (``code_version``: edits to
comments and docstrings don't count) and of the pandas and pyarrow versions; any
mismatch, or a cache that can't be read, parses the run again and rewrites it.
Each JSONL log is read only when a view first asks for its table
(``loader.TableReader``), and the parsed table is pickled to a file of its own with
that log's fingerprint (``CachedTableReader``), so the next session reads it back
in a fraction of the time.
Whenever a run is written, cached files not used for ``STALE_DAYS`` (deleted or
renamed runs, old fingerprints) are removed.

A run whose logs have only grown since it was cached (still training), loaded by
the same code with the same ``config.json``, is brought up to date by reading
just the new lines (``loader.extend_run``) rather than parsed again.

The folder is ``$BALATRO_ANALYSIS_CACHE``, else ``$XDG_CACHE_HOME/balatro-analysis``,
else ``~/.cache/balatro-analysis``.  ``BALATRO_ANALYSIS_CACHE=0`` (or empty)
turns the cache off.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from balatro_analysis import loader
from balatro_analysis.loader import (
    LOG_FILES,
    SPLIT_DIRS,
    TABLES,
    UNREAD,
    Progress,
    Run,
    Source,
    TableReader,
    extend_run,
    load_run,
)

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


_versions: dict[tuple, str] = {}


def code_version() -> str:
    """A hash of the code that decides what a loaded Run holds (``loader.py`` and
    this module), ignoring comments, docstrings and formatting: Runs loaded by
    code with the same version are the same, so they can be reused."""
    files = [Path(loader.__file__), Path(__file__)]
    stamp = tuple(f.stat().st_mtime_ns for f in files)
    if stamp not in _versions:
        h = hashlib.sha1()
        for f in files:
            h.update(_code_text(f.read_text()).encode())
        _versions.clear()
        _versions[stamp] = h.hexdigest()
    return _versions[stamp]


def _code_text(source: str) -> str:
    """``source``'s syntax tree without docstrings (comments are already gone)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def fingerprint(path: Path) -> tuple:
    """What a cached copy of the run at ``path`` must match to be reused."""
    dirs = [path, *(path / d for d in SPLIT_DIRS)]
    if loader.is_split_dir(path):
        dirs.append(path.parent)  # a split loaded on its own reads the run's config
    files = []
    for d in dirs:
        for name in SOURCES:
            f = d / name
            if f.is_file():
                st = f.stat()
                files.append((str(f), st.st_size, st.st_mtime_ns))
    return (code_version(), pd.__version__, pa.__version__, tuple(files))


def _cache_file(folder: Path, path: Path, root: str | Path | None) -> Path:
    key = f"{path.resolve()}\0{Path(root).resolve() if root else ''}"
    return folder / f"{hashlib.sha1(key.encode()).hexdigest()}.pkl"


def can_extend(old: tuple, new: tuple) -> bool:
    """A run loaded with fingerprint ``old`` can be brought up to fingerprint
    ``new`` by ``extend_run``: same code and library versions, same configs."""

    def configs(fp: tuple) -> set:
        return {f for f in fp[3] if f[0].endswith("config.json")}

    return len(old) == len(new) == 4 and old[:3] == new[:3] and configs(old) == configs(new)


def load_run_cached(
    path: str | Path,
    root: str | Path | None = None,
    folder: Path | None = None,
    progress: Progress | None = None,
    previous: Run | None = None,
) -> Run:
    """``loader.load_run``, from the cache in ``folder`` when the run hasn't
    changed since it was cached.  ``folder=None``: no cache.  ``progress`` as for
    ``load_run``; a cached copy reports once, as "cached copy".

    ``previous``: this run as loaded earlier (by this function, from the same
    ``path`` and ``root``), e.g. still in memory.  If only its logs have grown,
    only the new lines are read, and the cache isn't read."""
    path = Path(path)
    expected = fingerprint(path)
    if previous is not None and previous._fingerprint == expected:
        return previous
    base = (
        previous if previous is not None and can_extend(previous._fingerprint, expected) else None
    )
    file = _cache_file(folder, path, root) if folder is not None else None
    if base is None and file is not None:
        try:
            with file.open("rb") as fh:
                if progress is not None:
                    progress("cached copy", 0.0)
                cached = pickle.load(fh)
            run = cached["run"]
            if cached["fingerprint"] == expected:
                run._fingerprint = expected
                _touch(file)
                return run
            if can_extend(cached["fingerprint"], expected):
                base = run
        except Exception:  # missing, stale format or corrupt: parse it again
            pass
    run = None
    if base is not None:
        try:
            run = extend_run(base, progress)
        except Exception:  # anything unexpected: a full parse is always right
            run = None
    if run is None:
        run = load_run(path, root, progress)
    run._fingerprint = expected
    _attach_readers(run, folder)
    if file is None:
        return run
    if progress is not None:
        progress("saving to cache", 1.0)
    _dump(file, {"fingerprint": expected, "run": skeleton(run)})
    remove_stale(folder)
    return run


def skeleton(run: Run) -> Run:
    """``run`` without the tables it has read (left to its reader) or anything
    memoised: what the run-level cache holds."""
    out = Run(run.name, run.path, run.episodes, run.config, **dict.fromkeys(TABLES, UNREAD))
    out._reader, out._fingerprint = run._reader, run._fingerprint
    for split in SPLIT_DIRS:
        sub = getattr(run, split)
        setattr(out, split, skeleton(sub) if sub is not None else None)
    return out


def _attach_readers(run: Run, folder: Path | None) -> None:
    """Read ``run``'s tables (and its eval and test runs') through the table cache
    in ``folder`` (None: straight from the logs)."""
    for r in (run, *run.splits()):
        if isinstance(r._reader, TableReader):
            r._reader = (
                CachedTableReader(r.path, r.config, folder)
                if folder is not None
                else TableReader(r.path, r.config)
            )


def _dump(file: Path, obj: object) -> None:
    """Pickle ``obj`` to ``file`` atomically; a read-only or full disk is skipped."""
    tmp = file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            pickle.dump(obj, fh, protocol=5)
        os.replace(tmp, file)
    except OSError:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class CachedTableReader(TableReader):
    """``TableReader`` through a pickle per log in ``folder``, reused while the log's
    size and mtime, the loading code and the library versions are unchanged."""

    folder: Path | None = None

    def at(self, path: Path, config: dict) -> CachedTableReader:
        return CachedTableReader(path, config, self.folder)

    def _fingerprint(self, file: Path) -> tuple:
        st = file.stat()
        return (code_version(), pd.__version__, pa.__version__, st.st_size, st.st_mtime_ns)

    def read(self, table: str) -> tuple[pd.DataFrame | None, Source | None]:
        file = self.file(table)
        if self.folder is None or not file.is_file():
            return super().read(table)
        key = hashlib.sha1(str(file.resolve()).encode()).hexdigest()
        cached = self.folder / f"{key}.table.pkl"
        expected = self._fingerprint(file)
        try:
            with cached.open("rb") as fh:
                got = pickle.load(fh)
            if got["fingerprint"] == expected:
                _touch(cached)
                return got["frame"], got["source"]
        except Exception:  # missing, stale format or corrupt: read the log
            pass
        frame, source = super().read(table)
        _dump(cached, {"fingerprint": expected, "frame": frame, "source": source})
        return frame, source


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
