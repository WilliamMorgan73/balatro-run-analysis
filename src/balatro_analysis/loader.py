"""Load training runs written by balatro-shop-rl's ``scripts/train.py``.

The format is specified in ``docs/run-format.md``.

A run is a directory containing ``episodes.csv``, and optionally
``config.json``, ``steps.jsonl`` (one row per shop decision), ``rounds.jsonl``
(one row per blind played, newer runs with the cash-out breakdown) and
``shops.jsonl`` (newer runs: what each shop offered).

Runs made with ``--eval-every`` also have an ``eval/`` folder in the same format
(greedy episodes on fixed seeds, with ``eval_round`` and ``train_episodes``
columns).  It is loaded as ``Run.eval`` rather than listed as a run of its own.
Runs from balatro-shop-rl's log_version 14 also end with a ``test/`` folder: one fixed
set of games (``TEST_0``..) played once by the final policy, the same for every run.
It is loaded as ``Run.test``: the results to report, where eval is the learning curve.
From ``log_version`` 7, ``eval/`` also has ``hands.jsonl`` (every play and
discard) and ``decks.jsonl`` (the deck at the start of each blind).

Loading a run reads only its ``episodes.csv`` files (and config).  Each JSONL log
(``steps``, ``rounds``, ...) is read the first time something asks for it
(``run.steps``), by the run's ``TableReader``, and kept: a view that only needs
episodes never reads the hundreds of MB of training steps.  ``table_read_hook`` (if
set) hears about each read first, so a caller can make room for it.

The logs only ever grow, so a run loaded earlier can be brought up to date by
reading just what was appended since (``extend_run``).
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj

UNKNOWN = "unknown"


class _Unread:
    """Passed for a table to leave it unread: ``Run(..., steps=UNREAD)``."""

    def __repr__(self) -> str:
        return "UNREAD"


UNREAD = _Unread()
# serialises table reads, so two sessions asking for the same table read it once
_READ_LOCK = threading.RLock()
# called as hook(run, table, file_bytes) before a table is read from disk (not for
# a filtered run cutting its parent's table); may raise to refuse the read
table_read_hook: Callable[[Run, str, int], None] | None = None


class TableBudgetError(MemoryError):
    """A table wasn't read because it wouldn't fit in the memory set aside for runs."""

    def __init__(self, run: str, table: str, needed: int, budget: int):
        super().__init__(
            f"{run}: {table} ({needed / 1e9:.1f} GB) doesn't fit in {budget / 1e9:.1f} GB"
        )
        self.run, self.table, self.needed, self.budget = run, table, needed, budget


class _LazyTable:
    """A ``Run`` table read on first access through the run's ``_reader``.  As a
    dataclass field: defaults to None (no table); ``UNREAD`` leaves it to be read."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.table, self.attr = name, f"_{name}_df"

    def __get__(self, obj: Run | None, owner: type | None = None):
        if obj is None:
            return None  # the field's default
        try:
            return obj.__dict__[self.attr]
        except KeyError:
            return obj._read_table(self.table)

    def __set__(self, obj: Run, value) -> None:
        if value is UNREAD:
            obj.__dict__.pop(self.attr, None)
        else:
            obj.__dict__[self.attr] = value


@dataclass(eq=False, repr=False)
class Run:
    name: str
    path: Path
    episodes: pd.DataFrame
    config: dict = field(default_factory=dict)
    steps: pd.DataFrame | None = _LazyTable()
    rounds: pd.DataFrame | None = _LazyTable()
    shops: pd.DataFrame | None = _LazyTable()
    hands: pd.DataFrame | None = _LazyTable()  # eval/ and test/ only, log_version 7
    decks: pd.DataFrame | None = _LazyTable()  # eval/ and test/ only, log_version 7
    offers: pd.DataFrame | None = _LazyTable()  # eval/ and test/ only, log_version 12
    eval: Run | None = None  # greedy episodes on the fixed eval seeds, if logged
    test: Run | None = None  # the final policy on the fixed test games, if logged
    # frames derived from this run's tables (see ``memo``), so the tables must not
    # change once read; a filtered or replace()d run starts empty
    _memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # how far each JSONL log was read (``Source`` by table name), for ``extend_run``;
    # a filtered or replace()d run starts empty
    _sources: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # what the run was loaded from, set by ``runcache``
    _fingerprint: tuple = field(default=(), init=False, repr=False, compare=False)
    # reads the tables left ``UNREAD`` (a ``TableReader``); None: there are none
    _reader: TableReader | None = field(default=None, init=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return f"Run({self.name!r}, {len(self.episodes)} episodes, read: {self.loaded()})"

    def loaded(self) -> list[str]:
        """The tables read (or given) so far."""
        return [t for t in TABLES if f"_{t}_df" in self.__dict__]

    def has(self, table: str) -> bool:
        """Whether the run has ``table``, without reading it if it hasn't been."""
        if f"_{table}_df" in self.__dict__:
            return self.__dict__[f"_{table}_df"] is not None
        return self._reader is not None and self._reader.exists(table)

    def table_bytes(self, table: str) -> int:
        """Size of ``table``'s log on disk (0 if none, or not read from disk)."""
        return self._reader.size(table) if self._reader is not None else 0

    def read_bytes(self) -> int:
        """Log bytes behind the tables read so far, its eval and test runs' too:
        what they take in memory, roughly (``memory.estimate``)."""
        own = sum(src.end for t, src in self._sources.items() if f"_{t}_df" in self.__dict__)
        return own + sum(s.read_bytes() for s in self.splits())

    def splits(self) -> list[Run]:
        return [r for r in (getattr(self, d) for d in SPLIT_DIRS) if r is not None]

    def unload(self) -> int:
        """Drop the tables read from disk (they are read again when next asked
        for), here and in the eval and test runs, and everything memoised on
        them.  Returns the log bytes freed."""
        freed = 0
        if self._reader is not None and self._reader.from_disk:
            with _READ_LOCK:
                for t in self.loaded():
                    src = self._sources.pop(t, None)
                    freed += src.end if src is not None else 0
                    setattr(self, t, UNREAD)
                self._memo.clear()
        return freed + sum(s.unload() for s in self.splits())

    def _read_table(self, table: str) -> pd.DataFrame | None:
        if self._reader is None:
            return None
        with _READ_LOCK:
            if f"_{table}_df" in self.__dict__:  # read by another thread meanwhile
                return self.__dict__[f"_{table}_df"]
            if self._reader.from_disk and table_read_hook is not None:
                table_read_hook(self, table, self._reader.size(table))
            df, source = self._reader.read(table)
            setattr(self, table, df)
            if source is not None:
                self._sources[table] = source
            return df


def memo(run: Run, key: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """``build()`` once per run, so tabs calling the same helper share it.  Each
    caller gets its own (copy-on-write) view to add columns to."""
    if key not in run._memo:
        run._memo[key] = build()
    return run._memo[key].copy(deep=False)


EVAL_DIR = "eval"
TEST_DIR = "test"
# folders inside a run holding its other episodes, in the run's format: Run attributes
SPLIT_DIRS = (EVAL_DIR, TEST_DIR)
# <run>/replay/<tag>/: the run's test games played again (balatro-shop-rl's
# scripts/replay_test.py), in the test/ format plus a replay.json; ``Run.replays``
REPLAY_DIR = "replay"
# what a run (and its eval/ folder) is read from, in the order it is read
LOG_FILES = (
    "episodes.csv",
    "config.json",
    "steps.jsonl",
    "rounds.jsonl",
    "shops.jsonl",
    "hands.jsonl",
    "decks.jsonl",
    "offers.jsonl",
)

# the JSONL logs, as Run fields
TABLES = ("steps", "rounds", "shops", "hands", "decks", "offers")

# ``progress(label, fraction)``: called before each file of a run is read, with
# the fraction of the run's log bytes already read
Progress = Callable[[str, float], None]


def split_owner(path: Path) -> Path | None:
    """The run folder that ``path`` is a split of (an ``eval/``, ``test/`` or
    ``replay/<tag>/`` folder inside a run), or None if it isn't one."""
    if path.name in SPLIT_DIRS and (path.parent / "episodes.csv").is_file():
        return path.parent
    owner = path.parent.parent
    if path.parent.name == REPLAY_DIR and (owner / "episodes.csv").is_file():
        return owner
    return None


def is_split_dir(path: Path) -> bool:
    """An ``eval/``, ``test/`` or ``replay/<tag>/`` folder inside a run (it belongs
    to that run, so it isn't a run of its own)."""
    return split_owner(path) is not None


def find_runs(root: str | Path) -> list[Path]:
    """Every directory at or below ``root`` that contains an ``episodes.csv``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.parent
        for p in root.rglob("episodes.csv")
        if p.parent == root or not is_split_dir(p.parent)
    )


def _complete_lines(path: Path) -> str:
    """The file's text without an unterminated last line: a run that is still
    training may be mid-way through writing it."""
    return _complete_bytes(path).decode()


def _complete_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data if data.endswith(b"\n") else data[: data.rfind(b"\n") + 1]


# list columns of action ids stored as one bitmask per row (bit a = action a):
# ``legal`` (17 ids on every shop decision) is ~200 MB as Python lists per run
MASKS = {"legal": "legal_mask"}


TAIL_CHECK = 4096  # bytes before where a log was read up to, checked unchanged


@dataclass(frozen=True)
class Source:
    """How much of one JSONL log a run has read, so it can read on from there."""

    end: int  # bytes read (whole lines only)
    tail: str  # sha1 of the TAIL_CHECK bytes before ``end``: the file wasn't rewritten
    schema: pa.Schema | None  # the Arrow schema it was parsed with; None: pandas


def _tail_hash(data: bytes) -> str:
    return hashlib.sha1(data[-TAIL_CHECK:]).hexdigest()


def _read_jsonl(path: Path) -> tuple[pd.DataFrame | None, Source | None]:
    """One row per line, and how far the file was read (None: no file)."""
    if not path.is_file():
        return None, None
    data = _complete_bytes(path)
    df, schema = _parse_jsonl(data)
    return df, Source(len(data), _tail_hash(data), schema)


def _parse_jsonl(data: bytes, schema: pa.Schema | None = None):
    """``(frame, schema)`` of JSONL ``data``.  Parsed with pyarrow (a fraction of
    the time and memory of ``pd.read_json``); logs Arrow can't type, such as card
    lists mixing plain keys and objects, fall back to pandas (schema None).  With
    ``schema`` (from an earlier part of the same log), Arrow only, and the result's
    schema may differ from it: the caller checks."""
    if not data.strip():
        return None, schema
    try:
        table = _arrow_table(data, schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        if schema is not None:
            raise
        table = None
    if table is not None and not any(pa.types.is_struct(t) for t in table.schema.types):
        del data
        return _from_arrow(table), table.schema
    if schema is not None:
        raise pa.ArrowInvalid("log no longer reads as Arrow")
    # objects as values (dicts) would come back with every key any row had
    df = pd.read_json(io.StringIO(data.decode()), lines=True)
    for col, mask in MASKS.items():
        if col in df:
            df[mask] = mask_from_lists(df.pop(col))
    return df, None


def _arrow_table(data: bytes, schema: pa.Schema | None = None) -> pa.Table:
    """``data`` as an Arrow table; ArrowInvalid if Arrow can't read it right.

    pyarrow (26) can build a list-of-objects column with broken offsets when some
    items are null, as in ``steps.jsonl`` ``joker_state`` (log_version 15); reading
    again with the inferred schema given explicitly reads it right.  Only nested
    columns are checked, so other logs pay nothing extra.

    ``schema``: read with it, except fields whose type holds null (nothing logged
    in them yet), which are inferred again; fields missing from ``data`` come back
    as nulls of the schema's type, in the schema's order."""

    def read(schema: pa.Schema | None = None) -> pa.Table:
        options = pj.ParseOptions(newlines_in_values=False, explicit_schema=schema)
        return pj.read_json(pa.BufferReader(data), parse_options=options)

    if schema is not None:
        known = pa.schema([f for f in schema if not _holds_null(f.type)])
        table = read(known)
        if not _nested_valid(table):
            raise pa.ArrowInvalid("pyarrow built an invalid nested column")
        for f in schema:
            if f.name not in table.schema.names:
                table = table.append_column(f, pa.nulls(len(table), f.type))
        order = [*schema.names, *(n for n in table.schema.names if n not in schema.names)]
        return table.select(order)
    table = read()
    if _nested_valid(table):
        return table
    table = read(table.schema)
    if _nested_valid(table):
        return table
    raise pa.ArrowInvalid("pyarrow built an invalid nested column")


def _holds_null(t: pa.DataType) -> bool:
    """A null type, or a nested type with one inside (e.g. a list of only [])."""
    return pa.types.is_null(t) or any(_holds_null(t.field(i).type) for i in range(t.num_fields))


def _nested_valid(table: pa.Table) -> bool:
    for name, typ in zip(table.schema.names, table.schema.types, strict=True):
        if pa.types.is_nested(typ):
            try:
                table.column(name).validate(full=True)
            except pa.ArrowInvalid:
                return False
    return True


def _from_arrow(table: pa.Table) -> pd.DataFrame:
    """Plain columns as pandas would read them; lists of scalars as Python lists
    (rows holding equal lists share one list object, so they cost little: don't
    mutate them); lists of objects (shop items) as Arrow-backed columns, which
    iterate as dicts; ``MASKS`` columns as bitmasks."""
    lists, arrow, masks = {}, {}, {}
    for name, typ in zip(table.schema.names, table.schema.types, strict=True):
        if not pa.types.is_list(typ):
            continue
        col = table.column(name).combine_chunks()
        if name in MASKS and pa.types.is_integer(typ.value_type):
            masks[MASKS[name]] = _mask_from_arrow(col)
        elif pa.types.is_struct(typ.value_type) or pa.types.is_list(typ.value_type):
            arrow[name] = pd.arrays.ArrowExtensionArray(pc.fill_null(col, pa.scalar([], typ)))
        else:
            lists[name] = _shared_lists(col.to_pylist())
    plain = table.drop_columns([*lists, *arrow, *(c for c in MASKS if c in table.schema.names)])
    plain = plain.cast(
        pa.schema(
            pa.field(f.name, pa.float64()) if pa.types.is_null(f.type) else f for f in plain.schema
        )
    )  # an all-null column (e.g. a breakdown only logged later): NaN, not None
    df = plain.to_pandas()
    order = table.schema.names
    for name, values in {**lists, **arrow}.items():
        df.insert(min(order.index(name), len(df.columns)), name, values)
    for name, values in masks.items():
        df[name] = values
    return df


def _shared_lists(values: list) -> list[list]:
    """``values`` with nulls as ``[]`` and equal lists replaced by one object."""
    seen: dict[tuple, list] = {}
    return [seen.setdefault(tuple(v), v) if v else seen.setdefault((), []) for v in values]


def _mask_dtype(top: int):
    return np.int32 if top < 31 else np.int64


def _mask_from_arrow(col: pa.ListArray) -> np.ndarray:
    flat = pc.list_flatten(col).to_numpy(zero_copy_only=False)
    parents = pc.list_parent_indices(col).to_numpy(zero_copy_only=False)
    top = int(flat.max()) if len(flat) else 0
    dtype = _mask_dtype(top)
    out = np.zeros(len(col), dtype=dtype)
    np.bitwise_or.at(out, parents, np.left_shift(dtype(1), flat.astype(dtype)))
    return out


def mask_from_lists(values: pd.Series) -> np.ndarray:
    """Bitmask per row of a column of int lists (null/missing: 0)."""
    lists = [v if isinstance(v, list) else [] for v in values]
    top = max((max(v) for v in lists if v), default=0)
    dtype = _mask_dtype(top)
    return np.fromiter((sum(1 << int(a) for a in set(v)) for v in lists), dtype, len(lists))


def mask_actions(mask: int) -> list[int]:
    """The action ids set in one ``legal_mask`` value."""
    mask = int(mask)
    return [a for a in range(mask.bit_length()) if mask >> a & 1]


def log_files(path: str | Path) -> list[Path]:
    """The log files a run at ``path`` is read from (its own, its eval/ and test/ folders')."""
    path = Path(path)
    dirs = (path, *(path / d for d in SPLIT_DIRS))
    return [d / n for d in dirs for n in LOG_FILES if (d / n).is_file()]


def episodes_bytes(path: str | Path) -> int:
    """Size of the ``episodes.csv`` files of the run at ``path`` (its own, eval/ and
    test/): what loading it reads, before any table is asked for."""
    path = Path(path)
    total = 0
    for d in (path, *(path / s for s in SPLIT_DIRS)):
        try:
            total += (d / "episodes.csv").stat().st_size
        except OSError:
            pass
    return total


def log_bytes(path: str | Path) -> int:
    """Total size of ``log_files(path)``: how much loading the run reads."""
    total = 0
    for f in log_files(path):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _reporter(path: Path, progress: Progress | None) -> Callable[[Path], None]:
    """``report(file)`` before reading each file: passes ``progress`` the file and
    the fraction of the run's bytes read before it."""
    total = log_bytes(path) or 1
    done = 0

    def report(file: Path) -> None:
        nonlocal done
        if progress is None or not file.is_file():
            return
        label = str(file.relative_to(path)) if file.is_relative_to(path) else file.name
        progress(label, min(done / total, 1.0))
        done += file.stat().st_size

    return report


def load_run(
    path: str | Path, root: str | Path | None = None, progress: Progress | None = None
) -> Run:
    """Load one run directory. ``root`` (if given) makes the name relative to it.
    ``progress`` (if given) hears about each file as it is read."""
    path = Path(path)
    report = _reporter(path, progress)
    name = str(path.relative_to(root)) if root else path.name
    if name == ".":
        name = path.name
    config_path = path / "config.json"
    owner = split_owner(path)
    if not config_path.is_file() and owner is not None:
        config_path = owner / "config.json"  # loaded on its own: use the run's config
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    run = _load_tables(path, name, config, report)
    for split in SPLIT_DIRS:
        setattr(run, split, _load_split(run, split, None, report))
    return run


def extend_run(run: Run, progress: Progress | None = None) -> Run:
    """``run`` as ``load_run`` would load it now, reading only what was appended
    to its logs since it was loaded: a run still training is read up to date in
    a fraction of the time.  ``run`` must have been loaded with the same
    ``config.json``.  ``episodes.csv`` (small) is read again whole, and so is any
    log that can't be read on from where it stopped (rewritten, or read with
    pandas).  ``run`` itself is unchanged."""

    def report(file: Path) -> None:
        if progress is not None and file.is_file():
            progress(f"new data: {file.relative_to(run.path)}", 0.0)

    out = _load_tables(run.path, run.name, run.config, report, previous=run)
    for split in SPLIT_DIRS:
        setattr(out, split, _load_split(out, split, getattr(run, split), report))
    return out


def _load_split(
    run: Run, split: str, previous: Run | None, report: Callable[[Path], None]
) -> Run | None:
    """The run's ``eval/`` or ``test/`` episodes, or None if it has none (yet)."""
    path = run.path / split
    if not (path / "episodes.csv").is_file():
        return None
    out = _load_tables(path, run.name, run.config, report, previous=previous)
    return None if out.episodes.empty else out  # first eval round / test still being played


def _load_tables(
    path: Path,
    name: str,
    config: dict,
    report: Callable[[Path], None] = lambda f: None,
    previous: Run | None = None,
) -> Run:
    """The run's tables at ``path``; with ``previous`` (the same tables read
    earlier), each log is read on from where ``previous`` stopped."""
    report(path / "episodes.csv")
    episodes = _fix_episodes(
        pd.read_csv(io.StringIO(_complete_lines(path / "episodes.csv"))), config
    )
    run = Run(
        name=name, path=path, episodes=episodes, config=config, **dict.fromkeys(TABLES, UNREAD)
    )
    run._reader = (
        previous._reader.at(path, config)
        if previous is not None and previous._reader
        else TableReader(path, config)
    )
    if previous is None:
        return run
    # the tables ``previous`` had read: read on from where it stopped, or again whole
    for table in previous.loaded():
        if table not in previous._sources:
            continue
        file = path / f"{table}.jsonl"
        report(file)
        got = _read_on(
            file, previous.__dict__[f"_{table}_df"], previous._sources[table], table, config
        )
        if got is None:
            got = run._reader.read(table)
        setattr(run, table, got[0])
        if got[1] is not None:
            run._sources[table] = got[1]
    return run


@dataclass(frozen=True)
class TableReader:
    """Reads a run's JSONL logs (in ``path``) one table at a time, for ``Run``'s
    lazy tables.  Picklable, so a cached run keeps it."""

    path: Path
    config: dict = field(default_factory=dict, hash=False, compare=False)

    from_disk = True  # reads files (a table read is worth making room for)

    def at(self, path: Path, config: dict) -> TableReader:
        """The same kind of reader for another folder."""
        return TableReader(path, config)

    def file(self, table: str) -> Path:
        return self.path / f"{table}.jsonl"

    def exists(self, table: str) -> bool:
        return self.file(table).is_file()

    def size(self, table: str) -> int:
        try:
            return self.file(table).stat().st_size
        except OSError:
            return 0

    def read(self, table: str) -> tuple[pd.DataFrame | None, Source | None]:
        """``(frame, how far the file was read)``; ``(None, None)`` without a log."""
        df, source = _read_jsonl(self.file(table))
        return (_fix_table(table, df, self.config) if df is not None else None), source


class SubsetReader:
    """A filtered run's tables: its parent's, cut to ``episodes`` when first asked
    for, so filtering a run reads nothing the views don't use."""

    from_disk = False

    def __init__(self, parent: Run, episodes: set):
        self.parent, self.episodes = parent, episodes

    def exists(self, table: str) -> bool:
        return self.parent.has(table)

    def size(self, table: str) -> int:
        return 0

    def read(self, table: str) -> tuple[pd.DataFrame | None, Source | None]:
        df = getattr(self.parent, table)
        if df is None:
            return None, None
        return df[df["episode"].isin(self.episodes)].reset_index(drop=True), None


def _read_on(file: Path, cached: pd.DataFrame | None, source: Source, table: str, config: dict):
    """``(frame, source)``: ``cached`` (``file`` read up to ``source``) with the
    rows appended since, or None if ``file`` must be read again whole."""
    if not file.is_file() or file.stat().st_size < source.end:
        return None
    start = max(source.end - TAIL_CHECK, 0)
    with file.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    before, new = data[: source.end - start], data[source.end - start :]
    if _tail_hash(before) != source.tail:
        return None  # rewritten, not appended to
    new = new[: new.rfind(b"\n") + 1]
    if not new:
        return cached, source
    if source.schema is None:
        return None  # read with pandas: can't be sure a part reads the same
    try:
        df, schema = _parse_jsonl(new, source.schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return None
    if schema != source.schema:
        return None  # new fields, or values in one that was all null: types may differ
    df = _fix_table(table, df, config)
    if cached is not None:
        if list(df.columns) != list(cached.columns):
            return None
        df = pd.concat([cached, df], ignore_index=True)
    end = source.end + len(new)
    return df, Source(end, _tail_hash((before + new)[-TAIL_CHECK:]), schema)


def _fix_episodes(episodes: pd.DataFrame, config: dict) -> pd.DataFrame:
    if "final_jokers" in episodes:
        episodes["final_jokers"] = episodes["final_jokers"].fillna("").astype(str)

    # Older runs don't log deck/stake per episode: fill from config if it names
    # exactly one, else mark unknown.
    for col, cfg_key in (("deck", "decks"), ("stake", "stakes")):
        if col not in episodes:
            values = config.get(cfg_key) or []
            episodes[col] = values[0] if len(values) == 1 else UNKNOWN
        episodes[col] = episodes[col].fillna(UNKNOWN).astype(str)

    # log_version 6: ``ante`` is the ante the run ended on (9 for a win, lowered
    # by Hieroglyph/Petroglyph); ``max_ante`` is the highest ante played, which is
    # what every view means by "ante reached"
    if "max_ante" in episodes:
        episodes["final_ante"] = episodes["ante"]
        episodes["ante"] = episodes["max_ante"]
    return episodes


def _fix_table(table: str, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Row by row fixes to a log as read (so a log read in parts gets the same)."""
    # v1 configs have no log_version; a run with no config at all is taken as current
    v1 = bool(config) and config.get("log_version", 1) < 2
    if table == "rounds" and v1 and "blind_key" in df:
        # v1 logs read the ante after the round ended, and beating a boss advances
        # the ante first: move beaten bosses back to the ante they were played in.
        boss_won = ~df["blind_key"].isin(["bl_small", "bl_big"]) & df["beaten"].astype(bool)
        df.loc[boss_won, "ante"] -= 1
    if table == "steps":
        for col in ("jokers_before", "consumables_before"):
            if col in df:  # pandas fallback: nulls (and NaN) as []
                df[col] = _shared_lists([v if isinstance(v, list) else None for v in df[col]])
    return df


def load_runs(root: str | Path) -> list[Run]:
    return [load_run(p, root) for p in find_runs(root)]
