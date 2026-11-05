"""Load training runs written by balatro-shop-rl's ``scripts/train.py``.

The format is specified in ``docs/run-format.md``.

A run is a directory containing ``episodes.csv``, and optionally
``config.json``, ``steps.jsonl`` (one row per shop decision), ``rounds.jsonl``
(one row per blind played, newer runs with the cash-out breakdown) and
``shops.jsonl`` (newer runs: what each shop offered).

Runs made with ``--eval-every`` also have an ``eval/`` folder in the same format
(greedy episodes on fixed seeds, with ``eval_round`` and ``train_episodes``
columns).  It is loaded as ``Run.eval`` rather than listed as a run of its own.
From ``log_version`` 7, ``eval/`` also has ``hands.jsonl`` (every play and
discard) and ``decks.jsonl`` (the deck at the start of each blind).

The logs only ever grow, so a run loaded earlier can be brought up to date by
reading just what was appended since (``extend_run``).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj

UNKNOWN = "unknown"


@dataclass
class Run:
    name: str
    path: Path
    episodes: pd.DataFrame
    config: dict = field(default_factory=dict)
    steps: pd.DataFrame | None = None
    rounds: pd.DataFrame | None = None
    shops: pd.DataFrame | None = None
    hands: pd.DataFrame | None = None  # eval/ only, log_version 7
    decks: pd.DataFrame | None = None  # eval/ only, log_version 7
    eval: Run | None = None  # greedy episodes on the fixed eval seeds, if logged
    # frames derived from this run's tables (see ``memo``), so the tables must not
    # change once read; a filtered or replace()d run starts empty
    _memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # how far each JSONL log was read (``Source`` by table name), for ``extend_run``;
    # a filtered or replace()d run starts empty
    _sources: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # what the run was loaded from, set by ``runcache``
    _fingerprint: tuple = field(default=(), init=False, repr=False, compare=False)


def memo(run: Run, key: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """``build()`` once per run, so tabs calling the same helper share it.  Each
    caller gets its own (copy-on-write) view to add columns to."""
    if key not in run._memo:
        run._memo[key] = build()
    return run._memo[key].copy(deep=False)


EVAL_DIR = "eval"
# what a run (and its eval/ folder) is read from, in the order it is read
LOG_FILES = (
    "episodes.csv",
    "config.json",
    "steps.jsonl",
    "rounds.jsonl",
    "shops.jsonl",
    "hands.jsonl",
    "decks.jsonl",
)

TABLES = ("steps", "rounds", "shops", "hands", "decks")  # the JSONL logs, as Run fields

# ``progress(label, fraction)``: called before each file of a run is read, with
# the fraction of the run's log bytes already read
Progress = Callable[[str, float], None]


def is_eval_dir(path: Path) -> bool:
    """An ``eval/`` folder inside a run (it belongs to that run)."""
    return path.name == EVAL_DIR and (path.parent / "episodes.csv").is_file()


def find_runs(root: str | Path) -> list[Path]:
    """Every directory at or below ``root`` that contains an ``episodes.csv``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.parent
        for p in root.rglob("episodes.csv")
        if p.parent == root or not is_eval_dir(p.parent)
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
    """The log files a run at ``path`` is read from (its own and its eval/ folder's)."""
    path = Path(path)
    return [d / n for d in (path, path / EVAL_DIR) for n in LOG_FILES if (d / n).is_file()]


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
    if not config_path.is_file() and is_eval_dir(path):
        config_path = path.parent / "config.json"  # loaded on its own: use the run's config
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    run = _load_tables(path, name, config, report)
    run.eval = _load_eval(run, None, report)
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
    out.eval = _load_eval(out, run.eval, report)
    return out


def _load_eval(run: Run, previous: Run | None, report: Callable[[Path], None]) -> Run | None:
    path = run.path / EVAL_DIR
    if not (path / "episodes.csv").is_file():
        return None
    out = _load_tables(path, run.name, run.config, report, previous=previous)
    return None if out.episodes.empty else out  # first eval round still being played


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
    tables, sources = {}, {}
    for table in TABLES:
        file = path / f"{table}.jsonl"
        report(file)
        got = None
        if previous is not None and table in previous._sources:
            got = _read_on(file, getattr(previous, table), previous._sources[table], table, config)
        if got is None:
            df, src = _read_jsonl(file)
            got = (_fix_table(table, df, config) if df is not None else None), src
        tables[table], source = got
        if source is not None:
            sources[table] = source
    run = Run(name=name, path=path, episodes=episodes, config=config, **tables)
    run._sources = sources
    return run


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
