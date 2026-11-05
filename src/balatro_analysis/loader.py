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
"""

from __future__ import annotations

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


def memo(run: Run, key: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """``build()`` once per run, so tabs calling the same helper share it.  Each
    caller gets its own (copy-on-write) view to add columns to."""
    if key not in run._memo:
        run._memo[key] = build()
    return run._memo[key].copy(deep=False)


EVAL_DIR = "eval"


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


def _read_jsonl(path: Path) -> pd.DataFrame | None:
    """One row per line.  Parsed with pyarrow (a fraction of the time and memory
    of ``pd.read_json``); logs Arrow can't type, such as card lists mixing plain
    keys and objects, fall back to pandas."""
    if not path.is_file():
        return None
    data = _complete_bytes(path)
    if not data.strip():
        return None
    try:
        table = pj.read_json(
            pa.BufferReader(data), parse_options=pj.ParseOptions(newlines_in_values=False)
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        table = None
    if table is not None and not any(pa.types.is_struct(t) for t in table.schema.types):
        del data
        return _from_arrow(table)
    # objects as values (dicts) would come back with every key any row had
    df = pd.read_json(io.StringIO(data.decode()), lines=True)
    for col, mask in MASKS.items():
        if col in df:
            df[mask] = mask_from_lists(df.pop(col))
    return df


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


def load_run(path: str | Path, root: str | Path | None = None) -> Run:
    """Load one run directory. ``root`` (if given) makes the name relative to it."""
    path = Path(path)
    name = str(path.relative_to(root)) if root else path.name
    if name == ".":
        name = path.name
    config_path = path / "config.json"
    if not config_path.is_file() and is_eval_dir(path):
        config_path = path.parent / "config.json"  # loaded on its own: use the run's config
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    run = _load_tables(path, name, config)
    if (path / EVAL_DIR / "episodes.csv").is_file():
        run.eval = _load_tables(path / EVAL_DIR, name, config)
        if run.eval.episodes.empty:  # first eval round still being played
            run.eval = None
    return run


def _load_tables(path: Path, name: str, config: dict) -> Run:
    episodes = pd.read_csv(io.StringIO(_complete_lines(path / "episodes.csv")))
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

    steps = _read_jsonl(path / "steps.jsonl")
    rounds = _read_jsonl(path / "rounds.jsonl")
    shops = _read_jsonl(path / "shops.jsonl")
    hands = _read_jsonl(path / "hands.jsonl")
    decks = _read_jsonl(path / "decks.jsonl")
    # v1 configs have no log_version; a run with no config at all is taken as current
    v1 = bool(config) and config.get("log_version", 1) < 2
    if rounds is not None and v1 and "blind_key" in rounds:
        # v1 logs read the ante after the round ended, and beating a boss advances
        # the ante first: move beaten bosses back to the ante they were played in.
        boss_won = ~rounds["blind_key"].isin(["bl_small", "bl_big"]) & rounds["beaten"].astype(bool)
        rounds.loc[boss_won, "ante"] -= 1
    if steps is not None:
        for col in ("jokers_before", "consumables_before"):
            if col in steps:  # pandas fallback: nulls (and NaN) as []
                steps[col] = _shared_lists(
                    [v if isinstance(v, list) else None for v in steps[col]]
                )

    return Run(
        name=name,
        path=path,
        episodes=episodes,
        config=config,
        steps=steps,
        rounds=rounds,
        shops=shops,
        hands=hands,
        decks=decks,
    )


def load_runs(root: str | Path) -> list[Run]:
    return [load_run(p, root) for p in find_runs(root)]
