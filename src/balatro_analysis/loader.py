"""Load training runs written by balatro-shop-rl's ``scripts/train.py``.

The format is specified in ``docs/run-format.md``.

A run is a directory containing ``episodes.csv``, and optionally
``config.json``, ``steps.jsonl`` (one row per shop decision), ``rounds.jsonl``
(one row per blind played, newer runs with the cash-out breakdown) and
``shops.jsonl`` (newer runs: what each shop offered).

Runs made with ``--eval-every`` also have an ``eval/`` folder in the same format
(greedy episodes on fixed seeds, with ``eval_round`` and ``train_episodes``
columns).  It is loaded as ``Run.eval`` rather than listed as a run of its own.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

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
    text = path.read_text()
    return text if text.endswith("\n") else text[: text.rfind("\n") + 1]


def _read_jsonl(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    text = _complete_lines(path)
    if not text.strip():
        return None
    return pd.read_json(io.StringIO(text), lines=True)


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
    # v1 configs have no log_version; a run with no config at all is taken as current
    v1 = bool(config) and config.get("log_version", 1) < 2
    if rounds is not None and v1 and "blind_key" in rounds:
        # v1 logs read the ante after the round ended, and beating a boss advances
        # the ante first: move beaten bosses back to the ante they were played in.
        boss_won = ~rounds["blind_key"].isin(["bl_small", "bl_big"]) & rounds["beaten"].astype(bool)
        rounds.loc[boss_won, "ante"] -= 1
    if steps is not None:
        for col in ("jokers_before", "consumables_before"):
            if col in steps:
                steps[col] = [v if isinstance(v, list) else [] for v in steps[col]]

    return Run(
        name=name,
        path=path,
        episodes=episodes,
        config=config,
        steps=steps,
        rounds=rounds,
        shops=shops,
    )


def load_runs(root: str | Path) -> list[Run]:
    return [load_run(p, root) for p in find_runs(root)]
