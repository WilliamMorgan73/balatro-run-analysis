"""Load training runs written by jackdaw-balatro's ``scripts/train_random_shop.py``.

A run is a directory containing ``episodes.csv``, and optionally
``config.json``, ``steps.jsonl`` (one row per shop decision) and
``rounds.jsonl`` (one row per blind played).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Run:
    name: str
    path: Path
    episodes: pd.DataFrame
    config: dict = field(default_factory=dict)
    steps: pd.DataFrame | None = None
    rounds: pd.DataFrame | None = None


def find_runs(root: str | Path) -> list[Path]:
    """Every directory at or below ``root`` that contains an ``episodes.csv``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob("episodes.csv"))


def _read_jsonl(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return pd.read_json(path, lines=True)


def load_run(path: str | Path, root: str | Path | None = None) -> Run:
    """Load one run directory. ``root`` (if given) makes the name relative to it."""
    path = Path(path)
    name = str(path.relative_to(root)) if root else path.name
    if name == ".":
        name = path.name

    episodes = pd.read_csv(path / "episodes.csv")
    if "final_jokers" in episodes:
        episodes["final_jokers"] = episodes["final_jokers"].fillna("").astype(str)

    config_path = path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}

    steps = _read_jsonl(path / "steps.jsonl")
    rounds = _read_jsonl(path / "rounds.jsonl")
    if steps is not None:
        for col in ("jokers_before", "consumables_before"):
            if col in steps:
                steps[col] = steps[col].apply(lambda v: list(v) if isinstance(v, list) else [])

    return Run(
        name=name,
        path=path,
        episodes=episodes,
        config=config,
        steps=steps,
        rounds=rounds,
    )


def load_runs(root: str | Path) -> list[Run]:
    return [load_run(p, root) for p in find_runs(root)]
