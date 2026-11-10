"""Endless mode (balatro-shop-rl ``--endless``, log_version 17): a won run plays
on past ante 8 until it is lost, so ``ante`` (the highest ante reached) can pass 8
while ``won`` still means the ante-8 boss was beaten.

Two views, as in balatro-shop-rl's ``scripts/endless_report.py``: like for like
with normal runs (ante capped at 8, ``capped``), and how far won games go past 8
(``past_eight``)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from balatro_analysis.loader import Run

WIN_ANTE = 8  # the ante whose boss wins the run


def is_endless(run: Run) -> bool:
    return bool(run.config.get("endless"))


def capped(run: Run, cap: int = WIN_ANTE) -> Run:
    """``run`` with each episode's ante capped at ``cap`` (episodes only: what the
    study tables read), so endless runs compare like for like with normal ones.
    Kept on ``run`` (its memo), so tables memoised on the copy are reused."""
    key = f"endless:capped:{cap}"
    if key not in run._memo:
        e = run.episodes.copy()
        e["ante"] = e["ante"].clip(upper=cap)
        run._memo[key] = Run(run.name, run.path, e, run.config, replay=run.replay)
    return run._memo[key]


@dataclass
class PastEight:
    games: int
    won: int
    counts: dict[int, int]  # highest ante of a won game: games
    mean_win_ante: float  # mean over runs (seeds) of each run's mean highest ante of a win
    seed_low: float
    seed_high: float


def past_eight(runs: list[Run]) -> PastEight | None:
    """How far the won games went: as ``endless_report.past_eight`` (the highest
    ante of each won game, counted, and its mean per run, then over runs).  None
    without a won game."""
    counts: Counter[int] = Counter()
    win_antes, games = [], 0
    for r in runs:
        e = r.episodes
        games += len(e)
        wins = [int(a) for a in e.loc[e["won"].astype(bool), "ante"]]
        counts.update(wins)
        if wins:
            win_antes.append(np.mean(wins))
    if not counts:
        return None
    return PastEight(
        games=games,
        won=sum(counts.values()),
        counts=dict(sorted(counts.items())),
        mean_win_ante=float(np.mean(win_antes)),
        seed_low=float(min(win_antes)),
        seed_high=float(max(win_antes)),
    )
