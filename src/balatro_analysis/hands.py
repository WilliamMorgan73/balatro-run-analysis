"""Play metrics from the hand log (``eval/hands.jsonl`` and ``eval/decks.jsonl``,
``log_version`` 7): what was played and discarded in each blind, and the deck.

Only eval episodes have these logs; every function returns an empty frame for
runs without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from balatro_analysis.cards import joker_keys, joker_states, parse_cards
from balatro_analysis.loader import Run, memo
from balatro_analysis.metrics import _concat, blind_type, episode_rounds

# Balatro's hand order, best first (the order of the game's Run info screen)
HAND_TYPES = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)


def has_hands(run: Run) -> bool:
    return run.hands is not None and not run.hands.empty


def _plays(run: Run) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        p = run.hands[run.hands["kind"] == "play"].copy()
        p["score"] = pd.to_numeric(p["score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        p["score_vs_target"] = p["score"] / p["target"].where(p["target"] > 0)
        return p

    return memo(run, "hand_plays", build)


def episode_hands(run: Run, episode: int) -> pd.DataFrame:
    """The episode's plays and discards in order, with next to the logged columns:

    - ``cards``: the hand as ``PlayingCard``s
    - ``jokers_held``: joker keys held when acting (None if not logged)
    - ``joker_states``: per held joker, a scaling joker's current values
      (``cards.joker_state``; None for the rest, or all None before log_version 15)
    - ``level`` and ``level_logged``: the played hand's level, as logged, or else
      its level at the start of the blind from ``decks.jsonl`` (1 if not there)
    """
    if not has_hands(run):
        return pd.DataFrame()
    h = run.hands[run.hands["episode"] == episode].sort_values(["round", "hand_no"]).copy()
    h["cards"] = [parse_cards(v) for v in h["hand"]]
    h["jokers_held"] = [joker_keys(v) for v in h["jokers"]] if "jokers" in h else None
    h["joker_states"] = [joker_states(v) for v in h["jokers"]] if "jokers" in h else None
    logged = h["hand_level"] if "hand_level" in h else pd.Series(np.nan, index=h.index)
    decks = episode_decks(run, episode)
    start = [
        (decks.at[r, "hand_levels"] if r in decks.index and "hand_levels" in decks else None)
        for r in h["round"]
    ]
    h["level_logged"] = logged.notna() & (h["kind"] == "play")
    h["level"] = [
        int(lv) if pd.notna(lv) else (levels or {}).get(t, 1) if isinstance(t, str) else None
        for lv, levels, t in zip(logged, start, h["hand_type"], strict=True)
    ]
    return h.reset_index(drop=True)


def blind_jokers(run: Run, episode: int, round_: int) -> tuple[list[str], bool, list[dict | None]]:
    """The jokers to show for a blind, whether they were logged, and each one's
    current values if it scales (``cards.joker_state``, log_version 15; None
    otherwise): from the blind's first play or discard, else what was held
    leaving the shop before it (none for the first blind)."""
    if has_hands(run) and "jokers" in run.hands:
        h = run.hands[(run.hands["episode"] == episode) & (run.hands["round"] == round_)]
        if len(h):
            first = h.sort_values("hand_no").iloc[0]["jokers"]
            if isinstance(first, list):
                return joker_keys(first) or [], True, joker_states(first) or []
    rounds = episode_rounds(run, episode)
    if "jokers" in rounds:
        match = rounds.loc[rounds["round"] == round_]
        if len(match) and isinstance(match["jokers"].iloc[0], list):
            keys = list(match["jokers"].iloc[0])
            states = match["joker_state"].iloc[0] if "joker_state" in match else None
            states = joker_states(_as_list(states)) or [None] * len(keys)
            return keys, False, states
    return [], False, []


def _as_list(value):
    """A list cell as a list: Arrow-backed cells come back as arrays."""
    return list(value) if value is not None and not isinstance(value, (str, dict)) else value


def episode_decks(run: Run, episode: int) -> pd.DataFrame:
    """The deck at the start of each blind of the episode, indexed by round."""
    if run.decks is None or run.decks.empty:
        return pd.DataFrame()
    d = run.decks[run.decks["episode"] == episode].copy()
    d["cards"] = [parse_cards(v) for v in d["deck"]]
    return d.set_index("round")


def hand_type_stats(runs: list[Run]) -> pd.DataFrame:
    """Per run and hand type: plays, share of the run's plays, mean score and
    mean score ÷ blind target.  Hand types in the game's order."""
    frames = []
    for r in runs:
        if not has_hands(r):
            continue
        p = _plays(r)
        g = p.groupby("hand_type")
        out = pd.DataFrame(
            {
                "plays": g.size(),
                "mean_score": g["score"].mean(),
                "mean_score_vs_target": g["score_vs_target"].mean(),
            }
        )
        out["share"] = out["plays"] / out["plays"].sum()
        frames.append(out.reset_index().assign(run=r.name))
    out = _concat(frames)
    if out.empty:
        return out
    order = {h: i for i, h in enumerate(HAND_TYPES)}
    out["_order"] = out["hand_type"].map(order).fillna(len(order))
    return out.sort_values(["run", "_order"]).drop(columns="_order").reset_index(drop=True)


def most_played_hand(run: Run) -> str | None:
    """The run's most played hand type (as the game's Run info shows), or None."""
    if not has_hands(run):
        return None
    counts = _plays(run)["hand_type"].value_counts()
    return str(counts.index[0]) if len(counts) else None


def blind_play(runs: list[Run]) -> pd.DataFrame:
    """One row per blind played: hands played and discards used, with its ante
    and blind type, and whether it was beaten (from ``rounds.jsonl``)."""
    frames = []
    for r in runs:
        if not has_hands(r):
            continue
        h = r.hands
        g = h.groupby(["episode", "round"])
        out = pd.DataFrame(
            {
                "ante": g["ante"].first(),
                "blind_key": g["blind_key"].first(),
                "hands_played": g["kind"].agg(lambda k: int((k == "play").sum())),
                "discards_used": g["kind"].agg(lambda k: int((k == "discard").sum())),
            }
        ).reset_index()
        out["blind type"] = out["blind_key"].map(blind_type)
        if r.rounds is not None and "beaten" in r.rounds:
            beaten = r.rounds.set_index(["episode", "round"])["beaten"].astype(bool)
            out["beaten"] = out.set_index(["episode", "round"]).index.map(beaten)
        frames.append(out.assign(run=r.name))
    return _concat(frames)


def deck_composition(runs: list[Run]) -> pd.DataFrame:
    """Mean deck at the start of each ante's blinds, per run: size and how many
    cards are enhanced, have an edition or a seal."""
    frames = []
    for r in runs:
        if r.decks is None or r.decks.empty:
            continue
        rows = []
        for ante, deck in zip(r.decks["ante"], r.decks["deck"], strict=True):
            cards = parse_cards(deck)
            rows.append(
                {
                    "ante": ante,
                    "size": len(cards),
                    "enhanced": sum(c.enhancement is not None for c in cards),
                    "edition": sum(c.edition is not None for c in cards),
                    "sealed": sum(c.seal is not None for c in cards),
                }
            )
        d = pd.DataFrame(rows).groupby("ante").mean().reset_index()
        frames.append(d.assign(run=r.name))
    return _concat(frames)
