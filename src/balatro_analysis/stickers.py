"""How the shop agent handles Eternal (can't be sold) and Perishable (debuffed
after 5 rounds) jokers, from stakes 4 and 7 (log_version 17).

A port of balatro-shop-rl's ``scripts/stickers.py`` (``behaviour``), kept to give
the same numbers, per stake, over a run's shop decisions (steps.jsonl):

- Eternals held per decision;
- of the decisions with the joker slots full, the share holding an Eternal;
- for decisions holding an expired Perishable (``perish`` 0), how often one was
  sold, and how many decisions an expired Perishable is held before its sale
  (counted per episode and joker key, the selling decision included);
- expired Perishables never sold, and sells of an Eternal (a check: should be 0).

Over a group's runs (seeds): the mean of the per-run values (NaN dropped), and
the two counts summed.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from balatro_analysis.loader import Run, memo

# (metric, label, summed over runs rather than averaged)
METRICS = (
    ("eternal_held_per_decision", "Eternals held (per shop decision)", False),
    ("full_share_with_eternal", "slots full: share holding an Eternal", False),
    ("p_sell_expired", "expired Perishable held: P(sold this decision)", False),
    ("decisions_before_sold", "decisions an expired Perishable is held before sale", False),
    ("expired_never_sold", "expired Perishables never sold (total)", True),
    ("sell_eternal", "sells of an Eternal (should be 0)", True),
)
# the buy-rate rows the script shows (purchases.categories)
BUY_CATEGORIES = ("Joker: no sticker", "Joker: eternal", "Joker: perishable", "Joker")


def _stickers(e) -> list:
    st = e.get("st") if isinstance(e, dict) else None
    return list(st) if isinstance(st, list | tuple | np.ndarray) else []


def has_stickers(run: Run) -> bool:
    """Whether the run logs held jokers' stickers (log_version 17)."""
    version = run.config.get("log_version")
    if version is not None:
        return int(version) >= 17
    return run.steps is not None and "jokers_extra" in run.steps


def behaviour(run: Run) -> pd.DataFrame:
    """One row per stake (as episodes.csv has it) with ``METRICS`` and
    ``decisions`` / ``expired_decisions``.  Empty without steps.  Memoised."""

    def build() -> pd.DataFrame:
        steps = run.steps
        cols = [m for m, _, _ in METRICS]
        if steps is None or steps.empty:
            return pd.DataFrame(columns=["stake", "decisions", "expired_decisions", *cols])
        stake_of = dict(zip(run.episodes["episode"], run.episodes["stake"], strict=True))
        c: dict = defaultdict(Counter)
        held_for: dict = defaultdict(list)
        run_len: dict[tuple, int] = {}
        use = [
            x
            for x in ("episode", "jokers_before", "jokers_extra", "joker_slots", "action_name")
            if x in steps
        ]
        for st in steps[use].to_dict("records"):  # in log order, as the script reads it
            stake = stake_of.get(st["episode"], "?")
            extra = st.get("jokers_extra")
            extra = list(extra) if isinstance(extra, list | tuple | np.ndarray) else []
            k = c[stake]
            k["decisions"] += 1
            eternal = [i for i, e in enumerate(extra) if e and "eternal" in _stickers(e)]
            expired = [
                i
                for i, e in enumerate(extra)
                if e and "perishable" in _stickers(e) and e.get("perish") == 0
            ]
            held = st["jokers_before"]
            held = list(held) if isinstance(held, list | tuple | np.ndarray) else []
            slots = st.get("joker_slots")
            slots = 99 if slots is None or (isinstance(slots, float) and np.isnan(slots)) else slots
            full = len(held) >= slots
            k["full"] += full
            k["full_with_eternal"] += full and bool(eternal)
            k["eternal_held"] += len(eternal)
            name = str(st["action_name"])
            sold = int(name.rsplit("_", 1)[1]) if name.startswith("sell_joker") else None
            k["sell_eternal"] += sold in eternal
            if expired:
                k["expired_decisions"] += 1
                k["expired_sold"] += sold in expired
            for i in expired:
                key = (st["episode"], held[i])
                run_len[key] = run_len.get(key, 0) + 1
                if sold == i:
                    held_for[stake].append(run_len.pop(key))
        for ep, _key in run_len:  # never sold
            c[stake_of.get(ep, "?")]["expired_never_sold"] += 1
        rows = []
        for stake, k in c.items():
            d = max(k["decisions"], 1)
            rows.append(
                {
                    "stake": stake,
                    "decisions": k["decisions"],
                    "expired_decisions": k["expired_decisions"],
                    "eternal_held_per_decision": k["eternal_held"] / d,
                    "full_share_with_eternal": k["full_with_eternal"] / max(k["full"], 1),
                    "p_sell_expired": k["expired_sold"] / k["expired_decisions"]
                    if k["expired_decisions"]
                    else float("nan"),
                    "decisions_before_sold": float(np.mean(held_for[stake]))
                    if held_for[stake]
                    else float("nan"),
                    "expired_never_sold": k["expired_never_sold"],
                    "sell_eternal": k["sell_eternal"],
                }
            )
        return pd.DataFrame(rows, columns=["stake", "decisions", "expired_decisions", *cols])

    return memo(run, "stickers:behaviour", build)


def behaviour_table(groups: dict[str, list[Run]]) -> pd.DataFrame:
    """Per stake and metric, a column per group: the mean over its runs (NaN
    dropped) or, for the counts, the sum; NaN where no run has a value."""
    rows = []
    per = {g: [behaviour(r) for r in runs] for g, runs in groups.items()}
    stakes = sorted({s for frames in per.values() for f in frames for s in f["stake"]}, key=str)
    for stake in stakes:
        for metric, label, total in METRICS:
            row = {"stake": stake, "metric": label}
            for g, frames in per.items():
                vals = [
                    float(v)
                    for f in frames
                    for v in f.loc[f["stake"] == stake, metric]
                    if not np.isnan(float(v))
                ]
                row[g] = (np.sum(vals) if total else np.mean(vals)) if vals else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
