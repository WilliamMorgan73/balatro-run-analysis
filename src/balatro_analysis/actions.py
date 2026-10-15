"""Shop behaviour: what the agent does, in what context, in what order, and
how that changes over training.

A shop visit is one (episode, round), as in ``economy.shop_visits``.
Newer runs (``log_version`` 3) log which actions were ``legal`` at each
decision; ``availability`` and ``legal_counts`` return an empty frame for
older runs.

Like ``metrics``, these are pure pandas, key-only, and return long frames with
a ``run`` column (or take one run and return a table for it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from balatro_analysis import metrics
from balatro_analysis.loader import Run

FAMILIES = [
    "Buy card",
    "Buy pack",
    "Buy voucher",
    "Sell joker",
    "Sell consumable",
    "Use consumable",
    "Reroll",
    "Leave shop",
]
_PREFIXES = {
    "buy_card": "Buy card",
    "buy_pack": "Buy pack",
    "sell_joker": "Sell joker",
    "sell_cons": "Sell consumable",
    "use_cons": "Use consumable",
}
MONEY_BRACKETS = ["$0-4", "$5-9", "$10-14", "$15-19", "$20-24", "$25+"]
JOKERS_HELD = ["0", "1", "2", "3", "4", "5+"]
NEXT_BLIND = ["Small", "Big", "Boss"]
MAX_POSITION = 10  # positions past this are pooled as "10+"
POSITIONS = [str(i) for i in range(1, MAX_POSITION)] + [f"{MAX_POSITION}+"]
START, END = "START", "END"

# share_by() splits: label -> (column, order)
SPLITS = {
    "Ante": ("ante", None),
    "Money": ("money_bracket", MONEY_BRACKETS),
    "Jokers held": ("jokers_held", JOKERS_HELD),
    "Next blind": ("next_blind", NEXT_BLIND),
    "Position in visit": ("position_bin", POSITIONS),
}
VISIT = ["episode", "round"]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def family(action_name: str) -> str:
    """The kind of decision, ignoring the slot (buy_card_1 -> Buy card)."""
    if action_name == "buy_voucher":
        return "Buy voucher"
    if action_name == "reroll":
        return "Reroll"
    if action_name == "skip":
        return "Leave shop"
    head, _, slot = action_name.rpartition("_")
    if slot.isdigit() and head in _PREFIXES:
        return _PREFIXES[head]
    return action_name


def _bracket(dollars: float) -> str:
    return MONEY_BRACKETS[int(min(max(dollars, 0) // 5, len(MONEY_BRACKETS) - 1))]


def steps_with_context(run: Run) -> pd.DataFrame:
    """Shop steps with ``family``, ``position`` (1-based within the visit),
    ``visit_len``, ``money_bracket``, ``jokers_held`` and ``next_blind``."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame()
    s = run.steps.sort_values(["episode", "step"]).reset_index(drop=True)
    s["family"] = s["action_name"].map(family)
    s["position"] = s.groupby(VISIT).cumcount() + 1
    s["visit_len"] = s.groupby(VISIT)["step"].transform("size")
    s["position_bin"] = pd.Categorical(
        [POSITIONS[min(p, MAX_POSITION) - 1] for p in s["position"]], POSITIONS, ordered=True
    )
    s["money_bracket"] = pd.Categorical(
        s["dollars_before"].map(_bracket), MONEY_BRACKETS, ordered=True
    )
    if "jokers_before" in s:
        held = s["jokers_before"].map(lambda v: len(v) if isinstance(v, list) else 0)
        s["jokers_held"] = pd.Categorical(
            [JOKERS_HELD[min(n, 5)] for n in held], JOKERS_HELD, ordered=True
        )
    if "blind_on_deck" in s:
        s["next_blind"] = pd.Categorical(s["blind_on_deck"], NEXT_BLIND, ordered=True)
    return s


def _visits(s: pd.DataFrame) -> pd.DataFrame:
    """One row per visit: its ordered actions and ante."""
    return s.groupby(VISIT, as_index=False).agg(
        ante=("ante", "first"),
        actions=("action_name", list),
        families=("family", list),
        decisions=("step", "size"),
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def family_share(runs: list[Run]) -> pd.DataFrame:
    """Share of shop decisions per action family, per run (works without steps)."""
    freq = metrics.action_frequency(runs)
    if freq.empty:
        return freq
    freq = freq.assign(family=freq["action"].map(family))
    out = freq.groupby(["run", "family"], as_index=False)["count"].sum()
    out["share"] = out["count"] / out.groupby("run")["count"].transform("sum").replace(0, np.nan)
    order = {f: i for i, f in enumerate(FAMILIES)}
    out = out.sort_values(
        ["run", "family"], key=lambda c: c.map(order) if c.name == "family" else c
    )
    return out.reset_index(drop=True)


def decisions_per_visit(runs: list[Run]) -> pd.DataFrame:
    """One row per shop visit: run, episode, round, ante, decisions."""
    frames = []
    for r in runs:
        s = steps_with_context(r)
        if s.empty:
            continue
        v = _visits(s)[[*VISIT, "ante", "decisions"]]
        frames.append(v.assign(run=r.name))
    return _concat(frames)


def visit_summary(runs: list[Run]) -> pd.DataFrame:
    """Per run: visits, decisions per visit, share of visits that leave at once
    or include a buy / sell / reroll, and the most common first action."""
    rows = []
    for r in runs:
        s = steps_with_context(r)
        if s.empty:
            continue
        v = _visits(s)
        firsts = v["actions"].str[0].value_counts()

        def has(fams, prefix):
            return any(f.startswith(prefix) for f in fams)

        rows.append(
            {
                "run": r.name,
                "visits": len(v),
                "decisions_per_visit": v["decisions"].mean(),
                "leave_at_once": (v["actions"].str[0] == "skip").mean(),
                "with_buy": v["families"].map(lambda f: has(f, "Buy")).mean(),
                "with_sell": v["families"].map(lambda f: has(f, "Sell")).mean(),
                "with_reroll": v["families"].map(lambda f: "Reroll" in f).mean(),
                "top_first_action": firsts.index[0],
                "top_first_share": firsts.iloc[0] / len(v),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def share_by(run: Run, by: str) -> pd.DataFrame:
    """Action share within each value of ``by`` (one of ``SPLITS``' labels).
    Rows: action; columns: the split's values in order."""
    s = steps_with_context(run)
    col, order = SPLITS[by]
    if s.empty or col not in s:
        return pd.DataFrame()
    table = pd.crosstab(s["action_name"], s[col], normalize="columns")
    if order is not None:
        table = table.reindex(columns=[c for c in order if c in table.columns])
    rows = [n for n in metrics.action_names(run) if n in table.index]
    return table.reindex(rows)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def transitions(run: Run) -> pd.DataFrame:
    """Long frame of from -> to action counts and row-normalised rates.

    ``START`` is entering the shop.  Leaving the shop (skip) ends a visit, so
    skip has no outgoing row; ``END`` only appears for visits that stop on
    something else (the log ended)."""
    s = steps_with_context(run)
    if s.empty:
        return pd.DataFrame()
    prev = s.groupby(VISIT)["action_name"].shift(1).fillna(START)
    nxt = s.groupby(VISIT)["action_name"].shift(-1)
    pairs = pd.concat(
        [
            pd.DataFrame({"from": prev, "to": s["action_name"]}),
            pd.DataFrame({"from": s["action_name"], "to": END})[
                nxt.isna() & (s["action_name"] != "skip")
            ],
        ]
    )
    out = pairs.value_counts().rename("count").reset_index()
    out["rate"] = out["count"] / out.groupby("from")["count"].transform("sum")
    return out


def transition_matrix(run: Run) -> pd.DataFrame:
    """``transitions`` as a from x to table of rates, in action order."""
    t = transitions(run)
    if t.empty:
        return t
    m = t.pivot(index="from", columns="to", values="rate").fillna(0.0)
    names = metrics.action_names(run)
    rows = [START, *names]
    cols = [*names, END]
    return m.reindex(index=[r for r in rows if r in m.index], columns=[c for c in cols if c in m])


def top_sequences(run: Run, n: int = 15) -> pd.DataFrame:
    """The most common whole-visit sequences: ``sequence`` (keys joined with
    " → "), ``length``, ``count``, ``share`` of visits."""
    s = steps_with_context(run)
    if s.empty:
        return pd.DataFrame()
    seqs = _visits(s)["actions"].map(tuple)
    counts = seqs.value_counts()
    out = pd.DataFrame(
        {
            "sequence": [" → ".join(t) for t in counts.index],
            "length": [len(t) for t in counts.index],
            "count": counts.to_numpy(),
            "share": counts.to_numpy() / len(seqs),
        }
    )
    return (
        out.sort_values(["count", "length"], ascending=[False, True]).head(n).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Over training / vs outcome
# ---------------------------------------------------------------------------


def family_over_training(runs: list[Run], bins: int = 10) -> pd.DataFrame:
    """Family share of decisions within equal-width episode bins, per run.
    ``episode`` is the first episode of the bin."""
    frames = []
    for r in runs:
        s = steps_with_context(r)
        if s.empty:
            continue
        lo, hi = s["episode"].min(), s["episode"].max()
        width = max(1, int(np.ceil((hi - lo + 1) / bins)))
        s["episode_bin"] = lo + (s["episode"] - lo) // width * width
        t = pd.crosstab(s["episode_bin"], s["family"], normalize="index")
        t = t.reindex(columns=[f for f in FAMILIES if f in t.columns])
        long = t.rename_axis("episode").reset_index().melt("episode", var_name="family")
        frames.append(long.rename(columns={"value": "share"}).assign(run=r.name))
    return _concat(frames)


def action_outcome_correlations(runs: list[Run]) -> pd.DataFrame:
    """Spearman correlation of each family's count *per shop visit* (per
    episode) with ante reached, per run.  Per visit, so longer runs having more
    shops doesn't inflate every count."""
    rows = []
    for r in runs:
        s = steps_with_context(r)
        if s.empty:
            continue
        visits = s.groupby("episode")["round"].nunique()
        per_visit = pd.crosstab(s["episode"], s["family"]).div(visits, axis=0)
        ante = r.episodes.set_index("episode")["ante"]
        df = per_visit.join(ante, how="inner")
        for fam in [f for f in FAMILIES if f in per_visit.columns]:
            g = df[[fam, "ante"]].dropna()
            if len(g) < 3 or g[fam].nunique() < 2 or g["ante"].nunique() < 2:
                continue
            rho = g.corr(method="spearman").iloc[0, 1]
            if np.isnan(rho):
                continue
            rows.append({"run": r.name, "family": fam, "spearman_rho": rho, "episodes": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Availability (log_version 3)
# ---------------------------------------------------------------------------


def _with_legal(run: Run) -> pd.DataFrame:
    """Steps that carry a ``legal`` list, with ``n_legal``; empty for old logs."""
    if run.steps is None or "legal" not in run.steps:
        return pd.DataFrame()
    s = run.steps[run.steps["legal"].map(lambda v: isinstance(v, list) and len(v) > 0)]
    return s.assign(n_legal=s["legal"].map(len))


def availability(runs: list[Run]) -> pd.DataFrame:
    """Per action: how often it was legal, how often it was chosen when legal,
    and what a uniform-random agent would choose (mean of 1/n_legal over the
    decisions where it was legal).  ``preference`` = chosen / uniform, so 1 is
    random-like, above 1 favoured, below 1 avoided."""
    frames = []
    for r in runs:
        s = _with_legal(r)
        if s.empty:
            continue
        names = metrics.action_names(r)
        ex = s[["action", "legal", "n_legal"]].explode("legal")
        ex["legal"] = ex["legal"].astype(int)
        ex["chosen"] = ex["legal"] == ex["action"]
        ex["uniform"] = 1.0 / ex["n_legal"]
        g = ex.groupby("legal").agg(
            legal_count=("chosen", "size"),
            chosen=("chosen", "sum"),
            uniform_expected=("uniform", "mean"),
        )
        g = g.reindex(range(len(names)), fill_value=0)
        out = pd.DataFrame(
            {
                "run": r.name,
                "action": names,
                "decisions": len(s),
                "legal_count": g["legal_count"].to_numpy(),
                "chosen": g["chosen"].to_numpy(),
                "legal_rate": g["legal_count"].to_numpy() / len(s),
            }
        )
        legal = g["legal_count"].replace(0, np.nan).to_numpy()
        out["chosen_when_legal"] = g["chosen"].to_numpy() / legal
        out["uniform_expected"] = g["uniform_expected"].where(g["legal_count"] > 0).to_numpy()
        out["preference"] = out["chosen_when_legal"] / out["uniform_expected"]
        frames.append(out)
    return _concat(frames)


def legal_counts(runs: list[Run]) -> pd.DataFrame:
    """Distribution of the number of legal actions per decision, per run."""
    frames = []
    for r in runs:
        s = _with_legal(r)
        if s.empty:
            continue
        c = s["n_legal"].value_counts().sort_index()
        frames.append(
            pd.DataFrame(
                {
                    "run": r.name,
                    "n_legal": c.index,
                    "count": c.to_numpy(),
                    "fraction": c.to_numpy() / c.sum(),
                }
            )
        )
    return _concat(frames)
