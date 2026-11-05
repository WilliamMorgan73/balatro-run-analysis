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
from scipy import stats as sps

from balatro_analysis import loader, metrics, stats
from balatro_analysis.loader import Run, memo

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


def _binned(codes, labels: list[str]) -> pd.Categorical:
    """Ordered categorical of ``labels[code]``, codes clipped into range."""
    codes = np.clip(np.asarray(codes, dtype=float), 0, len(labels) - 1).astype(int)
    return pd.Categorical.from_codes(codes, labels, ordered=True)


def steps_with_context(run: Run) -> pd.DataFrame:
    """Shop steps with ``family``, ``position`` (1-based within the visit),
    ``visit_len``, ``money_bracket``, ``jokers_held`` and ``next_blind``."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame()
    return memo(run, "steps_with_context", lambda: _steps_with_context(run))


def _steps_with_context(run: Run) -> pd.DataFrame:
    s = run.steps.sort_values(["episode", "step"]).reset_index(drop=True)
    names = s["action_name"]
    s["family"] = names.map({a: family(a) for a in names.unique()})
    s["position"] = s.groupby(VISIT).cumcount() + 1
    s["visit_len"] = s.groupby(VISIT)["step"].transform("size")
    s["position_bin"] = _binned(s["position"] - 1, POSITIONS)
    s["money_bracket"] = _binned(s["dollars_before"] // 5, MONEY_BRACKETS)
    if "jokers_before" in s:
        held = s["jokers_before"].map(lambda v: len(v) if isinstance(v, list) else 0)
        s["jokers_held"] = _binned(held, JOKERS_HELD)
    if "blind_on_deck" in s:
        s["next_blind"] = pd.Categorical(s["blind_on_deck"], NEXT_BLIND, ordered=True)
    return s


def _visits(s: pd.DataFrame) -> pd.DataFrame:
    """One row per visit: its ante and number of decisions."""
    return s.groupby(VISIT, as_index=False).agg(
        ante=("ante", "first"),
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
        v = _visits(s)
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
        fam = s["family"]
        v = (
            s.assign(
                buy=fam.str.startswith("Buy"),
                sell=fam.str.startswith("Sell"),
                reroll=fam == "Reroll",
            )
            .groupby(VISIT)
            .agg(
                first=("action_name", "first"),
                decisions=("step", "size"),
                buy=("buy", "any"),
                sell=("sell", "any"),
                reroll=("reroll", "any"),
            )
        )
        firsts = v["first"].value_counts()
        rows.append(
            {
                "run": r.name,
                "visits": len(v),
                "decisions_per_visit": v["decisions"].mean(),
                "leave_at_once": (v["first"] == "skip").mean(),
                "with_buy": v["buy"].mean(),
                "with_sell": v["sell"].mean(),
                "with_reroll": v["reroll"].mean(),
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
    # visits are contiguous after the sort in steps_with_context: one tuple per run of rows
    starts = np.flatnonzero(s[VISIT].ne(s[VISIT].shift()).any(axis=1))
    names = s["action_name"].tolist()
    seqs = pd.Series(
        [tuple(names[a:b]) for a, b in zip(starts, [*starts[1:], len(s)], strict=True)]
    )
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
            rho, p = stats.spearman(g[fam], g["ante"])
            if np.isnan(rho):
                continue
            rows.append(
                {
                    "run": r.name,
                    "family": fam,
                    "spearman_rho": rho,
                    "p_value": p,
                    "episodes": len(g),
                }
            )
    return stats.with_q_values(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Availability (log_version 3)
# ---------------------------------------------------------------------------


def _legal_mask(steps: pd.DataFrame) -> pd.Series | None:
    """Per step, a bitmask of the legal action ids (0: not logged); None for old logs.
    Loaded runs carry ``legal_mask``; a ``legal`` list column (built by hand) works too."""
    if "legal_mask" in steps:
        return steps["legal_mask"]
    if "legal" in steps:
        return pd.Series(loader.mask_from_lists(steps["legal"]), index=steps.index)
    return None


def _with_legal(run: Run) -> pd.DataFrame:
    """Steps that carry a legal-action list, with ``legal_mask`` and ``n_legal``;
    empty for old logs."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame()
    mask = _legal_mask(run.steps)
    if mask is None:
        return pd.DataFrame()
    has = (mask != 0).to_numpy()
    m = mask[has].to_numpy().astype(np.int64)
    return run.steps[has].assign(legal_mask=m, n_legal=np.bitwise_count(m).astype(np.int64))


def availability(runs: list[Run]) -> pd.DataFrame:
    """Per action: how often it was legal, how often it was chosen when legal,
    and what a uniform-random agent would choose (mean of 1/n_legal over the
    decisions where it was legal).  ``preference`` = chosen / uniform, so 1 is
    random-like, above 1 favoured, below 1 avoided; ``preference_low/high`` is a
    95% interval (Wilson on the chosen rate, divided by the uniform rate).
    ``p_value`` tests "chosen as often as a uniform-random agent would": under that
    null the count chosen has mean sum(1/n_legal) and variance sum(p(1-p)) over
    the decisions where the action was legal (normal approximation); ``q_value``
    is BH-adjusted within each run."""
    frames = []
    for r in runs:
        s = _with_legal(r)
        if s.empty:
            continue
        names = metrics.action_names(r)
        ids = np.arange(len(names), dtype=np.int64)
        # decisions x actions: was the action legal / chosen there
        legal = ((s["legal_mask"].to_numpy()[:, None] >> ids) & 1).astype(bool)
        chosen = legal & (s["action"].to_numpy()[:, None] == ids)
        uniform = 1.0 / s["n_legal"].to_numpy(dtype=float)
        legal_count = legal.sum(axis=0)
        chosen_count = chosen.sum(axis=0)
        null_mean = uniform @ legal
        null_var = (uniform * (1 - uniform)) @ legal
        out = pd.DataFrame(
            {
                "run": r.name,
                "action": names,
                "decisions": len(s),
                "legal_count": legal_count,
                "chosen": chosen_count,
                "legal_rate": legal_count / len(s),
            }
        )
        has = legal_count > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out["chosen_when_legal"] = np.where(has, chosen_count / legal_count, np.nan)
            out["uniform_expected"] = np.where(has, null_mean / legal_count, np.nan)
        out["preference"] = out["chosen_when_legal"] / out["uniform_expected"]
        ci = [
            stats.wilson_ci(int(c), int(n)) if n else (np.nan, np.nan)
            for c, n in zip(chosen_count, legal_count, strict=True)
        ]
        out["preference_low"] = [lo for lo, _ in ci] / out["uniform_expected"]
        out["preference_high"] = [hi for _, hi in ci] / out["uniform_expected"]
        sd = np.sqrt(null_var)
        z = (chosen_count - null_mean) / np.where(sd > 0, sd, np.nan)
        out["p_value"] = 2 * sps.norm.sf(np.abs(z))
        frames.append(out)
    return stats.with_q_values(_concat(frames))


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
