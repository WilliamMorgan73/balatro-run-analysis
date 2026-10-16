"""Economy metrics: where the money comes from, where it goes, and whether it pays off.

Everything works from ``steps.jsonl`` (money before/after every shop decision).
Newer runs add exact cash-out income (``rounds.jsonl`` ``earn_*`` columns) and
what each shop offered (``shops.jsonl``); functions needing those return an
empty frame for older runs.

Like ``metrics``, these are pure pandas, key-only, and return long frames with
a ``run`` column.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from balatro_analysis import catalog, stats
from balatro_analysis.loader import Run

DEFAULT_INTEREST_CAP = 25
BUY_CATEGORIES = [
    "Jokers",
    "Tarot",
    "Planet",
    "Spectral",
    "Playing cards",
    "Packs",
    "Vouchers",
    "Rerolls",
]
INCOME_CATEGORIES = ["Selling jokers", "Selling consumables", "Using consumables"]
EARNINGS = {
    "earn_blind": "Blind reward",
    "earn_interest": "Interest",
    "earn_hands": "Unused hands",
    "earn_discards": "Unused discards",
    "earn_jokers": "Jokers",
    "earn_rent": "Rent",
    "earn_tags": "Tags",  # log_version 5: money tags pay at cash-out (Investment)
}


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _set_category(key) -> str:
    entry = catalog.centers().get(key) if isinstance(key, str) else None
    s = entry["set"] if entry else ""
    if s == "Joker":
        return "Jokers"
    if s in ("Tarot", "Planet", "Spectral"):
        return s
    return "Playing cards"


def category(action_name: str, item_key) -> str | None:
    """What a shop decision spends on (or earns from); None for leaving the shop."""
    if action_name == "reroll":
        return "Rerolls"
    if action_name.startswith("buy_pack"):
        return "Packs"
    if action_name == "buy_voucher":
        return "Vouchers"
    if action_name.startswith("buy_card"):
        return _set_category(item_key)
    if action_name.startswith("sell_joker"):
        return "Selling jokers"
    if action_name.startswith("sell_cons"):
        return "Selling consumables"
    if action_name.startswith("use_cons"):
        return "Using consumables"
    return None


def steps_with_money(run: Run) -> pd.DataFrame:
    """Shop steps with ``category``, ``spent`` and ``income`` (both >= 0), and
    ``pack_income``: money gained while a pack bought by this step was open
    (e.g. The Hermit; log_version 5, else 0)."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame()
    s = run.steps.sort_values(["episode", "step"]).copy()
    s["category"] = [category(a, k) for a, k in zip(s["action_name"], s["item_key"], strict=True)]
    delta = s["dollars_after"] - s["dollars_before"]
    s["spent"] = (-delta).clip(lower=0)
    s["income"] = delta.clip(lower=0)
    if "dollars_after_pack" in s:
        s["pack_income"] = (s["dollars_after_pack"] - s["dollars_after"]).clip(lower=0).fillna(0)
    else:
        s["pack_income"] = 0
    return s


def interest_cap(run: Run) -> int:
    """The run's interest cap: logged by newer runs, else the game default."""
    if run.rounds is not None and "interest_cap" in run.rounds:
        caps = run.rounds["interest_cap"].dropna()
        if len(caps):
            return int(caps.mode().iloc[0])
    return DEFAULT_INTEREST_CAP


# ---------------------------------------------------------------------------
# Shop visits
# ---------------------------------------------------------------------------


def shop_visits(runs: list[Run]) -> pd.DataFrame:
    """One row per shop visit (run, episode, round): money in/out, spend, income,
    rerolls, buys, plus the estimated interest the exit balance earns."""
    frames = []
    for r in runs:
        s = steps_with_money(r)
        if s.empty:
            continue
        sold = s["income"].where(s["category"].isin(INCOME_CATEGORIES[:2]), 0)
        s = s.assign(
            sold=sold,
            other_income=s["income"] - sold + s["pack_income"],
            is_reroll=s["category"] == "Rerolls",
            reroll_spend=s["spent"].where(s["category"] == "Rerolls", 0),
            is_buy=(s["spent"] > 0) & (s["category"] != "Rerolls"),
        )
        v = s.groupby(["episode", "round"], as_index=False).agg(
            ante=("ante", "first"),
            entry=("dollars_before", "first"),
            exit=("dollars_after", "last"),
            spent=("spent", "sum"),
            sold=("sold", "sum"),
            other_income=("other_income", "sum"),
            rerolls=("is_reroll", "sum"),
            reroll_spend=("reroll_spend", "sum"),
            buys=("is_buy", "sum"),
            decisions=("step", "size"),
        )
        cap = interest_cap(r)
        v["interest_cap"] = cap
        v["est_interest"] = np.minimum(v["exit"] // 5, cap // 5).clip(lower=0)
        v["at_cap"] = v["exit"] >= cap
        v["parked_above_cap"] = (v["exit"] - cap).clip(lower=0)
        frames.append(v.assign(run=r.name))
    return _concat(frames)


def money_by_round(runs: list[Run]) -> pd.DataFrame:
    """Median and inter-quartile range of money entering/leaving the shop, by round."""
    v = shop_visits(runs)
    if v.empty:
        return v
    rows = []
    for (run, rnd), g in v.groupby(["run", "round"]):
        for which in ("entry", "exit"):
            q = g[which].quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "run": run,
                    "round": rnd,
                    "ante": int(g["ante"].mode().iloc[0]),
                    "when": which,
                    "median": q[0.5],
                    "q25": q[0.25],
                    "q75": q[0.75],
                    "visits": len(g),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spending and income
# ---------------------------------------------------------------------------


def spend_by_category(runs: list[Run], by_ante: bool = False) -> pd.DataFrame:
    """Total and share of shop spending per category (optionally per ante)."""
    frames = []
    for r in runs:
        s = steps_with_money(r)
        if s.empty:
            continue
        s = s[s["category"].isin(BUY_CATEGORIES) & (s["spent"] > 0)]
        keys = ["ante", "category"] if by_ante else ["category"]
        g = s.groupby(keys, as_index=False)["spent"].sum()
        totals = g.groupby("ante")["spent"].transform("sum") if by_ante else g["spent"].sum()
        g["share"] = g["spent"] / totals
        g["per_episode"] = g["spent"] / max(len(r.episodes), 1)
        frames.append(g.assign(run=r.name))
    return _concat(frames)


def income_by_source(runs: list[Run]) -> pd.DataFrame:
    """Shop income (selling, using money-making consumables) per episode, and the
    cash-out components for newer runs."""
    frames = []
    for r in runs:
        n = max(len(r.episodes), 1)
        rows = []
        s = steps_with_money(r)
        if not s.empty:
            for cat, total in s[s["income"] > 0].groupby("category")["income"].sum().items():
                rows.append({"source": cat, "kind": "shop", "per_episode": total / n})
            if pack := s["pack_income"].sum():
                rows.append({"source": "Pack contents", "kind": "shop", "per_episode": pack / n})
        if r.rounds is not None and "earn_total" in r.rounds:
            paid = r.rounds.dropna(subset=["earn_total"])
            for col, label in EARNINGS.items():
                if col not in paid:
                    continue  # older logs
                total = paid[col].sum() * (-1 if col == "earn_rent" else 1)
                if total:
                    rows.append({"source": label, "kind": "cash-out", "per_episode": total / n})
        if rows:
            frames.append(pd.DataFrame(rows).assign(run=r.name))
    return _concat(frames)


def income_breakdown(runs: list[Run]) -> pd.DataFrame:
    """Mean cash-out components per beaten blind, by ante (newer runs only)."""
    frames = []
    for r in runs:
        if r.rounds is None or "earn_total" not in r.rounds:
            continue
        paid = r.rounds.dropna(subset=["earn_total"]).copy()
        if paid.empty:
            continue
        paid["earn_rent"] = -paid["earn_rent"]
        cols = [c for c in EARNINGS if c in paid]
        long = paid.melt(id_vars=["ante"], value_vars=cols, var_name="component")
        g = long.groupby(["ante", "component"], as_index=False)["value"].mean()
        g["component"] = g["component"].map(EARNINGS)
        frames.append(g.assign(run=r.name))
    return _concat(frames)


def interest_efficiency(runs: list[Run]) -> pd.DataFrame:
    """Interest actually earned vs the most the cap allows, by ante (newer runs only)."""
    frames = []
    for r in runs:
        if r.rounds is None or "earn_interest" not in r.rounds:
            continue
        paid = r.rounds.dropna(subset=["earn_interest"]).copy()
        if paid.empty:
            continue
        paid["max_interest"] = paid["interest_cap"].fillna(DEFAULT_INTEREST_CAP) // 5
        g = paid.groupby("ante", as_index=False).agg(
            interest=("earn_interest", "mean"), max_interest=("max_interest", "mean")
        )
        g["efficiency"] = g["interest"] / g["max_interest"]
        frames.append(g.assign(run=r.name))
    return _concat(frames)


# ---------------------------------------------------------------------------
# Rerolls and prices
# ---------------------------------------------------------------------------


def reroll_stats(runs: list[Run]) -> pd.DataFrame:
    """Per ante: mean rerolls and reroll spend per shop visit, share of visits with any."""
    v = shop_visits(runs)
    if v.empty:
        return v
    g = v.groupby(["run", "ante"], as_index=False).agg(
        visits=("rerolls", "size"),
        rerolls_per_visit=("rerolls", "mean"),
        reroll_spend_per_visit=("reroll_spend", "mean"),
        any_reroll=("rerolls", lambda x: (x > 0).mean()),
    )
    return g


def price_paid(runs: list[Run]) -> pd.DataFrame:
    """Every purchase (not rerolls): category, price paid, ante."""
    frames = []
    for r in runs:
        s = steps_with_money(r)
        if s.empty:
            continue
        b = s[s["category"].isin(BUY_CATEGORIES[:-1]) & (s["spent"] > 0)]
        frames.append(b[["episode", "ante", "category", "item_key", "spent"]].assign(run=r.name))
    return _concat(frames).rename(columns={"spent": "price"})


# ---------------------------------------------------------------------------
# Economy vs outcome
# ---------------------------------------------------------------------------

# Totals grow with run length (more shops), so they correlate with ante reached
# almost by construction; the per-visit versions are the fairer comparison.
OUTCOME_METRICS = {
    "mean_exit": "Mean $ leaving shop",
    "mean_entry": "Mean $ entering shop",
    "spent_per_visit": "$ spent per shop visit",
    "rerolls_per_visit": "Rerolls per shop visit",
    "total_spent": "Total $ spent (grows with run length)",
    "joker_share": "Share of spend on jokers",
    "reroll_spend": "$ on rerolls",
    "sold": "$ from selling",
    "at_cap_share": "Share of exits at interest cap",
    "final_dollars": "$ at end of run",
}


def episode_economy(runs: list[Run]) -> pd.DataFrame:
    """Per episode: economy summary joined with how far the run got."""
    frames = []
    visits = shop_visits(runs)
    for r in runs:
        e = r.episodes[["episode", "ante", "won"]].copy()
        if "final_dollars" in r.episodes:
            e["final_dollars"] = r.episodes["final_dollars"]
        v = visits[visits["run"] == r.name] if not visits.empty else visits
        if not v.empty:
            agg = v.groupby("episode").agg(
                mean_exit=("exit", "mean"),
                mean_entry=("entry", "mean"),
                total_spent=("spent", "sum"),
                reroll_spend=("reroll_spend", "sum"),
                sold=("sold", "sum"),
                at_cap_share=("at_cap", "mean"),
                spent_per_visit=("spent", "mean"),
                rerolls_per_visit=("rerolls", "mean"),
            )
            spend = spend_by_category_per_episode(r)
            agg["joker_share"] = spend.get("Jokers", 0) / agg["total_spent"].replace(0, np.nan)
            e = e.merge(agg, left_on="episode", right_index=True, how="left")
        frames.append(e.assign(run=r.name))
    return _concat(frames)


def spend_by_category_per_episode(run: Run) -> pd.DataFrame:
    s = steps_with_money(run)
    if s.empty:
        return pd.DataFrame()
    s = s[s["spent"] > 0]
    return s.pivot_table(
        index="episode", columns="category", values="spent", aggfunc="sum", fill_value=0
    )


def outcome_correlations(econ: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each economy metric with ante reached, per run."""
    rows = []
    for run, g in econ.groupby("run"):
        for col, label in OUTCOME_METRICS.items():
            if col not in g or g[col].notna().sum() < 3 or g[col].nunique() < 2:
                continue
            pair = g[[col, "ante"]].dropna()
            rho, p = stats.spearman(pair[col], pair["ante"])
            if np.isnan(rho):
                continue
            rows.append(
                {
                    "run": run,
                    "metric": label,
                    "spearman_rho": rho,
                    "p_value": p,
                    "episodes": len(pair),
                }
            )
    return stats.with_q_values(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Shop offers (newer runs)
# ---------------------------------------------------------------------------


def offers(run: Run) -> pd.DataFrame:
    """Every offered shop item, with ``affordable`` (at that moment) and ``bought``
    (bought before the shop next changed: the next reroll or the next shop)."""
    if run.shops is None or run.shops.empty:
        return pd.DataFrame()
    shops = run.shops.sort_values(["episode", "step"]).reset_index(drop=True)
    steps = run.steps if run.steps is not None else pd.DataFrame(columns=["episode", "step"])
    buys = steps[steps["action_name"].str.startswith("buy")] if len(steps) else steps
    rows = []
    for ep, eshops in shops.groupby("episode"):
        ebuys = buys[buys["episode"] == ep] if len(buys) else buys
        starts = [*eshops["step"], np.inf]
        for i, (_, offer) in enumerate(eshops.iterrows()):
            window = (
                ebuys[(ebuys["step"] >= starts[i]) & (ebuys["step"] < starts[i + 1])]
                if len(ebuys)
                else ebuys
            )
            remaining = Counter(window["item_key"]) if len(window) else Counter()
            for item in offer["items"]:
                bought = remaining[item["key"]] > 0
                if bought:
                    remaining[item["key"]] -= 1
                rows.append(
                    {
                        "episode": ep,
                        "step": offer["step"],
                        "ante": offer["ante"],
                        "round": offer["round"],
                        "reason": offer["reason"],
                        "dollars": offer["dollars"],
                        "kind": item["kind"],
                        "key": item["key"],
                        "cost": item["cost"],
                        "category": {"voucher": "Vouchers", "pack": "Packs"}.get(
                            item["kind"], _set_category(item["key"])
                        ),
                        "affordable": item["cost"] <= offer["dollars"],
                        "bought": bought,
                    }
                )
    return pd.DataFrame(rows).assign(run=run.name)


def offer_conversion(runs: list[Run]) -> pd.DataFrame:
    """Per category: items offered, affordable, bought; buy rates."""
    o = _concat([offers(r) for r in runs])
    if o.empty:
        return o
    g = o.groupby(["run", "category"], as_index=False).agg(
        offered=("key", "size"),
        affordable=("affordable", "sum"),
        bought=("bought", "sum"),
        mean_price=("cost", "mean"),
    )
    g["affordable_share"] = g["affordable"] / g["offered"]
    g["buy_rate"] = g["bought"] / g["offered"]
    g["buy_rate_when_affordable"] = g["bought"] / g["affordable"].replace(0, np.nan)
    return g


def skipped_jokers(runs: list[Run], top: int = 16) -> pd.DataFrame:
    """Jokers most often affordable but not bought."""
    o = _concat([offers(r) for r in runs])
    if o.empty:
        return o
    j = o[(o["category"] == "Jokers") & o["affordable"]]
    g = j.groupby("key").agg(affordable=("key", "size"), bought=("bought", "sum"))
    g["skipped"] = g["affordable"] - g["bought"]
    return g.sort_values("skipped", ascending=False).head(top).reset_index()


def kpis(runs: list[Run]) -> pd.DataFrame:
    """One summary row per run for the KPI strip."""
    visits = shop_visits(runs)
    rows = []
    for r in runs:
        n = max(len(r.episodes), 1)
        v = visits[visits["run"] == r.name] if not visits.empty else visits
        rows.append(
            {
                "run": r.name,
                "spent_per_episode": r.episodes["dollars_spent"].mean()
                if len(r.episodes)
                else np.nan,
                "sold_per_episode": v["sold"].sum() / n if len(v) else np.nan,
                "reroll_per_episode": v["reroll_spend"].sum() / n if len(v) else np.nan,
                "mean_entry": v["entry"].mean() if len(v) else np.nan,
                "mean_exit": v["exit"].mean() if len(v) else np.nan,
                "exits_at_cap": v["at_cap"].mean() if len(v) else np.nan,
                "final_dollars": r.episodes["final_dollars"].mean()
                if "final_dollars" in r.episodes and len(r.episodes)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
