"""Per shop item (each voucher, Tarot, Planet, Spectral, playing card, pack and joker,
by key): how often the agent bought it when it could, skipped it, and sold or used it.

Offers and legality are ``purchases.offers`` (by default a voucher counts once per ante
and a pack once per shop visit); sells and uses are the steps.jsonl ``sell_*`` and
``use_cons*`` decisions on that key.  Buy rate is bought ÷ legal per run, then mean ±
SD over the group's runs, as ``purchases.buy_rates``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from balatro_analysis import catalog, purchases
from balatro_analysis.loader import Run, memo

# The sub-tabs of the Shop items tab: label -> centers.json sets (or a kind)
SECTIONS = {
    "Vouchers": ("Voucher",),
    "Tarots": ("Tarot",),
    "Planets": ("Planet",),
    "Spectrals": ("Spectral",),
    "Playing cards": ("Default", "Enhanced"),
    "Packs": ("Booster",),
    "Jokers": ("Joker",),
}
ANTE_BANDS = ((1, 2, "1-2"), (3, 5, "3-5"), (6, 8, "6-8"), (9, None, "9+"))
COLUMNS = [
    "part", "group", "key", "set", "offered", "legal", "bought", "rate", "sd", "runs",
    "skipped", "sold", "used", "sell_rate", "use_rate", "first_ante", "win_rate_bought",
    "mean_cost",
]  # fmt: skip


def item_set(kind: str, key: str) -> str:
    """The item's centers.json set (Joker, Tarot, Planet, Spectral, Default,
    Enhanced), or Voucher / Booster for those kinds."""
    if kind == "voucher":
        return "Voucher"
    if kind == "pack":
        return "Booster"
    return catalog.centers().get(key, {}).get("set") or "?"


def ante_band(ante) -> str:
    if ante is None or (isinstance(ante, float) and np.isnan(ante)):
        return "?"
    for lo, hi, label in ANTE_BANDS:
        if ante >= lo and (hi is None or ante <= hi):
            return label
    return "?"


def _parts(run: Run, by: str | None, episode: pd.Series, ante: pd.Series) -> pd.Series:
    if by is None:
        return pd.Series("all", index=episode.index)
    if by == "ante":
        return ante.map(ante_band)
    if by in run.episodes:
        of = dict(zip(run.episodes["episode"], run.episodes[by], strict=True))
        return episode.map(lambda e: of.get(e, "?"))
    return pd.Series("?", index=episode.index)


def sold_used(run: Run) -> pd.DataFrame:
    """The run's shop sells and uses: ``episode``, ``ante``, ``key``, ``what``
    (sold / used).  Memoised on the run."""

    def build() -> pd.DataFrame:
        s = run.steps
        cols = ["episode", "ante", "key", "what"]
        if s is None or s.empty or "item_key" not in s:
            return pd.DataFrame(columns=cols)
        name = s["action_name"].astype(str)
        what = np.where(
            name.str.startswith("sell_"), "sold", np.where(name.str.startswith("use_"), "used", "")
        )
        keep = (what != "") & s["item_key"].notna()
        out = pd.DataFrame(
            {
                "episode": s.loc[keep, "episode"],
                "ante": s.loc[keep, "ante"] if "ante" in s else np.nan,
                "key": s.loc[keep, "item_key"],
                "what": what[keep.to_numpy()],
            }
        )
        return out.reset_index(drop=True)

    return memo(run, "items:sold_used", build)


def run_items(run: Run, by: str | None = None, per_offer: bool = True) -> pd.DataFrame:
    """One run's per-item counts: ``part``, ``key``, ``set``, ``offered``, ``legal``,
    ``bought``, ``sold``, ``used``, ``cost_sum``, ``cost_n`` and, per episode it was
    bought in, ``bought_eps`` / ``won_eps`` / ``first_antes`` (a list)."""

    def build() -> pd.DataFrame:
        o = purchases.offers(run, per_offer)
        cols = [
            "part", "key", "set", "offered", "legal", "bought", "sold", "used",
            "cost_sum", "cost_n", "bought_eps", "won_eps", "first_antes",
        ]  # fmt: skip
        if o.empty:
            return pd.DataFrame(columns=cols)
        o = o.assign(
            part=_parts(run, by, o["episode"], o["ante"]),
            set=[item_set(k, key) for k, key in zip(o["kind"], o["key"], strict=True)],
            legal=o["legal"].astype(bool),
            bought=o["bought"].astype(bool),
            cost=pd.to_numeric(o["cost"], errors="coerce"),
        )
        g = o.groupby(["part", "key", "set"], dropna=False)
        out = pd.DataFrame(
            {
                "offered": g.size(),
                "legal": g["legal"].sum(),
                "bought": g["bought"].sum(),
                "cost_sum": g["cost"].sum(),
                "cost_n": g["cost"].count(),
            }
        )
        won = (
            dict(zip(run.episodes["episode"], run.episodes["won"].astype(bool), strict=True))
            if "won" in run.episodes
            else {}
        )
        b = o[o["bought"]]
        if len(b):
            per_ep = b.groupby(["part", "key", "set", "episode"], dropna=False)["ante"].min()
            per_ep = per_ep.reset_index()
            per_ep["won"] = per_ep["episode"].map(lambda e: won.get(e, False))
            eg = per_ep.groupby(["part", "key", "set"], dropna=False)
            out["bought_eps"] = eg.size()
            out["won_eps"] = eg["won"].sum()
            out["first_antes"] = eg["ante"].agg(lambda a: [x for x in a if pd.notna(x)])
        su = sold_used(run)
        if len(su):
            su = su.assign(part=_parts(run, by, su["episode"], su["ante"]))
            counts = su.groupby(["part", "key", "what"]).size().unstack(fill_value=0)
            counts = counts.reindex(columns=["sold", "used"], fill_value=0)
            joined = out.reset_index().merge(counts.reset_index(), on=["part", "key"], how="left")
            out = joined.set_index(["part", "key", "set"])
        out = out.reset_index()
        for c in ("sold", "used", "bought_eps", "won_eps"):
            out[c] = out[c].fillna(0).astype(int) if c in out else 0
        if "first_antes" not in out:
            out["first_antes"] = [[] for _ in range(len(out))]
        out["first_antes"] = out["first_antes"].map(lambda v: v if isinstance(v, list) else [])
        return out[cols]

    return memo(run, f"items:{by}:{per_offer}", build)


def item_rates(
    groups: dict[str, list[Run]], by: str | None = None, per_offer: bool = True
) -> pd.DataFrame:
    """Per ``part``, ``group`` and item ``key``: offers ``offered`` / ``legal`` /
    ``bought`` pooled over the group's runs; ``rate`` and ``sd``, the mean and SD
    (ddof 1) over runs with a legal offer of bought ÷ legal; ``runs`` with a rate;
    ``skipped`` = legal - bought; ``sold`` / ``used`` (shop decisions) and
    ``sell_rate`` / ``use_rate`` per shop buy; ``first_ante``, the median ante it
    was first bought in an episode; ``win_rate_bought``, the win rate of episodes
    that bought it; ``mean_cost`` offered."""
    rows = []
    for g, runs in groups.items():
        per_run = [run_items(r, by, per_offer) for r in runs]
        per_run = [c.assign(run=i) for i, c in enumerate(per_run) if len(c)]
        if not per_run:
            continue
        both = pd.concat(per_run, ignore_index=True)
        for (part, key, set_), cell in both.groupby(["part", "key", "set"], sort=True):
            ok = cell[cell["legal"] > 0]
            rates = (ok["bought"] / ok["legal"]).to_numpy(dtype=float)
            legal, bought = int(cell["legal"].sum()), int(cell["bought"].sum())
            sold, used = int(cell["sold"].sum()), int(cell["used"].sum())
            antes = [a for v in cell["first_antes"] for a in v]
            eps = int(cell["bought_eps"].sum())
            cost_n = cell["cost_n"].sum()
            rows.append(
                {
                    "part": part,
                    "group": g,
                    "key": key,
                    "set": set_,
                    "offered": int(cell["offered"].sum()),
                    "legal": legal,
                    "bought": bought,
                    "rate": float(np.mean(rates)) if len(rates) else np.nan,
                    "sd": (float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0)
                    if len(rates)
                    else np.nan,
                    "runs": len(rates),
                    "skipped": legal - bought,
                    "sold": sold,
                    "used": used,
                    "sell_rate": sold / bought if bought else np.nan,
                    "use_rate": used / bought if bought else np.nan,
                    "first_ante": float(np.median(antes)) if antes else np.nan,
                    "win_rate_bought": cell["won_eps"].sum() / eps if eps else np.nan,
                    "mean_cost": cell["cost_sum"].sum() / cost_n if cost_n else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def section(rates: pd.DataFrame, name: str) -> pd.DataFrame:
    """The rows of ``item_rates`` for one Shop items sub-tab (see ``SECTIONS``)."""
    return rates[rates["set"].isin(SECTIONS[name])]


def wide_table(rates: pd.DataFrame, part, min_legal: int = 0) -> pd.DataFrame:
    """One part of ``item_rates`` for display: a row per item with ``min_legal``
    legal offers pooled in some group, most legal offers first; with several groups,
    the group is a column and items repeat per group."""
    sub = rates[rates["part"] == part]
    if sub.empty:
        return sub
    shown = sub.groupby("key")["legal"].max() >= min_legal
    sub = sub[sub["key"].isin(shown[shown].index)]
    order = sub.groupby("key")["legal"].sum().sort_values(ascending=False).index
    sub = sub.assign(_o=sub["key"].map({k: i for i, k in enumerate(order)}))
    groups = list(dict.fromkeys(rates["group"]))
    sub = sub.assign(_g=sub["group"].map({g: i for i, g in enumerate(groups)}))
    sub = sub.sort_values(["_o", "_g"]).drop(columns=["_o", "_g", "part", "set"])
    if len(groups) == 1:
        sub = sub.drop(columns="group")
    return sub.reset_index(drop=True)
