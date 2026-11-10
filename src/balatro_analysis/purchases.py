"""How often the shop agent buys each kind of item it is offered: jokers by rarity,
edition and sticker, Negative jokers with the joker slots full or free, consumables,
playing cards, packs and vouchers.

A port of balatro-shop-rl's ``scripts/special_buys.py``, kept to give the same numbers:

- a **listing** is a shops.jsonl row (entering a shop, or a reroll); it is up from its
  ``step`` until the episode's next listing's, and the steps.jsonl decisions in that
  window are its decisions (returning from a pack adds no listing);
- an item is **legal** if at one of those decisions its key is in that step's
  ``shop[kind]`` at slot *i* and buying slot *i* is legal (a pricier editioned joker
  often becomes affordable only after a sell).  Items are matched by key, not slot,
  as the script does: two copies of a key in one listing share legal and bought;
- **bought** if one of those decisions is a ``buy_<kind>`` of that key and edition;
- **slots full**: no free joker slot at the window's first decision.

Rate = bought ÷ legal per run, then mean ± SD over the group's runs (seeds), with the
legal listings pooled.  Editions, stickers, slots and the shop per decision need
log_version 17; older logs only know what was legal at each listing's first decision.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from balatro_analysis import catalog
from balatro_analysis.loader import Run, memo

# shop action ids that buy each kind of item, by slot (balatro-shop-rl shop_rl.actions)
BUY_ACTIONS = {"card": (0, 1), "pack": (2, 3), "voucher": (4,)}
RARITY = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Legendary"}
PACK_RE = re.compile(r"^p_([a-z]+)_(normal|jumbo|mega)")
MIN_LEGAL = 20  # the script's --min-legal: rows with fewer legal listings (pooled) are hidden
COUNTS = ("offered", "legal", "bought")


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _value(v):
    return None if _missing(v) else v


def categories(
    kind: str, key: str, ed: str | None, st: list[str] | None, slots_full: bool | None
) -> list[str]:
    """Every row of the table an item counts in."""
    c = catalog.centers().get(key, {})
    set_ = c.get("set", "")
    if kind == "voucher":
        return ["Voucher"]
    if kind == "pack":
        m = PACK_RE.match(key)
        return ["Pack", f"Pack: {m[1]}" if m else "Pack: ?", f"Pack: {m[2]}" if m else "Pack: ?"]
    if set_ == "Joker":
        out = [
            "Joker",
            f"Joker: {RARITY.get(c.get('rarity'), '?')}",
            f"Joker: {ed or 'no edition'}",
        ]
        if ed:
            out.append("Joker: any edition")
        if ed == "negative" and slots_full is not None:
            out.append(f"Joker: negative, slots {'full' if slots_full else 'free'}")
        if ed is None and slots_full is not None:
            out.append(f"Joker: no edition, slots {'full' if slots_full else 'free'}")
        for s in st or ["no sticker"]:
            out.append(f"Joker: {s}")
        return out
    if set_ in ("Default", "Enhanced"):
        return ["Playing card", f"Playing card: {ed or 'no edition'}"]
    return [set_ or "?"]


def has_editions(run: Run) -> bool:
    """Whether the run's logs have editions, stickers and the shop per decision
    (log_version 17), without reading them if the config says."""
    version = run.config.get("log_version")
    if version is not None:
        return int(version) >= 17
    return run.steps is not None and "item_edition" in run.steps


def _decisions(steps: pd.DataFrame) -> dict[int, list[dict]]:
    cols = [
        c
        for c in (
            "episode",
            "step",
            "jokers_before",
            "joker_slots",
            "shop",
            "legal_mask",
            "action_name",
            "item_key",
            "item_edition",
        )
        if c in steps
    ]
    out: dict[int, list[dict]] = defaultdict(list)
    for row in steps[cols].sort_values(["episode", "step"], kind="stable").to_dict("records"):
        out[int(row["episode"])].append(row)
    return out


def run_counts(run: Run, by: str | None = None) -> pd.DataFrame:
    """One run's counts: ``part`` (the episode's ``by`` value, or "all"),
    ``category``, ``offered``, ``legal``, ``bought``.  Empty without shops and steps
    logs.  Memoised on the run."""

    def build() -> pd.DataFrame:
        shops, steps = run.shops, run.steps
        if shops is None or steps is None or shops.empty:
            return pd.DataFrame(columns=["part", "category", *COUNTS])
        part_of = (
            dict(zip(run.episodes["episode"], run.episodes[by], strict=True))
            if by and by in run.episodes
            else {}
        )
        decisions = _decisions(steps)
        counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        listings_by_ep: dict[int, list[dict]] = defaultdict(list)
        for shop in shops[["episode", "step", "items"]].to_dict("records"):
            listings_by_ep[int(shop["episode"])].append(shop)
        for ep, listings in listings_by_ep.items():
            part = part_of.get(ep, "?") if by else "all"
            ep_steps = decisions.get(ep, [])
            for n, shop in enumerate(listings):
                end = listings[n + 1]["step"] if n + 1 < len(listings) else float("inf")
                window = [d for d in ep_steps if shop["step"] <= d["step"] < end]
                first = window[0] if window else None
                slots_full = None
                if first is not None and not _missing(first.get("joker_slots")):
                    held = first.get("jokers_before")
                    held = held if not _missing(held) else []
                    slots_full = len(held) >= first["joker_slots"]
                items = shop["items"] if isinstance(shop["items"], list | np.ndarray) else []
                for item in items:
                    if _value(item.get("hidden")):
                        continue
                    kind, key, ed = item["kind"], item["key"], _value(item.get("ed"))
                    acts = BUY_ACTIONS[kind]
                    legal = bought = False
                    for d in window:
                        offer = _value(d.get("shop"))
                        on_offer = _value(offer.get(kind)) if isinstance(offer, dict) else None
                        if on_offer is None:  # before log_version 17: legal at the listing only
                            on_offer = (
                                [i["key"] for i in items if i["kind"] == kind] if d is first else []
                            )
                        mask = int(_value(d.get("legal_mask")) or 0)
                        legal |= any(
                            i < len(acts) and mask >> acts[i] & 1
                            for i, k in enumerate(on_offer)
                            if k == key
                        )
                        if (
                            str(d["action_name"]).startswith(f"buy_{kind}")
                            and _value(d.get("item_key")) == key
                            and _value(d.get("item_edition")) == ed
                        ):
                            bought = True
                    st = _value(item.get("st"))
                    for cat in categories(
                        kind, key, ed, list(st) if st is not None else None, slots_full
                    ):
                        c = counts[(part, cat)]
                        c["offered"] += 1
                        c["legal"] += legal
                        c["bought"] += bought
        rows = [
            {"part": p, "category": cat, **{k: c[k] for k in COUNTS}}
            for (p, cat), c in counts.items()
        ]
        return pd.DataFrame(rows, columns=["part", "category", *COUNTS])

    return memo(run, f"purchases:{by}", build)


def buy_rates(groups: dict[str, list[Run]], by: str | None = None) -> pd.DataFrame:
    """Per ``part``, ``group`` and ``category``: ``rate`` (mean over the group's runs
    with a legal listing of bought ÷ legal), ``sd`` (ddof 1; 0 for one run), ``runs``
    (with a rate), and ``legal`` / ``bought`` / ``offered`` pooled over the runs."""
    rows = []
    for g, runs in groups.items():
        per_run = [run_counts(r, by) for r in runs]
        per_run = [c for c in per_run if len(c)]
        if not per_run:
            continue
        both = pd.concat([c.assign(run=i) for i, c in enumerate(per_run)], ignore_index=True)
        for (part, cat), cell in both.groupby(["part", "category"], sort=True):
            ok = cell[cell["legal"] > 0]
            rates = (ok["bought"] / ok["legal"]).to_numpy(dtype=float)
            rows.append(
                {
                    "part": part,
                    "group": g,
                    "category": cat,
                    "rate": float(np.mean(rates)) if len(rates) else np.nan,
                    "sd": (float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0)
                    if len(rates)
                    else np.nan,
                    "runs": len(rates),
                    "legal": int(cell["legal"].sum()),
                    "bought": int(cell["bought"].sum()),
                    "offered": int(cell["offered"].sum()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["part", "group", "category", "rate", "sd", "runs", "legal", "bought", "offered"],
    )


def rate_table(rates: pd.DataFrame, part, min_legal: int = MIN_LEGAL) -> pd.DataFrame:
    """One part of ``buy_rates`` as the script prints it: a row per category (those
    with ``min_legal`` legal listings pooled in some group), a column per group of
    "rate ± sd (legal)" ("- (legal)" without a legal listing)."""
    sub = rates[rates["part"] == part]
    if sub.empty:
        return pd.DataFrame()
    shown = sub.groupby("category")["legal"].max() >= min_legal
    sub = sub[sub["category"].isin(shown[shown].index)]

    def cell(r) -> str:
        if r["runs"] == 0:
            return f"- ({r['legal']})"
        return f"{r['rate']:.2f} ± {r['sd']:.2f} ({r['legal']})"

    wide = (
        sub.assign(cell=sub.apply(cell, axis=1))
        .pivot(index="category", columns="group", values="cell")
        .sort_index()
    )
    order = list(dict.fromkeys(rates["group"]))
    wide = wide[[g for g in order if g in wide.columns]]
    # a group with this part but none of a category: "- (0)", as the script
    return wide.fillna("- (0)").reset_index().rename(columns={"category": "item"})
