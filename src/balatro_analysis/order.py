"""The Order's offers by the xMult jokers held: does the agent buy The Order (X3 Mult
for a Straight) more when it already holds xMult jokers?

Part 1 of balatro-shop-rl's ``scripts/order_context.py`` (part 2 needs the model),
from ``offers.jsonl`` (log_version 10, ``--offer-jokers j_order``): one row per
decision and shop slot offering The Order where buying it was legal, so these are
decision-level counts, unlike Shop purchases' listings.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from balatro_analysis import report_stats
from balatro_analysis.loader import Run, memo

ORDER = "j_order"
# jokers whose joker-algebra spec has an x_mult term (shop_rl.jokers.SPECS, as
# order_context.XMULT computes it); copied, as this repo doesn't depend on shop_rl
XMULT = frozenset(
    {
        "j_acrobat", "j_ancient", "j_baron", "j_baseball", "j_blackboard", "j_bloodstone",
        "j_caino", "j_campfire", "j_card_sharp", "j_cavendish", "j_constellation",
        "j_drivers_license", "j_duo", "j_family", "j_flower_pot", "j_glass",
        "j_hit_the_road", "j_hologram", "j_idol", "j_loyalty_card", "j_lucky_cat",
        "j_madness", "j_obelisk", "j_order", "j_photograph", "j_ramen", "j_seeing_double",
        "j_steel_joker", "j_stencil", "j_throwback", "j_tribe", "j_triboulet", "j_trio",
        "j_vampire", "j_yorick",
    }
)  # fmt: skip
HAND_XMULT = frozenset({"j_duo", "j_trio", "j_family", "j_tribe"})
BUCKETS = (
    "all",
    "xmult 0",
    "xmult 1",
    "xmult 2+",
    "hand-type xmult no",
    "hand-type xmult yes",
)


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2+"


def has_order_offers(run: Run) -> bool:
    offers = run.offers if run.has("offers") else None
    return offers is not None and "key" in offers and (offers["key"] == ORDER).any()


def from_logs(run: Run) -> pd.DataFrame:
    """Per bucket of held jokers: legal offers ``n``, buy ``rate`` and mean
    ``adv`` (``q_buy_adv``, NaN without one).  Empty without The Order's offers."""

    def build() -> pd.DataFrame:
        offers = run.offers
        cols = ["bucket", "n", "rate", "adv"]
        if offers is None or "key" not in offers:
            return pd.DataFrame(columns=cols)
        rows = offers[(offers["key"] == ORDER) & offers["buy_legal"].astype(bool)]
        groups: dict[str, list] = defaultdict(list)
        for jokers, bought, adv in zip(
            rows["jokers"], rows["bought"], rows.get("q_buy_adv", [None] * len(rows)), strict=True
        ):
            held = [
                k for k in (jokers if isinstance(jokers, list | np.ndarray) else []) if k != ORDER
            ]
            entry = (bool(bought), adv)
            groups[f"xmult {_bucket(sum(k in XMULT for k in held))}"].append(entry)
            hand = "yes" if any(k in HAND_XMULT for k in held) else "no"
            groups[f"hand-type xmult {hand}"].append(entry)
            groups["all"].append(entry)
        out = []
        for name, entries in groups.items():
            advs = [a for _, a in entries if a is not None and not np.isnan(a)]
            out.append(
                {
                    "bucket": name,
                    "n": len(entries),
                    "rate": float(np.mean([b for b, _ in entries])),
                    "adv": float(np.mean(advs)) if advs else float("nan"),
                }
            )
        return pd.DataFrame(out, columns=cols)

    return memo(run, "order:from_logs", build)


def order_table(runs: list[Run]) -> tuple[pd.DataFrame, dict | None]:
    """Over a group's runs (seeds), as the script: per bucket, offers summed, buy
    rate mean ± SD (ddof 1) over the runs with that bucket, and the nanmean of their
    ``adv``; and the paired rate("xmult 1") - rate("xmult 0") over seeds with both
    (None with fewer than 2)."""
    per = [f.set_index("bucket") for f in (from_logs(r) for r in runs) if len(f)]
    rows = []
    for name in BUCKETS:
        vals = [f.loc[name] for f in per if name in f.index]
        if not vals:
            continue
        rates = [float(v["rate"]) for v in vals]
        advs = np.array([float(v["adv"]) for v in vals])
        rows.append(
            {
                "held": name,
                "offers": int(sum(v["n"] for v in vals)),
                "buy rate": float(np.mean(rates)),
                "SD (seeds)": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                "mean q_buy_adv": float(np.nanmean(advs)) if (~np.isnan(advs)).any() else np.nan,
            }
        )
    both = [
        float(f.at["xmult 1", "rate"] - f.at["xmult 0", "rate"])
        for f in per
        if "xmult 0" in f.index and "xmult 1" in f.index
    ]
    return pd.DataFrame(rows), report_stats.paired(both) if len(both) > 1 else None
