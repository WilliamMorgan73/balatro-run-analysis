"""One switch for how game keys appear: in-game names or exact keys.

``Names(pretty=True)`` shows "Greedy Joker", "Black Deck", "The Hook", "Buy card 1";
``Names(pretty=False)`` shows ``j_greedy_joker``, ``b_black``, ``bl_hook``, ``buy_card_0``.
Metrics stay key-only; everything user-facing goes through a ``Names``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from balatro_analysis import catalog

COLUMNS = {
    "run": "Run",
    "episode": "Episode",
    "episodes": "Episodes",
    "ante": "Ante",
    "won": "Won",
    "return": "Return",
    "shop_steps": "Shop decisions",
    "shops_visited": "Shops visited",
    "dollars_spent": "$ spent",
    "seconds": "Seconds",
    "seed": "Seed",
    "deck": "Deck",
    "stake": "Stake",
    "final_round": "Final round",
    "final_dollars": "Final $",
    "final_jokers": "Final jokers",
    "step": "Step",
    "round": "Round",
    "blind_on_deck": "Next blind",
    "dollars_before": "$ before",
    "dollars_after": "$ after",
    "action": "Action",
    "action_name": "Action",
    "item_key": "Item",
    "item_cost": "Cost",
    "reward": "Reward",
    "done": "Done",
    "jokers_before": "Jokers held",
    "consumables_before": "Consumables held",
    "blind_key": "Blind",
    "target": "Target",
    "chips": "Chips scored",
    "beaten": "Beaten",
    "ratio": "Score / target",
    "blind type": "Blind type",
    "mean_ante": "Mean ante",
    "max_ante": "Max ante",
    "win_rate": "Win rate",
    "joker": "Joker",
    "bought": "Bought",
    "sold": "Sold",
    "count": "Count",
    "share": "Share",
    "fraction": "Fraction",
    "outcome": "Outcome",
    "ci_low": "95% CI low",
    "ci_high": "95% CI high",
    "std": "Std",
    "median": "Median",
    "baseline": "Baseline",
    "baseline_episodes": "Baseline episodes",
    "mean_ante_diff": "Δ mean ante",
    "diff_ci_low": "Δ 95% CI low",
    "diff_ci_high": "Δ 95% CI high",
    "rank_biserial": "Rank-biserial",
    "mann_whitney_p": "Mann-Whitney p",
    "win_rate_diff": "Δ win rate",
    "fisher_p": "Fisher p",
    "paired_episodes": "Paired episodes",
    "paired_mean_diff": "Paired Δ ante",
    "wilcoxon_p": "Wilcoxon p",
    "spearman_rho": "Spearman ρ",
    "p_value": "p",
    "q_value": "q (BH)",
    "significant": "q < 0.05",
}

ACTION_VERBS = {
    "buy_card": "Buy card",
    "buy_pack": "Buy pack",
    "sell_joker": "Sell joker",
    "sell_cons": "Sell consumable",
    "use_cons": "Use consumable",
}
ACTION_WORDS = {"buy_voucher": "Buy voucher", "reroll": "Reroll", "skip": "Leave shop"}

# Column -> kind for columns whose meaning is unambiguous everywhere
DEFAULT_KEYS = {
    "deck": "deck",
    "stake": "stake",
    "blind_key": "blind",
    "action_name": "action",
    "item_key": "item",
    "joker": "item",
    "jokers_before": "items",
    "consumables_before": "items",
    "final_jokers": "items",
}


def _split_items(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [v.strip() for v in re.split(r"[;,]", value) if v.strip()]
    return []


@dataclass(frozen=True)
class Names:
    pretty: bool = True

    def item(self, key: Any) -> str:
        if not isinstance(key, str) or not key:
            return ""
        return catalog.name(key) if self.pretty else key

    deck = item
    blind = item

    def items(self, value: Any) -> str:
        return ", ".join(self.item(k) for k in _split_items(value))

    def stake(self, stake: Any) -> str:
        return catalog.stake_name(stake) if self.pretty else str(stake)

    def stakes(self, value: Any) -> str:
        return ", ".join(self.stake(s) for s in _split_items(value))

    def action(self, name: Any) -> str:
        if not isinstance(name, str):
            return str(name)
        if not self.pretty:
            return name
        if name in ACTION_WORDS:
            return ACTION_WORDS[name]
        head, _, slot = name.rpartition("_")
        if head in ACTION_VERBS and slot.isdigit():
            return f"{ACTION_VERBS[head]} {int(slot) + 1}"
        return name.replace("_", " ").capitalize()

    def column(self, name: Any) -> str:
        if not self.pretty or not isinstance(name, str):
            return str(name)
        return COLUMNS.get(name, name.replace("_", " ").capitalize())

    def of(self, kind: str) -> Callable[[Any], str]:
        return {
            "item": self.item,
            "deck": self.deck,
            "blind": self.blind,
            "items": self.items,
            "stake": self.stake,
            "stakes": self.stakes,
            "action": self.action,
        }[kind]

    def frame(
        self,
        df: pd.DataFrame,
        keys: dict[str, str] | None = None,
        rename: bool = True,
    ) -> pd.DataFrame:
        """Copy of ``df`` with key columns mapped and headers renamed.

        ``keys`` maps column -> kind (item/items/deck/stake/blind/action) and
        is merged over ``DEFAULT_KEYS``.  List columns are always joined into
        text so tables render the same in both modes.
        """
        out = df.copy()
        kinds = {**DEFAULT_KEYS, **(keys or {})}
        for col, kind in kinds.items():
            if col in out:
                out[col] = out[col].map(self.of(kind))
        if rename:
            out = out.rename(columns=self.column)
        return out

    def labels(self, keys: Iterable[Any], kind: str = "item") -> list[str]:
        f = self.of(kind)
        return [f(k) for k in keys]
