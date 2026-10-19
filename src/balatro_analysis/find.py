"""Finding episodes and runs worth looking at: sortable tables, filters and
suggestions ("best episode", "closest loss", "best run per group", ...)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from balatro_analysis.loader import Run
from balatro_analysis.names import split_items

EPISODE_COLUMNS = [
    "episode",
    "eval_round",
    "ante",
    "won",
    "return",
    "final_dollars",
    "dollars_spent",
    "shop_steps",
    "deck",
    "stake",
    "seed",
    "final_jokers",
]
OUTCOMES = ("All", "Won", "Lost")


def episode_table(run: Run) -> pd.DataFrame:
    """One row per episode, with ``score_vs_target`` (chips / target on the blind
    that ended a lost episode: 0.95 means it fell 5% short)."""
    e = run.episodes
    out = e[[c for c in EPISODE_COLUMNS if c in e]].copy()
    if "final_jokers" in out:
        out["final_jokers"] = out["final_jokers"].map(split_items)
    lost = _losing_blinds(run)
    if lost is not None:
        out = out.merge(lost, on="episode", how="left")
    return out.reset_index(drop=True)


def _losing_blinds(run: Run) -> pd.DataFrame | None:
    r = run.rounds
    if r is None or not {"beaten", "chips", "target"} <= set(r):
        return None
    lost = r[~r["beaten"].astype(bool)].copy()
    lost["score_vs_target"] = lost["chips"] / lost["target"].where(lost["target"] > 0)
    cols = ["episode", "score_vs_target"] + (["blind_key"] if "blind_key" in lost else [])
    return lost[cols].rename(columns={"blind_key": "lost_to"}).drop_duplicates("episode")


def filter_episode_table(
    table: pd.DataFrame,
    decks: list[str] | None = None,
    stakes: list[str] | None = None,
    antes: tuple[int, int] | None = None,
    outcome: str = "All",
    jokers: list[str] | None = None,
) -> pd.DataFrame:
    """Rows matching every filter given (None = no filter).  ``jokers``: the
    episode ended holding all of them."""
    keep = pd.Series(True, index=table.index)
    if decks is not None:
        keep &= table["deck"].isin(decks)
    if stakes is not None:
        keep &= table["stake"].isin(stakes)
    if antes is not None:
        keep &= table["ante"].between(*antes)
    if outcome != "All":
        keep &= table["won"].astype(bool) == (outcome == "Won")
    if jokers:
        need = set(jokers)
        keep &= table["final_jokers"].map(lambda held: need <= set(held))
    return table[keep]


@dataclass(frozen=True)
class Suggestion:
    label: str
    episode: int
    reason: str


def suggested_episodes(table: pd.DataFrame) -> list[Suggestion]:
    """Episodes worth opening first, each with why it was picked.  Suggestions
    with nothing to point at (e.g. no losses) are left out."""
    if table.empty:
        return []
    t = table.reset_index(drop=True)
    out: list[Suggestion] = []

    def add(label: str, i: int, reason: str) -> None:
        out.append(Suggestion(label, int(t.at[i, "episode"]), reason))

    best = t.sort_values(["ante", "return", "final_dollars"], ascending=False, kind="stable").index[
        0
    ]
    won = " and won" if t.at[best, "won"] else ""
    add("Best", best, f"reached ante {t.at[best, 'ante']}{won}: the furthest in this run")

    if "score_vs_target" in t and t["score_vs_target"].notna().any():
        i = t["score_vs_target"].idxmax()
        blind = f" {t.at[i, 'lost_to']}" if "lost_to" in t else ""
        add(
            "Closest loss",
            i,
            f"lost on{blind} with {t.at[i, 'score_vs_target']:.0%} of the score needed",
        )

    for label, col, what in (
        ("Richest finish", "final_dollars", "${} at the end"),
        ("Biggest spender", "dollars_spent", "${} spent in the shop"),
        ("Most shop decisions", "shop_steps", "{} shop decisions"),
    ):
        if col in t and t[col].notna().any() and t[col].max() > 0:
            i = t[col].idxmax()
            add(label, i, what.format(int(t.at[i, col])))

    if "final_jokers" in t:
        builds = t["final_jokers"]
        seen = Counter(j for held in builds for j in set(held))
        rarity = builds.map(
            lambda held: sum(seen[j] for j in held) / len(held) / len(t) if held else None
        )
        if rarity.notna().any():
            i = rarity.astype(float).idxmin()
            add(
                "Rarest build",
                i,
                f"its jokers ended {rarity[i]:.0%} of episodes on average: an unusual build",
            )

    if t["ante"].nunique() > 1:
        excess = t["ante"] - t.groupby(["deck", "stake"])["ante"].transform("mean")
        i = excess.idxmax()
        if excess[i] > 0:
            add(
                "Biggest outlier",
                i,
                f"{excess[i]:+.1f} antes above the mean for its deck and stake",
            )

    if "eval_round" in t and t["eval_round"].nunique() > 1:
        first, last = t["eval_round"].min(), t["eval_round"].max()
        a0 = t[t["eval_round"] == first].set_index("seed")["ante"]
        a1 = t[t["eval_round"] == last].reset_index().set_index("seed")
        gain = (a1["ante"] - a0).dropna()
        if len(gain) and gain.max() > 0:
            seed = gain.idxmax()
            add(
                "Most improved eval game",
                int(a1.at[seed, "index"]),
                f"{seed}: ante {a0[seed]} in eval round {first} → {a1.at[seed, 'ante']} "
                f"in round {last}",
            )
    return out


# --- Runs -------------------------------------------------------------------

RUN_SORTS = {
    "Name": ("run", True),
    "Mean ante": ("mean_ante", False),
    "Win rate": ("win_rate", False),
    "Episodes": ("episodes", False),
    "Newest": ("started", False),
    "Shop agent": ("shop_agent", True),
    "Group": ("group", True),
}


def run_index(runs: dict[str, Run | None]) -> pd.DataFrame:
    """One row per run name (None: not loadable yet) with what the run list can
    be sorted and filtered by."""
    rows = []
    for name, r in runs.items():
        e = r.episodes if r is not None else pd.DataFrame()
        c = r.config if r is not None else {}
        rows.append(
            {
                "run": name,
                "group": c.get("group") or "",
                "shop_agent": c.get("shop_agent") or "?",
                "play_agent": c.get("play_agent") or "?",
                "started": c.get("started") or "",
                "episodes": len(e),
                "mean_ante": e["ante"].mean() if len(e) else float("nan"),
                "win_rate": e["won"].mean() if len(e) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def find_runs_in(
    index: pd.DataFrame,
    search: str = "",
    sort: str = "Name",
    shop_agents: list[str] | None = None,
    play_agents: list[str] | None = None,
    groups: list[str] | None = None,
) -> list[str]:
    """Run names matching the search and filters, in ``sort`` order (runs with
    nothing to sort by last)."""
    keep = index["run"].str.contains(search, case=False, regex=False) if search else True
    keep = pd.Series(keep, index=index.index)
    for col, allowed in (
        ("shop_agent", shop_agents),
        ("play_agent", play_agents),
        ("group", groups),
    ):
        if allowed is not None:
            keep &= index[col].isin(allowed)
    col, ascending = RUN_SORTS[sort]
    shown = index[keep].sort_values(
        [col, "run"], ascending=[ascending, True], na_position="last", kind="stable"
    )
    return shown["run"].tolist()


def best_per_group(index: pd.DataFrame) -> list[str]:
    """The highest mean-ante run of each group; with no groups, the best run."""
    scored = index.dropna(subset=["mean_ante"])
    if scored.empty:
        return []
    grouped = scored[scored["group"] != ""]
    if grouped.empty:
        return [scored.loc[scored["mean_ante"].idxmax(), "run"]]
    best = grouped.loc[grouped.groupby("group")["mean_ante"].idxmax()]
    return best.sort_values("group")["run"].tolist()
