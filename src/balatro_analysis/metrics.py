"""Pure pandas metrics over one or more runs.

Functions that take several runs return long-format frames with a ``run``
column, so plots can colour by run.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from balatro_analysis import catalog
from balatro_analysis.loader import Run

BUY_ACTIONS = ("buy_card_0", "buy_card_1", "buy_pack_0", "buy_pack_1", "buy_voucher")
SELL_JOKER_PREFIX = "sell_joker_"


def filter_episodes(
    run: Run,
    start: int,
    end: int,
    decks: list[str] | None = None,
    stakes: list[str] | None = None,
    eval_rounds: list[int] | None = None,
) -> Run:
    """Restrict a run (all its tables) to episodes in ``[start, end]`` that were
    played on one of ``decks`` / ``stakes`` and, for eval episodes, in one of
    ``eval_rounds`` (None = no filter).  ``run.eval`` keeps every eval round and
    gets the deck/stake filter only (the episode range is for training)."""
    e = run.episodes
    keep = (e["episode"] >= start) & (e["episode"] <= end)
    if decks is not None:
        keep &= e["deck"].isin(decks)
    if stakes is not None:
        keep &= e["stake"].isin(stakes)
    if eval_rounds is not None and "eval_round" in e:
        keep &= e["eval_round"].isin(eval_rounds)
    episodes = set(e.loc[keep, "episode"])

    def cut(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        return df[df["episode"].isin(episodes)].reset_index(drop=True)

    return Run(
        name=run.name,
        path=run.path,
        episodes=cut(run.episodes),
        config=run.config,
        steps=cut(run.steps),
        rounds=cut(run.rounds),
        shops=cut(run.shops),
        eval=filter_episodes(run.eval, 0, np.inf, decks, stakes) if run.eval else None,
    )


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def summary(runs: list[Run]) -> pd.DataFrame:
    rows = []
    for r in runs:
        e = r.episodes
        rows.append(
            {
                "run": r.name,
                "play agent": r.config.get("play_agent", "?"),
                "shop agent": r.config.get("shop_agent", "?"),
                "decks": ";".join(sorted(e["deck"].unique())),
                "stakes": ";".join(sorted(e["stake"].unique())),
                "episodes": len(e),
                "mean ante": e["ante"].mean() if len(e) else float("nan"),
                "max ante": e["ante"].max() if len(e) else float("nan"),
                "win rate": e["won"].mean() if len(e) else float("nan"),
                "mean return": e["return"].mean() if len(e) else float("nan"),
                "shop steps / ep": e["shop_steps"].mean() if len(e) else float("nan"),
                "s / ep": e["seconds"].mean() if len(e) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


CURVE_METRICS = {"mean ante": "ante", "win rate": "won", "mean return": "return"}
Z95 = 1.959964


def learning_curve(runs: list[Run], window: int) -> pd.DataFrame:
    """Rolling mean ante, win rate and return per episode, each with a 95%
    normal-approximation band (``<metric> low`` / ``<metric> high``: mean ± 1.96
    standard errors over the window; zero width until the window has 2 episodes)."""
    frames = []
    cols = list(CURVE_METRICS.values())
    for r in runs:
        e = r.episodes.sort_values("episode")
        rolling = e[cols].astype(float).rolling(window, min_periods=1)
        mean, std, n = rolling.mean(), rolling.std().fillna(0), rolling.count()
        half = Z95 * std / np.sqrt(n)
        frame = {"run": r.name, "episode": e["episode"].to_numpy()}
        for label, col in CURVE_METRICS.items():
            frame[label] = mean[col].to_numpy()
            frame[f"{label} low"] = (mean[col] - half[col]).to_numpy()
            frame[f"{label} high"] = (mean[col] + half[col]).to_numpy()
        frames.append(pd.DataFrame(frame))
    return _concat(frames)


def by_deck(runs: list[Run]) -> pd.DataFrame:
    """Per (run, deck, stake): episodes, mean/max ante, win rate."""
    frames = []
    for r in runs:
        if r.episodes.empty:
            continue
        g = r.episodes.groupby(["deck", "stake"], as_index=False).agg(
            episodes=("episode", "size"),
            mean_ante=("ante", "mean"),
            max_ante=("ante", "max"),
            win_rate=("won", "mean"),
        )
        frames.append(g.assign(run=r.name))
    return _concat(frames)


def ante_distribution(runs: list[Run]) -> pd.DataFrame:
    """Fraction of episodes ending at each ante, per run."""
    frames = []
    for r in runs:
        counts = r.episodes["ante"].value_counts(normalize=True).sort_index()
        frames.append(
            pd.DataFrame({"run": r.name, "ante": counts.index, "fraction": counts.to_numpy()})
        )
    return _concat(frames)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def action_names(run: Run) -> list[str]:
    names = run.config.get("action_names")
    if names:
        return list(names)
    known = {"reroll", "skip"}
    return [
        c for c in run.episodes.columns if c.startswith(("buy_", "sell_", "use_")) or c in known
    ]


def action_frequency(runs: list[Run]) -> pd.DataFrame:
    """Share of shop decisions spent on each action, per run."""
    frames = []
    for r in runs:
        names = [n for n in action_names(r) if n in r.episodes]
        totals = r.episodes[names].sum()
        share = totals / totals.sum() if totals.sum() else totals
        frames.append(
            pd.DataFrame(
                {
                    "run": r.name,
                    "action": names,
                    "count": totals.to_numpy(),
                    "share": share,
                }
            )
        )
    return _concat(frames)


def action_by_ante(run: Run) -> pd.DataFrame:
    """Action share within each ante (rows: action, columns: ante)."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame()
    table = pd.crosstab(run.steps["action_name"], run.steps["ante"], normalize="columns")
    order = [n for n in action_names(run) if n in table.index]
    return table.reindex(order)


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------


def money_by_ante(runs: list[Run]) -> pd.DataFrame:
    """Money held before each shop decision."""
    return _concat(
        [
            r.steps[["episode", "ante", "dollars_before"]].assign(run=r.name)
            for r in runs
            if r.steps is not None
        ]
    )


def spend_per_episode(runs: list[Run]) -> pd.DataFrame:
    return _concat(
        [
            r.episodes[["episode", "dollars_spent", "ante"]].assign(run=r.name)
            for r in runs
            if "dollars_spent" in r.episodes
        ]
    )


# ---------------------------------------------------------------------------
# Jokers
# ---------------------------------------------------------------------------


def joker_purchases(run: Run) -> pd.DataFrame:
    """Shop steps that bought a joker (``item_key`` starts with ``j_``)."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame(columns=["episode", "ante", "item_key"])
    s = run.steps
    bought = s["action_name"].isin(BUY_ACTIONS) & s["item_key"].map(catalog.is_joker)
    return s.loc[bought, ["episode", "step", "ante", "round", "item_key", "item_cost"]]


def most_bought(runs: list[Run], top: int = 20) -> pd.DataFrame:
    frames = []
    for r in runs:
        counts = joker_purchases(r)["item_key"].value_counts().head(top)
        frames.append(
            pd.DataFrame(
                {
                    "run": r.name,
                    "joker": counts.index,
                    "bought": counts.values,
                }
            )
        )
    return _concat(frames)


def first_bought_ante(runs: list[Run]) -> pd.DataFrame:
    """For each (episode, joker), the ante it was first bought in."""
    frames = []
    for r in runs:
        p = joker_purchases(r)
        if p.empty:
            continue
        first = p.groupby(["episode", "item_key"], as_index=False)["ante"].min()
        frames.append(first.rename(columns={"item_key": "joker"}).assign(run=r.name))
    return _concat(frames)


def final_builds(run: Run) -> list[list[str]]:
    """Jokers held at the end of each episode."""
    if "final_jokers" not in run.episodes:
        return []
    return [[j for j in s.split(";") if j] for s in run.episodes["final_jokers"]]


JOKER_STATS_COLUMNS = [
    "bought",
    "episodes_bought",
    "held_at_end",
    "mean_ante_held",
    "win_rate_held",
    "first_ante",
]


def joker_stats(runs: list[Run]) -> pd.DataFrame:
    """Per joker, across ``runs``: times bought, episodes it was bought in, episodes
    it was held at the end, mean ante reached and win rate in those, median ante
    first bought.

    Indexed by joker key; jokers never seen are absent.
    """
    bought_rows, held_rows = [], []
    for r in runs:
        p = joker_purchases(r)
        if not p.empty:
            bought_rows.append(p[["episode", "ante", "item_key"]].assign(run=r.name))
        if "final_jokers" in r.episodes:
            won = r.episodes["won"] if "won" in r.episodes else [0] * len(r.episodes)
            for ep, ante, win, build in zip(
                r.episodes["episode"], r.episodes["ante"], won, final_builds(r), strict=True
            ):
                held_rows += [
                    {"run": r.name, "episode": ep, "ante": ante, "won": bool(win), "joker": j}
                    for j in set(build)
                ]
    parts = []
    if bought_rows:
        b = _concat(bought_rows).rename(columns={"item_key": "joker"})
        first = b.groupby(["run", "episode", "joker"])["ante"].min()
        parts.append(
            pd.DataFrame(
                {
                    "bought": b.groupby("joker").size(),
                    "episodes_bought": first.groupby("joker").size(),
                    "first_ante": first.groupby("joker").median(),
                }
            )
        )
    if held_rows:
        h = pd.DataFrame(held_rows).groupby("joker")
        parts.append(
            pd.DataFrame(
                {
                    "held_at_end": h.size(),
                    "mean_ante_held": h["ante"].mean(),
                    "win_rate_held": h["won"].mean(),
                }
            )
        )
    stats = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    stats = stats.reindex(columns=JOKER_STATS_COLUMNS)
    for col in ("bought", "episodes_bought", "held_at_end"):
        stats[col] = stats[col].fillna(0).astype(int)
    stats.index.name = "joker"
    return stats


def joker_held_by_run(runs: list[Run]) -> pd.DataFrame:
    """Per (joker, run): episodes that ended holding the joker, and the run's episodes."""
    rows = []
    for r in runs:
        builds = final_builds(r)
        counts: dict[str, int] = {}
        for b in builds:
            for j in set(b):
                counts[j] = counts.get(j, 0) + 1
        rows += [
            {"joker": j, "run": r.name, "held_at_end": n, "episodes": len(builds)}
            for j, n in counts.items()
        ]
    return pd.DataFrame(rows, columns=["joker", "run", "held_at_end", "episodes"])


def deck_top_jokers(runs: list[Run], deck: str, n: int = 3) -> pd.Series:
    """The ``n`` jokers most often held at the end of episodes played with ``deck``."""
    counts: dict[str, int] = {}
    for r in runs:
        if "deck" not in r.episodes:
            continue
        for d, build in zip(r.episodes["deck"], final_builds(r), strict=False):
            if d == deck:
                for j in set(build):
                    counts[j] = counts.get(j, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return pd.Series(dict(top), dtype=int, name="held_at_end")


def joker_cooccurrence(run: Run, top: int = 15) -> pd.DataFrame:
    """Symmetric count matrix: how often two jokers ended an episode together.

    The diagonal is how often each joker ended an episode held at all.
    Limited to the ``top`` most common jokers.
    """
    builds = final_builds(run)
    singles = Counter(j for b in builds for j in set(b))
    keep = [j for j, _ in singles.most_common(top)]
    if not keep:
        return pd.DataFrame()
    m = pd.DataFrame(0, index=keep, columns=keep, dtype=int)
    for b in builds:
        held = sorted(set(b) & set(keep))
        for j in held:
            m.loc[j, j] += 1
        for a, c in combinations(held, 2):
            m.loc[a, c] += 1
            m.loc[c, a] += 1
    return m


def sell_then_buy(run: Run) -> pd.DataFrame:
    """Pairs (joker sold -> next joker bought in the same episode)."""
    if run.steps is None or run.steps.empty:
        return pd.DataFrame(columns=["sold", "bought", "count"])
    pairs: Counter[tuple[str, str]] = Counter()
    for _, ep in run.steps.sort_values(["episode", "step"]).groupby("episode"):
        pending = None
        for name, key in zip(ep["action_name"], ep["item_key"], strict=True):
            if name.startswith(SELL_JOKER_PREFIX):
                pending = key
            elif pending and name in BUY_ACTIONS and catalog.is_joker(key):
                pairs[(pending, key)] += 1
                pending = None
    rows = [{"sold": s, "bought": b, "count": n} for (s, b), n in pairs.most_common()]
    return pd.DataFrame(rows, columns=["sold", "bought", "count"])


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def score_ratio(runs: list[Run]) -> pd.DataFrame:
    """Chips scored / blind target for every blind played."""
    frames = []
    for r in runs:
        if r.rounds is None or r.rounds.empty:
            continue
        d = r.rounds.copy()
        d["ratio"] = d["chips"] / d["target"].where(d["target"] > 0)
        d["blind type"] = d["blind_key"].map(blind_type)
        frames.append(d.assign(run=r.name))
    return _concat(frames)


def blind_type(key: str) -> str:
    if key == "bl_small":
        return "Small"
    if key == "bl_big":
        return "Big"
    return "Boss"


def deaths(runs: list[Run]) -> pd.DataFrame:
    """The blind that ended each lost episode."""
    frames = []
    for r in runs:
        if r.rounds is None or r.rounds.empty:
            continue
        lost = r.rounds[~r.rounds["beaten"].astype(bool)]
        frames.append(lost[["episode", "ante", "blind_key"]].assign(run=r.name))
    return _concat(frames)


BOSS_STATS_COLUMNS = ["faced", "beaten", "losses", "beat_rate", "mean_ratio", "antes"]


def boss_stats(runs: list[Run]) -> pd.DataFrame:
    """Per boss blind across ``runs``: times faced, beaten and lost to, beat rate,
    mean chips scored / target, and the antes it was met at (sorted, distinct).

    Indexed by blind key, most-faced first.
    """
    frames = [
        r.rounds.assign(run=r.name) for r in runs if r.rounds is not None and not r.rounds.empty
    ]
    if not frames:
        return pd.DataFrame(columns=BOSS_STATS_COLUMNS)
    d = _concat(frames)
    d = d[d["blind_key"].map(blind_type) == "Boss"].copy()
    if d.empty:
        return pd.DataFrame(columns=BOSS_STATS_COLUMNS)
    d["beaten"] = d["beaten"].astype(bool)
    d["ratio"] = d["chips"] / d["target"].where(d["target"] > 0)
    g = d.groupby("blind_key")
    out = pd.DataFrame(
        {
            "faced": g.size(),
            "beaten": g["beaten"].sum().astype(int),
            "mean_ratio": g["ratio"].mean(),
            "antes": g["ante"].agg(lambda a: sorted({int(x) for x in a})),
        }
    )
    out["losses"] = out["faced"] - out["beaten"]
    out["beat_rate"] = out["beaten"] / out["faced"]
    out.index.name = "blind"
    return out[BOSS_STATS_COLUMNS].sort_values(["faced"], ascending=False, kind="stable")


# ---------------------------------------------------------------------------
# Episode explorer
# ---------------------------------------------------------------------------


def episode_timeline(run: Run, episode: int) -> pd.DataFrame:
    if run.steps is None:
        return pd.DataFrame()
    s = run.steps[run.steps["episode"] == episode].sort_values("step").copy()
    cols = [
        "step",
        "ante",
        "round",
        "dollars_before",
        "action_name",
        "item_key",
        "item_cost",
        "dollars_after",
        "reward",
        "jokers_before",
        "consumables_before",
    ]
    return s[[c for c in cols if c in s]]


def episode_rounds(run: Run, episode: int) -> pd.DataFrame:
    """The episode's blinds, with the jokers and consumables held entering each one.

    Those come from the shop visit before the blind: its last decision is leaving
    the shop, which changes nothing, so its ``*_before`` lists are what was held
    walking into the next blind (after any pack picks).  The first blind comes
    before any shop, so it has none (a deck's starting consumables aren't logged).
    """
    if run.rounds is None:
        return pd.DataFrame()
    rounds = run.rounds[run.rounds["episode"] == episode].reset_index(drop=True)
    if run.steps is None or not {"jokers_before", "consumables_before"} <= set(run.steps):
        return rounds
    s = run.steps[run.steps["episode"] == episode].sort_values("step")
    last = s.groupby("round").tail(1).set_index("round")  # leaving each shop
    for col, src in (("jokers", "jokers_before"), ("consumables", "consumables_before")):
        if col in rounds:  # logged directly by a newer harness
            continue
        rounds[col] = [
            last.at[r - 1, src] if r - 1 in last.index else None for r in rounds["round"]
        ]
    if "beaten" in rounds:  # held items next to the outcome, before the cash-out columns
        held = [c for c in ("jokers", "consumables") if c in rounds]
        rest = [c for c in rounds if c not in held]
        at = rest.index("beaten") + 1
        rounds = rounds[rest[:at] + held + rest[at:]]
    return rounds


def item_actions(runs: list[Run]) -> pd.DataFrame:
    """How many times each item (by key) was bought, sold and used in the shop,
    over all ``runs``: index ``item_key``, columns ``bought``, ``sold``, ``used``."""
    frames = [r.steps[["action_name", "item_key"]] for r in runs if r.steps is not None]
    s = pd.concat(frames, ignore_index=True).dropna() if frames else pd.DataFrame()
    if s.empty:
        return pd.DataFrame(columns=["bought", "sold", "used"])
    kind = s["action_name"].map(
        lambda a: "bought" if a in BUY_ACTIONS else "sold" if a.startswith("sell_") else
        "used" if a.startswith("use_") else None
    )  # fmt: skip
    counts = pd.crosstab(s["item_key"], kind)
    return counts.reindex(columns=["bought", "sold", "used"], fill_value=0)
