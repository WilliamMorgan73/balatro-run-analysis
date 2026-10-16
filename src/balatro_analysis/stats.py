"""Uncertainty and significance: confidence intervals per run and tests between runs.

Everything here is deterministic (fixed bootstrap seed) so cached tables and
exports don't change between reruns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from balatro_analysis.loader import Run

LEVEL = 0.95
N_BOOT = 2000
SEED = 0

# episodes.csv column -> label used in the tables
OUTCOMES = {"ante": "ante reached", "won": "win rate", "return": "return"}


def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of ``n_boot`` resamples of ``x``.  A resample is just how many times
    each distinct value is drawn (multinomial), so memory scales with the number
    of distinct values, not episodes; antes have a handful.  Batched so long
    runs of continuous values stay around a million cells at a time."""
    values, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    batch = max(1, 1_000_000 // len(values))
    out = [
        rng.multinomial(len(x), p, size=min(batch, n_boot - i)) @ values / len(x)
        for i in range(0, n_boot, batch)
    ]
    return np.concatenate(out)


def bootstrap_ci(
    x: np.ndarray | pd.Series, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean; (nan, nan) for fewer than 2 values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    means = _boot_means(x, n_boot, np.random.default_rng(seed))
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def bootstrap_diff_ci(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``mean(a) - mean(b)`` for independent samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = _boot_means(a, n_boot, rng) - _boot_means(b, n_boot, rng)
    q = (1 - level) / 2
    return float(np.quantile(diffs, q)), float(np.quantile(diffs, 1 - q))


def wilson_ci(successes: int, n: int, level: float = LEVEL) -> tuple[float, float]:
    """Wilson score interval for a proportion; unlike the bootstrap it isn't
    stuck at (0, 0) when nothing has been won yet."""
    if n == 0:
        return float("nan"), float("nan")
    z = sps.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def summary_ci(runs: list[Run]) -> pd.DataFrame:
    """Per run and outcome: episodes, mean, std, median and a 95% CI of the mean
    (Wilson for win rate, bootstrap otherwise)."""
    rows = []
    for r in runs:
        e = r.episodes
        for col, label in OUTCOMES.items():
            if col not in e:
                continue
            x = e[col].dropna().astype(float)
            n = len(x)
            if col == "won":
                lo, hi = wilson_ci(int(x.sum()), n)
            else:
                lo, hi = bootstrap_ci(x)
            rows.append(
                {
                    "run": r.name,
                    "outcome": label,
                    "episodes": n,
                    "mean": x.mean() if n else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "std": x.std() if n > 1 else float("nan"),
                    "median": x.median() if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def eval_curve(runs: list[Run]) -> pd.DataFrame:
    """Per run and eval round: training episodes done so far, eval games, mean ante
    with a bootstrap CI and win rate with a Wilson CI.  ``runs`` are eval runs
    (episodes with ``eval_round`` / ``train_episodes``); others are skipped."""
    rows = []
    for r in runs:
        e = r.episodes
        if e.empty or "eval_round" not in e:
            continue
        for rnd, g in e.groupby("eval_round"):
            ante, won = g["ante"].astype(float), g["won"].astype(int)
            lo, hi = bootstrap_ci(ante)
            w_lo, w_hi = wilson_ci(int(won.sum()), len(won))
            rows.append(
                {
                    "run": r.name,
                    "eval_round": int(rnd),
                    "train_episodes": int(g["train_episodes"].iloc[0]),
                    "games": len(g),
                    "mean ante": ante.mean(),
                    "mean ante low": lo,
                    "mean ante high": hi,
                    "win rate": won.mean(),
                    "win rate low": w_lo,
                    "win rate high": w_hi,
                }
            )
    return pd.DataFrame(rows)


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U p-value and rank-biserial effect size
    (= Cliff's delta: +1 when every ``a`` beats every ``b``)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    if np.unique(np.concatenate([a, b])).size == 1:  # all tied: no evidence either way
        return 1.0, 0.0
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(p), float(2 * u / (len(a) * len(b)) - 1)


def wilcoxon(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired differences."""
    if len(diffs) == 0:
        return float("nan")
    if not np.any(diffs):  # every pair tied
        return 1.0
    return float(sps.wilcoxon(diffs, zero_method="zsplit").pvalue)


def _pair_keys(e: pd.DataFrame) -> list[str]:
    """Columns that pin down the game played: the seed, plus deck and stake when
    the deck/stake is drawn per episode, plus the eval round for eval episodes
    (the same eval seeds are replayed every round)."""
    return [c for c in ("seed", "deck", "stake", "eval_round") if c in e]


def paired_ante(run: Run, baseline: Run) -> pd.DataFrame:
    """Episodes both runs played on the same seed (and deck/stake): ante in each.
    Only the first episode per key is kept, so a reused seed can't be double-counted."""
    keys = _pair_keys(run.episodes)
    if "seed" not in keys or "seed" not in baseline.episodes:
        return pd.DataFrame(columns=[*keys, "ante", "ante_baseline"])
    keys = [k for k in keys if k in baseline.episodes]
    a = run.episodes.drop_duplicates(keys)[[*keys, "ante"]]
    b = baseline.episodes.drop_duplicates(keys)[[*keys, "ante"]]
    return a.merge(b, on=keys, suffixes=("", "_baseline"))


def compare(runs: list[Run], baseline: str) -> pd.DataFrame:
    """Each run against ``baseline`` (by name).

    Unpaired: difference in mean ante with a bootstrap CI, Mann-Whitney U on ante
    reached with its rank-biserial effect size, difference in win rate with
    Fisher's exact test.  Paired: on episodes played on the same seed, the mean
    per-seed ante difference and a Wilcoxon signed-rank test.
    """
    base = next((r for r in runs if r.name == baseline), None)
    if base is None:
        return pd.DataFrame()
    b_ante = base.episodes["ante"].to_numpy(dtype=float)
    b_won = base.episodes["won"].to_numpy(dtype=int)
    rows = []
    for r in runs:
        if r.name == baseline:
            continue
        a_ante = r.episodes["ante"].to_numpy(dtype=float)
        a_won = r.episodes["won"].to_numpy(dtype=int)
        lo, hi = bootstrap_diff_ci(a_ante, b_ante)
        p_mw, effect = mann_whitney(a_ante, b_ante)
        if len(a_won) and len(b_won):
            table = [
                [a_won.sum(), len(a_won) - a_won.sum()],
                [b_won.sum(), len(b_won) - b_won.sum()],
            ]
            p_win = float(sps.fisher_exact(table).pvalue)
            win_diff = a_won.mean() - b_won.mean()
        else:
            p_win = win_diff = float("nan")
        pairs = paired_ante(r, base)
        diffs = (pairs["ante"] - pairs["ante_baseline"]).to_numpy(dtype=float)
        rows.append(
            {
                "run": r.name,
                "baseline": baseline,
                "episodes": len(a_ante),
                "baseline_episodes": len(b_ante),
                "mean_ante_diff": a_ante.mean() - b_ante.mean() if len(a_ante) else float("nan"),
                "diff_ci_low": lo,
                "diff_ci_high": hi,
                "rank_biserial": effect,
                "mann_whitney_p": p_mw,
                "win_rate_diff": win_diff,
                "fisher_p": p_win,
                "paired_episodes": len(diffs),
                "paired_mean_diff": diffs.mean() if len(diffs) else float("nan"),
                "wilcoxon_p": wilcoxon(diffs),
            }
        )
    return pd.DataFrame(rows)


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rho and its two-sided p-value; (nan, nan) if either side is constant."""
    if pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return float("nan"), float("nan")
    res = sps.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def bh_adjust(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs stay NaN."""
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.any():
        out[ok] = sps.false_discovery_control(p[ok].to_numpy(dtype=float), method="bh")
    return out


def with_q_values(df: pd.DataFrame, by: str = "run") -> pd.DataFrame:
    """Add ``q_value`` (BH within each ``by`` group) and a ``significant`` flag (q < 0.05)."""
    if df.empty or "p_value" not in df:
        return df
    df = df.copy()
    df["q_value"] = df.groupby(by, group_keys=False)["p_value"].apply(bh_adjust)
    df["significant"] = df["q_value"] < 1 - LEVEL
    return df
