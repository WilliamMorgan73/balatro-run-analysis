"""Uncertainty and significance: confidence intervals per run and tests between runs.

Everything here is deterministic (fixed bootstrap seed) so cached tables and
exports don't change between reruns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from balatro_analysis import metrics
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


# ---------------------------------------------------------------------------
# Groups: one config trained with several seeds (``--group``)
# ---------------------------------------------------------------------------


def group_of(run: Run) -> str:
    """The run's ``--group``, or its own name when it has none."""
    return run.config.get("group") or run.name


def groups(runs: list[Run]) -> dict[str, list[Run]]:
    """Runs by group, in order of first appearance."""
    out: dict[str, list[Run]] = {}
    for r in runs:
        out.setdefault(group_of(r), []).append(r)
    return out


def t_ci(x: np.ndarray | pd.Series, level: float = LEVEL) -> tuple[float, float]:
    """Student-t CI of the mean; (nan, nan) for fewer than 2 values.  Used across
    seeds, where there are too few values for a bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    half = sps.t.ppf(1 - (1 - level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def run_means(runs: list[Run]) -> pd.DataFrame:
    """One row per run: its group, training seed, episodes and mean of each outcome."""
    rows = []
    for r in runs:
        e = r.episodes
        rows.append(
            {
                "group": group_of(r),
                "run": r.name,
                "seed": r.config.get("seed"),
                "episodes": len(e),
                **{
                    label: e[col].astype(float).mean() if len(e) else float("nan")
                    for col, label in OUTCOMES.items()
                    if col in e
                },
            }
        )
    return pd.DataFrame(rows)


def group_summary(runs: list[Run]) -> pd.DataFrame:
    """Per group and outcome: the mean over its runs' means, with a t CI across
    runs (seeds), so the interval reflects seed-to-seed variation rather than
    the number of episodes pooled."""
    means = run_means(runs)
    rows = []
    for group, g in means.groupby("group", sort=False):
        for label in OUTCOMES.values():
            if label not in g:
                continue
            x = g[label].to_numpy(dtype=float)
            lo, hi = t_ci(x)
            rows.append(
                {
                    "group": group,
                    "outcome": label,
                    "runs": len(x),
                    "episodes": int(g["episodes"].sum()),
                    "mean": float(np.nanmean(x)) if len(x) else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "std_across_runs": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's t-test: (p, ci_low, ci_high) for mean(a) - mean(b); nan if either has < 2."""
    if len(a) < 2 or len(b) < 2 or (np.std(a) == 0 and np.std(b) == 0):
        return float("nan"), float("nan"), float("nan")
    with warnings.catch_warnings():
        # one group can have no spread (e.g. every seed reaches ante 1); Welch still
        # works off the other group's variance, scipy just warns about it
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sps.ttest_ind(a, b, equal_var=False)
        ci = res.confidence_interval(LEVEL)
    return float(res.pvalue), float(ci.low), float(ci.high)


def compare_groups(runs: list[Run], baseline: str) -> pd.DataFrame:
    """Each group against the ``baseline`` group, on the per-run means: difference
    in mean ante and win rate with Welch's t-test and CI (seeds as the sample)."""
    means = run_means(runs)
    base = means[means["group"] == baseline]
    if base.empty:
        return pd.DataFrame()
    rows = []
    for group, g in means.groupby("group", sort=False):
        if group == baseline:
            continue
        row = {"group": group, "baseline": baseline, "runs": len(g), "baseline_runs": len(base)}
        for label, key in (("ante reached", "ante"), ("win rate", "win_rate")):
            a = g[label].to_numpy(dtype=float)
            b = base[label].to_numpy(dtype=float)
            p, lo, hi = _welch(a, b)
            row |= {
                f"{key}_diff": a.mean() - b.mean(),
                f"{key}_ci_low": lo,
                f"{key}_ci_high": hi,
                f"{key}_welch_p": p,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _group_curve(
    per_run: pd.DataFrame, group_by_run: dict[str, str], x: str, metrics: list[str]
) -> pd.DataFrame:
    """Average per-run curves within each group at each ``x``, with a t band across
    runs.  ``train_episodes`` / ``games`` (eval curves) are carried over / summed.
    The result's ``run`` column holds the group, so the run plots can draw it."""
    if per_run.empty:
        return pd.DataFrame()
    per_run = per_run.assign(group=per_run["run"].map(group_by_run))
    rows = []
    for (group, xv), g in per_run.groupby(["group", x], sort=False):
        row = {"run": group, x: xv, "runs": len(g)}
        if "train_episodes" in g:
            row["train_episodes"] = int(g["train_episodes"].iloc[0])
        if "games" in g:
            row["games"] = int(g["games"].sum())
        for m in metrics:
            lo, hi = t_ci(g[m])
            row |= {m: g[m].mean(), f"{m} low": lo, f"{m} high": hi}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run", x]).reset_index(drop=True)


def group_learning_curve(runs: list[Run], window: int) -> pd.DataFrame:
    """Rolling learning curve averaged over each group's runs, band across runs."""
    per_run = metrics.learning_curve(runs, window)
    by_run = {r.name: group_of(r) for r in runs}
    return _group_curve(per_run, by_run, "episode", list(metrics.CURVE_METRICS))


def group_eval_curve(eval_runs: list[Run]) -> pd.DataFrame:
    """``eval_curve`` averaged over each group's runs at each eval round, band
    across runs; ``games`` is the total over the runs."""
    by_run = {r.name: group_of(r) for r in eval_runs}
    return _group_curve(eval_curve(eval_runs), by_run, "eval_round", ["mean ante", "win rate"])


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
