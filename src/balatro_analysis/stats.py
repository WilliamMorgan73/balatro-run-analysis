"""Uncertainty and significance: confidence intervals per run and tests between runs.

Everything here is deterministic (fixed bootstrap seed) so cached tables and
exports don't change between reruns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import brentq

from balatro_analysis import metrics
from balatro_analysis.loader import Run, memo

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
    """Percentile bootstrap CI of ``mean(a) - mean(b)`` for independent samples
    (NaNs dropped); (nan, nan) if either has fewer than 2 values."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
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
    frames = [
        memo(r, "eval_curve", lambda r=r: _eval_curve(r))
        for r in runs
        if not r.episodes.empty and "eval_round" in r.episodes
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _eval_curve(r: Run) -> pd.DataFrame:
    """``eval_curve`` for one eval run."""
    rows = []
    for rnd, g in r.episodes.groupby("eval_round"):
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
                "mean_ante_diff": a_ante.mean() - b_ante.mean()
                if len(a_ante) and len(b_ante)
                else float("nan"),
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


# ---------------------------------------------------------------------------
# Interquartile mean (Agarwal et al. 2021, "Deep RL at the edge of the
# statistical precipice"; the ``rliable`` library)
# ---------------------------------------------------------------------------

IQM_OUTCOMES = {"ante": "ante reached", "return": "return"}  # not win rate: 0/1
SHARED_GAMES = "seeds × shared games"
WITHIN_RUNS = "episodes within runs"
# share of each run's games the others must also have played for the score matrix
MIN_SHARED = 0.5
FEW_VALUES = 64  # distinct values up to which bootstrap IQMs are counted, not sorted


def iqm(x: np.ndarray | pd.Series) -> float:
    """Interquartile mean: the mean of the middle 50% of all values, trimming
    ``int(0.25 n)`` from each end (``scipy.stats.trim_mean``, as rliable does)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    return float(sps.trim_mean(x, 0.25)) if len(x) else float("nan")


def _iqm_from_counts(values: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """``trim_mean(sample, 0.25)`` for each row of ``counts``: how many times each
    of the sorted distinct ``values`` is in a sample of size ``n``."""
    lo = int(0.25 * n)
    hi = n - lo
    end = np.cumsum(counts, axis=-1)
    start = end - counts
    kept = np.clip(np.minimum(end, hi) - np.maximum(start, lo), 0, None)
    return kept @ values / (hi - lo)


def _strata_counts(
    samples: list[np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    values: np.ndarray | None = None,
):
    """Bootstrap samples of the pooled ``samples`` with each sample (a run's
    episodes) resampled at its own size, as a stratum.  Yields ``(values,
    counts)`` batches: ``counts[b, i]`` is how often ``values[i]`` (sorted,
    distinct; or the given ``values``, a superset, so two calls batch alike) is
    in sample ``b``; batched so a batch stays ~1M cells."""
    if values is None:
        values = np.unique(np.concatenate(samples))
    strata = []
    for s in samples:
        v, c = np.unique(s, return_counts=True)
        strata.append((len(s), np.searchsorted(values, v), c / len(s)))
    rng = np.random.default_rng(seed)
    batch = max(1, 1_000_000 // len(values))
    for i in range(0, n_boot, batch):
        b = min(batch, n_boot - i)
        counts = np.zeros((b, len(values)))
        for size, where, p in strata:
            counts[:, where] += rng.multinomial(size, p, size=b)
        yield values, counts


def _clean(samples: list) -> list[np.ndarray]:
    samples = [s[~np.isnan(s)] for s in (np.asarray(s, dtype=float) for s in samples)]
    return [s for s in samples if len(s)]


def _interval(boot: np.ndarray, level: float = LEVEL) -> tuple:
    """Percentile interval of bootstrap statistics along the first axis."""
    a = (1 - level) / 2
    return np.quantile(boot, a, axis=0), np.quantile(boot, 1 - a, axis=0)


def _episodes_iqm_ci(
    samples: list[np.ndarray], n_boot: int = N_BOOT, level: float = LEVEL, seed: int = SEED
) -> tuple[float, float]:
    """Stratified bootstrap CI of the IQM of the pooled ``samples``: each sample
    (a run's episodes) is resampled with its own size, as a stratum.  Done with
    multinomial counts over the distinct values, like ``_boot_means``."""
    samples = _clean(samples)
    n = sum(len(s) for s in samples)
    if n < 2:
        return float("nan"), float("nan")
    iqms = np.concatenate(
        [_iqm_from_counts(v, c, n) for v, c in _strata_counts(samples, n_boot, seed)]
    )
    lo, hi = _interval(iqms, level)
    return float(lo), float(hi)


def score_matrix(runs: list[Run], col: str = "ante") -> np.ndarray | None:
    """Runs (rows) × games every run played (columns: same seed, deck/stake and
    eval round; the first episode per game), the "runs × tasks" matrix rliable
    bootstraps.  None for fewer than two runs, without seeds, or when the shared
    games are under ``MIN_SHARED`` of some run's games."""
    if len(runs) < 2 or any(col not in r.episodes for r in runs):
        return None
    keys = [k for k in _pair_keys(runs[0].episodes) if all(k in r.episodes for r in runs)]
    if "seed" not in keys:
        return None
    frames = [r.episodes.drop_duplicates(keys)[[*keys, col]] for r in runs]
    merged = frames[0].rename(columns={col: 0})
    for i, f in enumerate(frames[1:], 1):
        merged = merged.merge(f.rename(columns={col: i}), on=keys)
    if merged.empty or len(merged) < MIN_SHARED * max(len(f) for f in frames):
        return None
    return merged[list(range(len(runs)))].to_numpy(dtype=float).T


def _matrix_draws(shape: tuple[int, int], n_boot: int, rng: np.random.Generator, batch: int):
    """rliable's stratified resampling of a runs × tasks matrix: row indices
    drawn with replacement independently in each column, ``(batch, runs, tasks)``
    at a time."""
    r, t = shape
    for i in range(0, n_boot, batch):
        yield rng.integers(0, r, size=(min(batch, n_boot - i), r, t))


def _batch(cells_per_sample: int, cells: int = 2_000_000) -> int:
    """Bootstrap samples per batch so a batch holds about ``cells`` cells."""
    return max(1, cells // max(cells_per_sample, 1))


def _matrix_counts(matrix: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """``_strata_counts`` for a runs × tasks matrix of few distinct values,
    resampled as rliable does: yields ``(values, counts)`` batches."""
    values, codes = np.unique(matrix, return_inverse=True)
    codes = codes.reshape(matrix.shape)
    k = len(values)
    rng, batch = np.random.default_rng(seed), _batch(matrix.size)
    for idx in _matrix_draws(matrix.shape, n_boot, rng, batch):
        drawn = codes[idx, np.arange(matrix.shape[1])].reshape(len(idx), -1)
        flat = (drawn + k * np.arange(len(idx))[:, None]).ravel()
        yield values, np.bincount(flat, minlength=k * len(idx)).reshape(len(idx), k)


def stratified_bootstrap_ci(
    matrix: np.ndarray, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = SEED
) -> tuple[float, float]:
    """rliable's stratified bootstrap CI of the IQM of a runs × tasks matrix: each
    sample redraws the runs with replacement independently within each task
    (column), and the IQM is over every cell.  Percentile interval."""
    m = np.asarray(matrix, dtype=float)
    if len(np.unique(m)) <= FEW_VALUES:  # antes: count values instead of sorting cells
        iqms = np.concatenate(
            [_iqm_from_counts(v, c, m.size) for v, c in _matrix_counts(m, n_boot, seed)]
        )
    else:
        rng = np.random.default_rng(seed)
        iqms = np.concatenate(
            [
                sps.trim_mean(m[idx, np.arange(m.shape[1])].reshape(len(idx), -1), 0.25, axis=1)
                for idx in _matrix_draws(m.shape, n_boot, rng, _batch(m.size))
            ]
        )
    lo, hi = _interval(iqms, level)
    return float(lo), float(hi)


def _iqm_rows(name: str, runs: list[Run]) -> pd.DataFrame:
    rows = []
    for col, label in IQM_OUTCOMES.items():
        if any(col not in r.episodes for r in runs):
            continue
        matrix = score_matrix(runs, col)
        if matrix is not None:
            cells, method = matrix, SHARED_GAMES
            lo, hi = stratified_bootstrap_ci(matrix)
            games = matrix.shape[1]
        else:
            samples = [r.episodes[col].to_numpy(dtype=float) for r in runs]
            cells, method = np.concatenate(samples), WITHIN_RUNS
            lo, hi = _episodes_iqm_ci(samples)
            games = len(cells)
        rows.append(
            {
                "name": name,
                "outcome": label,
                "runs": len(runs),
                "games": games,
                "mean": float(np.nanmean(cells)) if np.size(cells) else float("nan"),
                "iqm": iqm(cells),
                "iqm_low": lo,
                "iqm_high": hi,
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def iqm_summary(runs: list[Run], by_group: bool = False) -> pd.DataFrame:
    """Per run (or group) and outcome: mean and IQM with a 95% stratified
    bootstrap CI.  A group whose runs played the same games (eval episodes) uses
    rliable's runs × games matrix (``SHARED_GAMES``); otherwise each run's
    episodes are a stratum (``WITHIN_RUNS``).  ``games`` is the matrix's columns,
    or the episodes pooled.  The key column is ``group`` or ``run``."""
    units = groups(runs) if by_group else {r.name: [r] for r in runs}
    frames = []
    for name, members in units.items():
        members = [m for m in members if len(m.episodes)]
        if not members:
            continue
        # memo on the first run, keyed by the exact runs (filtered copies differ)
        key = "iqm:" + repr([(m.name, id(m), len(m.episodes)) for m in members])
        frames.append(memo(members[0], key, lambda n=name, ms=members: _iqm_rows(n, ms)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(
        columns={"name": "group" if by_group else "run"}
    )


def _units(runs: list[Run], by_group: bool) -> dict[str, list[Run]]:
    """Runs (or groups' runs) with episodes, by name."""
    units = groups(runs) if by_group else {r.name: [r] for r in runs}
    kept = {
        n: [m for m in ms if len(m.episodes) and "ante" in m.episodes] for n, ms in units.items()
    }
    return {n: ms for n, ms in kept.items() if ms}


def _memo_key(tag: str, *members: list[Run]) -> str:
    """A ``loader.memo`` key naming the exact runs (filtered copies differ)."""
    return f"{tag}:" + repr([[(m.name, id(m), len(m.episodes)) for m in ms] for ms in members])


def _profile_rows(name: str, runs: list[Run], thresholds: np.ndarray) -> pd.DataFrame:
    matrix = score_matrix(runs, "ante")
    if matrix is not None:
        cells, method, games = matrix.ravel(), SHARED_GAMES, matrix.shape[1]
        batches = _matrix_counts(matrix)
    else:
        samples = _clean([r.episodes["ante"] for r in runs])
        cells, method = np.concatenate(samples), WITHIN_RUNS
        games = len(cells)
        batches = _strata_counts(samples)
    boot = []
    for values, counts in batches:
        reach = (values[:, None] >= thresholds[None, :]).astype(float)  # values × thresholds
        boot.append(counts @ reach / counts.sum(axis=1, keepdims=True))
    lo, hi = _interval(np.concatenate(boot))
    return pd.DataFrame(
        {
            "name": name,
            "ante": thresholds.astype(int),
            "fraction": (cells[:, None] >= thresholds[None, :]).mean(axis=0),
            "fraction low": lo,
            "fraction high": hi,
            "runs": len(runs),
            "games": games,
            "method": method,
        }
    )


def performance_profile(runs: list[Run], by_group: bool = False) -> pd.DataFrame:
    """Performance profile (Agarwal et al. 2021): per run (or group) and ante k,
    the share of games reaching ante >= k, with a 95% stratified bootstrap band
    (resampled as in ``iqm_summary``).  It shows where along the run each agent
    breaks down, which one mean can't.  Thresholds run from 1 to the highest
    ante reached in the selection; the key column is ``group`` or ``run``."""
    units = _units(runs, by_group)
    if not units:
        return pd.DataFrame()
    top = max(int(m.episodes["ante"].max()) for ms in units.values() for m in ms)
    ks = np.arange(1, max(top, 1) + 1, dtype=float)
    frames = [
        memo(ms[0], _memo_key(f"profile{top}", ms), lambda n=n, ms=ms: _profile_rows(n, ms, ks))
        for n, ms in units.items()
    ]
    return pd.concat(frames, ignore_index=True).rename(
        columns={"name": "group" if by_group else "run"}
    )


def _improvement_row(a: list[Run], b: list[Run]) -> dict:
    """P(X > Y) + P(X = Y) / 2 for a game X of ``a`` and Y of ``b``, with a 95%
    bootstrap CI.  Shared games (``score_matrix`` over both): one run each is
    paired game by game (games resampled); groups follow rliable (P per game
    over every seed pair, averaged over games; seeds resampled within each game,
    independently per group).  Otherwise unpaired, each run's episodes a stratum."""
    matrix = score_matrix([*a, *b], "ante")
    if matrix is not None:
        ma, mb = matrix[: len(a)], matrix[len(a) :]
        if len(a) == len(b) == 1:
            g = _beats_pairs(ma[0], mb[0])
            point = g.mean()
            lo, hi = _interval(_boot_means(g, N_BOOT, np.random.default_rng(SEED)))
            method = "paired games"
        else:
            point = _rliable_p(ma[None], mb[None])[0]
            batch = _batch(ma.shape[0] * mb.shape[0] * ma.shape[1], 500_000)
            draws_a = _matrix_draws(ma.shape, N_BOOT, np.random.default_rng(SEED), batch)
            draws_b = _matrix_draws(mb.shape, N_BOOT, np.random.default_rng(SEED + 1), batch)
            t = np.arange(ma.shape[1])
            boot = [
                _rliable_p(ma[ia, t], mb[ib, t]) for ia, ib in zip(draws_a, draws_b, strict=True)
            ]
            lo, hi = _interval(np.concatenate(boot))
            method = SHARED_GAMES
        games = matrix.shape[1]
    else:
        xa, xb = _clean([r.episodes["ante"] for r in a]), _clean([r.episodes["ante"] for r in b])
        if not xa or not xb:
            return {}
        values = np.unique(np.concatenate([*xa, *xb]))
        beats = _beats_between(values, values)

        def dist(xs):
            c = np.zeros(len(values))
            for x in xs:
                v, n = np.unique(x, return_counts=True)
                c[np.searchsorted(values, v)] += n
            return c / c.sum()

        point = float(dist(xa) @ beats @ dist(xb))
        boot = []
        side_a = _strata_counts(xa, seed=SEED, values=values)
        side_b = _strata_counts(xb, seed=SEED + 1, values=values)
        for (_, ca), (_, cb) in zip(side_a, side_b, strict=True):
            pa, pb = ca / ca.sum(1, keepdims=True), cb / cb.sum(1, keepdims=True)
            boot.append(((pa @ beats) * pb).sum(axis=1))
        lo, hi = _interval(np.concatenate(boot))
        method, games = WITHIN_RUNS, sum(map(len, xa))
    return {
        "games": games,
        "p_improvement": float(point),
        "p_low": float(lo),
        "p_high": float(hi),
        "method": method,
    }


def _beats_pairs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elementwise: 1 where ``x > y``, 0.5 where equal, else 0."""
    return (x > y) + 0.5 * (x == y)


def _beats_between(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """``[i, j]`` = 1 if ``vx[i] > vy[j]``, 0.5 if equal, else 0."""
    return (vx[:, None] > vy[None, :]) + 0.5 * (vx[:, None] == vy[None, :])


def _rliable_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """rliable's probability of improvement for ``(batch, runs, tasks)`` arrays:
    per task the share of (a run, b run) pairs where a beats b (ties half),
    averaged over tasks."""
    return _beats_pairs(a[:, :, None, :], b[:, None, :, :]).mean(axis=(1, 2, 3))


def improvement(runs: list[Run], baseline: str, by_group: bool = False) -> pd.DataFrame:
    """Probability of improvement of each run (or group) over ``baseline`` on
    ante reached: the chance one of its games gets further than one of the
    baseline's, counting ties as half (Agarwal et al. 2021).  0.5 means no
    difference; above 0.5, this one tends to get further.  See
    ``_improvement_row`` for pairing and the bootstrap."""
    units = _units(runs, by_group)
    if baseline not in units:
        return pd.DataFrame()
    base = units[baseline]
    key = "group" if by_group else "run"
    rows = []
    for name, ms in units.items():
        if name == baseline:
            continue
        key_ = _memo_key("improvement", ms, base)
        row = memo_value(ms[0], key_, lambda ms=ms: _improvement_row(ms, base))
        if row:
            rows.append({key: name, "baseline": baseline, **row})
    return pd.DataFrame(rows)


def memo_value(run: Run, key: str, build):
    """Like ``loader.memo`` for a value that isn't a frame (returned as is)."""
    if key not in run._memo:
        run._memo[key] = build()
    return run._memo[key]


# ---------------------------------------------------------------------------
# Power: how small a difference in mean ante the selection can detect
# ---------------------------------------------------------------------------

ALPHA = 0.05
POWER = 0.8
MAX_N = 100_000  # give up searching sample sizes beyond this
NC_SURE = 37.0  # noncentrality past which power is taken as 1 (true even at df 1)


def _power(effect: float, se: float, df: float, alpha: float = ALPHA) -> float:
    """Power of a two-sided t-test of a true difference ``effect`` whose
    estimate has standard error ``se`` (noncentral t)."""
    if se <= 0 or not np.isfinite(se):
        return 1.0 if effect else alpha
    crit = sps.t.ppf(1 - alpha / 2, df)
    nc = abs(effect) / se
    if nc > NC_SURE:  # scipy's nct gives NaN far out, where power is ~1 anyway
        return 1.0
    upper = sps.nct.sf(crit, df, nc)
    lower = sps.nct.cdf(-crit, df, nc)  # wrong-sign rejections: tiny, NaN far out
    return float(min(1.0, np.nan_to_num(upper, nan=1.0) + np.nan_to_num(lower, nan=0.0)))


def _mde(se: float, df: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest true difference detected with ``power``, given the estimate's SE."""
    if not np.isfinite(se) or df < 1:
        return float("nan")
    if se == 0:
        return 0.0
    return float(brentq(lambda d: _power(d, se, df, alpha) - power, 0.0, se * NC_SURE))


def _n_needed(target: float, se_of_n, df_of_n, start: int = 2) -> int | None:
    """Smallest n >= start whose SE (``se_of_n(n)``) gives ``POWER`` for ``target``;
    None if not within ``MAX_N``.  Doubling then bisection: power rises with n."""
    if not target > 0:
        return None

    def ok(n: int) -> bool:
        return _power(target, se_of_n(n), df_of_n(n)) >= POWER

    hi = start
    while not ok(hi):
        if hi >= MAX_N:
            return None
        hi = min(hi * 2, MAX_N)
    lo = max(start, hi // 2)
    if ok(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def power_runs(runs: list[Run], baseline: str, target: float = 0.25) -> pd.DataFrame:
    """Each run against ``baseline``: the smallest difference in mean ante a
    two-sided test at ``ALPHA`` detects with ``POWER`` (``mde``), and the games
    per run needed to detect ``target``.  Paired (per-game differences, df n-1)
    when the runs share games, else two-sample (pooled SD, df 2n-2)."""
    base = next((r for r in runs if r.name == baseline), None)
    if base is None:
        return pd.DataFrame()
    b = base.episodes["ante"].to_numpy(dtype=float)
    rows = []
    for r in runs:
        if r.name == baseline:
            continue
        a = r.episodes["ante"].to_numpy(dtype=float)
        pairs = paired_ante(r, base)
        paired = len(pairs) >= MIN_SHARED * min(len(a), len(b)) and len(pairs) > 1
        if paired:
            d = (pairs["ante"] - pairs["ante_baseline"]).to_numpy(dtype=float)
            n, sd = len(d), float(d.std(ddof=1))

            def se(k, sd=sd):
                return sd / np.sqrt(k)

            def df(k):
                return k - 1
        else:
            if len(a) < 2 or len(b) < 2:
                continue
            n = min(len(a), len(b))
            pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
                len(a) + len(b) - 2
            )
            sd = float(np.sqrt(pooled))

            def se(k, sd=sd):
                return sd * np.sqrt(2 / k)

            def df(k):
                return 2 * k - 2

        rows.append(
            {
                "run": r.name,
                "baseline": baseline,
                "paired": paired,
                "games": n,
                "sd": sd,
                "mde": _mde(se(n), df(n)),
                "games_needed": _n_needed(target, se, df),
            }
        )
    return pd.DataFrame(rows)


def variance_components(runs: list[Run], col: str = "ante") -> dict[str, float]:
    """One-way random-effects split of a group's ``col`` into game-to-game
    variance (pooled within runs) and seed-to-seed variance (of the runs' true
    means: var of run means minus the share explained by game noise, >= 0).
    ``games``: the mean games per run."""
    xs = [r.episodes[col].dropna().to_numpy(dtype=float) for r in runs]
    xs = [x for x in xs if len(x) > 1]
    if not xs:
        return {"seeds": 0, "games": 0, "var_game": float("nan"), "var_seed": float("nan")}
    n = float(np.mean([len(x) for x in xs]))
    var_game = float(sum((len(x) - 1) * x.var(ddof=1) for x in xs) / sum(len(x) - 1 for x in xs))
    means = np.array([x.mean() for x in xs])
    var_seed = max(0.0, float(means.var(ddof=1)) - var_game / n) if len(xs) > 1 else float("nan")
    return {"seeds": len(xs), "games": n, "var_game": var_game, "var_seed": var_seed}


def power_groups(runs: list[Run], target: float = 0.25) -> pd.DataFrame:
    """Per group, two groups like it compared on the mean of their runs' mean
    ante (as ``compare_groups`` does): the smallest detectable difference with
    the group's seeds and games per run (``mde``), the seeds per group needed to
    detect ``target`` with that many games, and with unlimited games
    (``seeds_needed_any_games``: only seed-to-seed variation left; None when
    that variation is estimated as 0, i.e. too small to see with these games)."""
    rows = []
    for group, members in groups(runs).items():
        vc = variance_components(members)
        k, n = vc["seeds"], vc["games"]
        if k < 2 or not np.isfinite(vc["var_seed"]):
            continue

        def se(seeds, games=n, vc=vc):
            return np.sqrt(2 * (vc["var_seed"] + vc["var_game"] / games) / seeds)

        def df(seeds):
            return 2 * seeds - 2

        def se_inf(seeds, vc=vc):
            return np.sqrt(2 * vc["var_seed"] / seeds)

        rows.append(
            {
                "group": group,
                "seeds": k,
                "games_per_seed": round(n),
                "sd_game": np.sqrt(vc["var_game"]),
                "sd_seed": np.sqrt(vc["var_seed"]),
                "mde": _mde(se(k), df(k)),
                "seeds_needed": _n_needed(target, se, df),
                # seed variation estimated as 0: unlimited games would look free
                "seeds_needed_any_games": _n_needed(target, se_inf, df)
                if vc["var_seed"] > 0
                else None,
            }
        )
    return pd.DataFrame(rows)
