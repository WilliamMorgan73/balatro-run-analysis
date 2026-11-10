"""The statistics balatro-shop-rl's experiment scripts report (``shop_rl.report_stats``),
ported so the dashboard prints the same numbers as ``endless_report.py``,
``deck_transfer.py`` and ``stickers.py``.

Per group of runs (seeds): win % ± seed SD, mean final round, mean ante, and the IQM
ante over seeds × games with a stratified bootstrap 95% CI (2000 resamples, numpy
seed 0).  Per pair A - B over the seeds both have: the paired difference of
per-seed means with a Student-t CI and p (the t CDF integrated as the script does),
seeds better, and the probability of improvement over the same test games.

The bootstrap draws depend on the order of seeds (sorted) and of games (as in
episodes.csv), both kept as the scripts read them.  The dashboard's own study tables
(``studies``) use rliable's estimators instead; this module is only for matching the
scripts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from balatro_analysis.loader import Run


def _seed_key(run: Run):
    seed = run.config.get("seed", run.name)
    return (0, seed, "") if isinstance(seed, int | float) else (1, 0, str(seed))


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50%."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = math.floor(n * 0.25), math.ceil(n * 0.75)
    return float(x[lo:hi].mean()) if hi > lo else float("nan")


def iqm_ci(
    per_seed: list[np.ndarray], reps: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """IQM over seeds × games and a 95% stratified bootstrap CI (games resampled
    within each seed)."""
    rng = np.random.default_rng(seed)
    point = iqm(np.concatenate(per_seed))
    boots = [
        iqm(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in per_seed]))
        for _ in range(reps)
    ]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _t_cdf(t: float, df: int) -> float:
    """Student's t CDF by integrating its density (Simpson's rule), as the script."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    x = np.linspace(0.0, abs(t), 4001)
    y = c * (1 + x * x / df) ** (-(df + 1) / 2)
    h = x[1] - x[0] if len(x) > 1 else 0.0
    half = float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))
    return 0.5 + half if t >= 0 else 0.5 - half


def _t_crit(df: int, conf: float = 0.95) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, df) < 0.5 + conf / 2 else (lo, mid)
    return (lo + hi) / 2


def paired(diffs: Sequence[float]) -> dict[str, float]:
    """Mean of paired differences, its 95% t CI, two-sided p, seeds better."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        nan = float("nan")
        return {"n": n, "mean": mean, "lo": nan, "hi": nan, "p": nan, "better": int((d > 0).sum())}
    se = float(d.std(ddof=1) / math.sqrt(n))
    t = mean / se if se > 0 else math.copysign(math.inf, mean)
    crit = _t_crit(n - 1)
    p = 2 * (1 - _t_cdf(abs(t), n - 1)) if math.isfinite(t) else 0.0
    return {
        "n": n,
        "mean": mean,
        "lo": mean - crit * se,
        "hi": mean + crit * se,
        "p": p,
        "better": int((d > 0).sum()),
    }


def prob_improvement(
    a: list[np.ndarray], b: list[np.ndarray], reps: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """P(A scores more than B on the same test game), ties half, over paired seeds ×
    games (same seed, same game index); 95% CI by resampling games within each seed.
    A seed whose two sides played different numbers of games is left out."""
    wins = [(x > y) + 0.5 * (x == y) for x, y in zip(a, b, strict=True) if len(x) == len(y)]
    if not wins:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [
        np.concatenate([w[rng.integers(0, len(w), len(w))] for w in wins]).mean()
        for _ in range(reps)
    ]
    point = float(np.concatenate(wins).mean())
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def group_summary(runs: list[Run], cap: int | None = None) -> dict[str, Any] | None:
    """A group's row (its runs = seeds, over the episodes they hold): None without
    games.  ``ante`` is the highest ante reached, capped at ``cap`` if given."""
    per_seed, wins, rounds = {}, [], []
    for r in sorted(runs, key=_seed_key):
        e = r.episodes
        if e.empty:
            continue
        a = e["ante"].to_numpy(dtype=float)
        per_seed[r.config.get("seed", r.name)] = np.minimum(a, cap) if cap else a
        wins.append(100 * float(np.mean(e["won"].to_numpy(dtype=float))))
        rounds.append(float(np.mean(e["final_round"].to_numpy(dtype=float))))
    if not per_seed:
        return None
    point, lo, hi = iqm_ci(list(per_seed.values()))
    return {
        "seeds": len(per_seed),
        "win": float(np.mean(wins)),
        "win_sd": float(np.std(wins, ddof=1)) if len(wins) > 1 else 0.0,
        "final_round": float(np.mean(rounds)),
        "ante": float(np.mean([v.mean() for v in per_seed.values()])),
        "iqm": point,
        "iqm_lo": lo,
        "iqm_hi": hi,
        "per_seed": per_seed,
        "win_per_seed": dict(zip(per_seed, wins, strict=True)),
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """A - B over the seeds both have: ante, win rate, probability of improvement."""
    seeds = sorted(set(a["per_seed"]) & set(b["per_seed"]), key=lambda s: (str(type(s)), s))
    if not seeds:
        return {"n": 0}
    pa, pb = [a["per_seed"][s] for s in seeds], [b["per_seed"][s] for s in seeds]
    out = {
        "seeds": seeds,
        "ante": paired([x.mean() - y.mean() for x, y in zip(pa, pb, strict=True)]),
        "win": paired([a["win_per_seed"][s] - b["win_per_seed"][s] for s in seeds]),
    }
    out["poi"] = prob_improvement(pa, pb)
    return out


GROUP_COLUMNS = ["arm", "seeds", "win %", "final round", "ante", "IQM ante [95% CI]"]
COMPARE_COLUMNS = [
    "A - B",
    "ante (paired)",
    "p",
    "seeds better",
    "win % (paired)",
    "P(improvement)",
]


def group_row(name: str, g: dict[str, Any] | None) -> list[str]:
    """The script's ``fmt_group`` as table cells."""
    if g is None:
        return [name, "0", "-", "-", "-", "-"]
    return [
        name,
        str(g["seeds"]),
        f"{g['win']:.1f} ± {g['win_sd']:.1f}",
        f"{g['final_round']:.2f}",
        f"{g['ante']:.2f}",
        f"{g['iqm']:.2f} [{g['iqm_lo']:.2f}, {g['iqm_hi']:.2f}]",
    ]


def compare_row(name: str, c: dict[str, Any]) -> list[str]:
    """The script's ``fmt_compare`` as table cells."""
    if not c.get("seeds"):
        return [name, "no paired seeds", "", "", "", ""]
    a, w, (p, lo, hi) = c["ante"], c["win"], c["poi"]
    return [
        name,
        f"{a['mean']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}]",
        f"{a['p']:.3g}",
        f"{a['better']}/{a['n']}",
        f"{w['mean']:+.1f} [{w['lo']:+.1f}, {w['hi']:+.1f}]",
        f"{p:.3f} [{lo:.3f}, {hi:.3f}]",
    ]


def markdown(columns: list[str], rows: list[list[str]]) -> str:
    """Rows as the scripts print them (a Markdown table)."""
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([head, *("| " + " | ".join(r) + " |" for r in rows)])
