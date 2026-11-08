"""Runs listed by a manifest, and the named studies in it.

balatro-shop-rl's ``scripts/run_manifest.py`` writes ``runs/manifest.json``: every run
in its runs folder and archives (so one group can span both), with each run's splits
and a test summary, the studies (groups to compare, and pairs "A - B" matched by
training seed), and problems such as missing seeds.  ``load_manifest`` reads it;
``group_table`` and ``pair_table`` are a study's results, on whichever episodes the
runs given hold (the dashboard passes the test games by default).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from balatro_analysis import stats
from balatro_analysis.loader import Run

MANIFEST = "manifest.json"
SUPPORTED_VERSION = 1


@dataclass
class Manifest:
    path: Path  # the manifest file
    base: Path  # what run paths are relative to
    runs: dict[str, dict]  # name -> the run's entry
    groups: dict[str, list[str]] = field(default_factory=dict)
    studies: dict[str, dict] = field(default_factory=dict)
    problems: list[dict] = field(default_factory=list)
    generated: str = ""

    def run_dir(self, name: str) -> Path:
        return self.base / self.runs[name]["path"]

    def study_runs(self, study: str) -> list[str]:
        """The runs of the study's groups, group by group, in seed order."""
        return [r for g in self.studies[study].get("groups", []) for r in self.groups.get(g, [])]

    def study_problems(self, study: str) -> list[str]:
        """What's wrong with the study's runs: its own problems (missing seeds or
        groups), its runs' (no test set) and its groups' (several commits)."""
        groups = set(self.studies[study].get("groups", []))
        runs = set(self.study_runs(study))
        return [
            p["message"]
            for p in self.problems
            if p.get("study") == study
            or p.get("run") in runs
            or (p.get("group") in groups and "study" not in p)
        ]


def load_manifest(path: str | Path) -> Manifest:
    """Read a manifest; ValueError if it isn't one this code understands."""
    path = Path(path).expanduser()
    data = json.loads(path.read_text())
    if data.get("manifest_version") != SUPPORTED_VERSION:
        raise ValueError(f"{path}: manifest_version {data.get('manifest_version')!r}")
    return Manifest(
        path=path,
        base=(path.parent / data.get("base", ".")).resolve(),
        runs={r["name"]: r for r in data["runs"]},
        groups=data.get("groups", {}),
        studies=data.get("studies", {}),
        problems=data.get("problems", []),
        generated=data.get("generated", ""),
    )


def where(run: Run, col: str, value) -> Run:
    """``run`` with only the episodes whose ``col`` is ``value`` (episodes only:
    what the study tables read)."""
    e = run.episodes
    return Run(run.name, run.path, e[e[col] == value].reset_index(drop=True), run.config)


def _seed(run: Run):
    return run.config.get("seed", run.name)


def group_table(runs: list[Run], groups: list[str]) -> pd.DataFrame:
    """One row per group (in ``groups`` order; groups without runs left out): seeds,
    win rate (% , mean and SD over seeds), mean final round, mean ante (over seeds'
    means) and the IQM of ante with its stratified-bootstrap 95% CI."""
    by_group = stats.groups(runs)
    rows = []
    for g in groups:
        members = [r for r in by_group.get(g, []) if len(r.episodes)]
        if not members:
            continue
        win = np.array([r.episodes["won"].astype(float).mean() * 100 for r in members])
        ante = np.array([r.episodes["ante"].astype(float).mean() for r in members])
        final = [
            r.episodes["final_round"].astype(float).mean()
            for r in members
            if "final_round" in r.episodes
        ]
        iq = stats.iqm_summary(members, by_group=True)
        iq = iq[iq["outcome"] == stats.IQM_OUTCOMES["ante"]]
        rows.append(
            {
                "group": g,
                "seeds": len(members),
                "games": int(sum(len(r.episodes) for r in members)),
                "win_pct": win.mean(),
                "win_pct_sd": win.std(ddof=1) if len(win) > 1 else np.nan,
                "final_round": np.mean(final) if final else np.nan,
                "ante": ante.mean(),
                "ante_iqm": iq["iqm"].iloc[0] if len(iq) else np.nan,
                "ante_iqm_low": iq["iqm_low"].iloc[0] if len(iq) else np.nan,
                "ante_iqm_high": iq["iqm_high"].iloc[0] if len(iq) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_by_seed(a: list[Run], b: list[Run], col: str = "ante", scale: float = 1.0) -> dict:
    """A - B on the mean of ``col``, matched by training seed (same seed: same
    training games in balatro-shop-rl, so game luck cancels): the mean of the
    per-seed differences with a Student-t 95% CI and two-sided p, and how many
    seeds A is ahead in.  Seeds only one side has are left out (``n`` counts the
    pairs).  ``scale``: e.g. 100 for a win rate in points."""
    ma = {_seed(r): r.episodes[col].astype(float).mean() for r in a if len(r.episodes)}
    mb = {_seed(r): r.episodes[col].astype(float).mean() for r in b if len(r.episodes)}
    common = sorted(set(ma) & set(mb), key=str)
    d = np.array([ma[s] - mb[s] for s in common]) * scale
    out = {
        "n": len(d),
        "unpaired": sorted(set(ma) ^ set(mb), key=str),
        "diff": d.mean() if len(d) else np.nan,
    }
    if len(d) > 1:
        lo, hi = stats.t_ci(d)
        sd = d.std(ddof=1)
        p = float(sps.ttest_1samp(d, 0).pvalue) if sd > 0 else (0.0 if d.mean() else 1.0)
        out |= {"low": lo, "high": hi, "p": p}
    else:
        out |= {"low": np.nan, "high": np.nan, "p": np.nan}
    out["better"] = int((d > 0).sum())
    return out


def pair_table(runs: list[Run], pairs: list[list[str]]) -> pd.DataFrame:
    """One row per pair ``[A, B]`` (groups) with runs on both sides: A - B in ante
    and win rate (points), paired by seed (``paired_by_seed``), and the probability
    that a game of A gets further than one of B (``stats.improvement``, rliable's
    over seeds on shared games)."""
    by_group = stats.groups(runs)
    rows = []
    for a, b in pairs:
        ra, rb = by_group.get(a, []), by_group.get(b, [])
        if not ra or not rb:
            continue
        ante = paired_by_seed(ra, rb, "ante")
        win = paired_by_seed(ra, rb, "won", scale=100)
        poi = stats.improvement([*ra, *rb], b, by_group=True)
        poi = poi[poi["group"] == a] if len(poi) else poi
        rows.append(
            {
                "a": a,
                "b": b,
                "pairs": ante["n"],
                "unpaired_seeds": ", ".join(map(str, ante["unpaired"])),
                "ante_diff": ante["diff"],
                "ante_low": ante["low"],
                "ante_high": ante["high"],
                "p": ante["p"],
                "a_better": f"{ante['better']}/{ante['n']}",
                "win_diff": win["diff"],
                "win_low": win["low"],
                "win_high": win["high"],
                "p_improvement": poi["p_improvement"].iloc[0] if len(poi) else np.nan,
                "p_low": poi["p_low"].iloc[0] if len(poi) else np.nan,
                "p_high": poi["p_high"].iloc[0] if len(poi) else np.nan,
            }
        )
    return pd.DataFrame(rows)
