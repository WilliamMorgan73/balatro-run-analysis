"""Endless mode (log_version 17): antes past 8, the capped like-for-like view and
the past-8 numbers, as balatro-shop-rl's scripts/endless_report.py."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from test_test_split import _app

from balatro_analysis import endless, metrics
from balatro_analysis.loader import Run, load_run

V17 = Path(__file__).parent / "fixtures" / "v17"


def _run(name: str, antes: list[int], won: list[int], seed: int = 0, end: bool = True) -> Run:
    e = pd.DataFrame(
        {
            "episode": range(len(antes)),
            "ante": antes,
            "won": won,
            "return": 0.0,
            "deck": "b_red",
            "stake": 1,
            "seed": [f"TEST_{i}" for i in range(len(antes))],
        }
    )
    return Run(name, Path(name), e, {"endless": end, "group": "endless", "seed": seed})


def test_endless_ante_past_eight_loads(tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(V17 / "run_v17", run_dir)
    e = pd.read_csv(run_dir / "test" / "episodes.csv")
    e.loc[0, ["ante", "max_ante", "won"]] = [11, 11, 1]  # lost in ante 11 after a win
    e.to_csv(run_dir / "test" / "episodes.csv", index=False)
    test = load_run(run_dir).test
    assert test.episodes.loc[0, "ante"] == 11 and test.episodes.loc[0, "won"] == 1
    assert endless.is_endless(test)


def test_capped():
    run = _run("a", [3, 9, 12], [0, 1, 1])
    capped = endless.capped(run)
    assert list(capped.episodes["ante"]) == [3, 8, 8]
    assert list(run.episodes["ante"]) == [3, 9, 12]
    assert endless.capped(run) is capped  # kept, so what is memoised on it is reused


def test_past_eight_as_endless_report():
    runs = [_run("a", [3, 9, 12, 8], [0, 1, 1, 1]), _run("b", [10, 2], [1, 0], seed=1)]
    res = endless.past_eight(runs)
    assert (res.games, res.won) == (6, 4)
    assert res.counts == {8: 1, 9: 1, 10: 1, 12: 1}
    assert res.mean_win_ante == np.mean([np.mean([9, 12, 8]), 10])
    assert (res.seed_low, res.seed_high) == (29 / 3, 10)
    assert endless.past_eight([_run("c", [3], [0])]) is None


def test_learning_curve_is_decimated():
    e = pd.DataFrame({"episode": range(10_001), "ante": 1, "won": 0, "return": 0.0})
    run = Run("long", Path("long"), e, {})
    curve = metrics.learning_curve([run], window=50)
    assert len(curve) <= metrics.CURVE_POINTS + 1
    assert curve["episode"].iloc[-1] == 10_000
    assert len(metrics.learning_curve([run], window=50, max_points=0)) == 10_001


def test_app_shows_past_eight(tmp_path):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    e = pd.read_csv(root / "run_v17" / "test" / "episodes.csv")
    e.loc[0, ["ante", "max_ante", "won"]] = [11, 10, 1]
    e.to_csv(root / "run_v17" / "test" / "episodes.csv", index=False)
    at = _app(root, "Test", "Overview")
    assert not at.exception
    assert any("past ante 8" in m.value for m in at.markdown)
