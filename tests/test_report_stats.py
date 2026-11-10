"""The port of balatro-shop-rl's shop_rl.report_stats: the expected values below were
computed by that module on the same arrays, so the dashboard prints the scripts'
numbers (bootstrap CIs included)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from balatro_analysis import report_stats as rs
from balatro_analysis.loader import Run

A = [
    [1, 7, 6, 4, 4, 7, 1, 6, 2, 1, 5, 8, 6, 7, 6, 7, 5, 2, 7, 4],
    [5, 3, 2, 8, 7, 6, 4, 7, 5, 4, 4, 2, 1, 5, 8, 1, 7, 7, 3, 6],
    [2, 7, 6, 3, 1, 8, 4, 8, 6, 7, 7, 2, 3, 4, 4, 1, 5, 2, 6, 6],
]
B = [
    [8, 6, 3, 8, 4, 3, 8, 3, 1, 4, 7, 2, 4, 2, 6, 4, 3, 2, 5, 6],
    [8, 4, 2, 7, 6, 6, 1, 3, 7, 7, 4, 7, 7, 4, 8, 3, 2, 6, 6, 2],
    [7, 2, 7, 1, 7, 7, 7, 6, 4, 6, 3, 7, 5, 4, 5, 5, 1, 2, 2, 1],
]


def _arrays(xs):
    return [np.array(x, dtype=float) for x in xs]


def test_iqm_ci_as_the_script():
    assert rs.iqm_ci(_arrays(A)) == pytest.approx(
        (4.966666666666667, 4.066666666666666, 5.666666666666667), abs=1e-12
    )


def test_paired_as_the_script():
    got = rs.paired([0.5, 0.2, 0.9, -0.1])
    assert got == pytest.approx(
        {
            "n": 4,
            "mean": 0.375,
            "lo": -0.3047708287903993,
            "hi": 1.0547708287903994,
            "p": 0.1774175633522026,
            "better": 3,
        },
        abs=1e-12,
    )
    assert np.isnan(rs.paired([1.0])["p"])
    assert rs.paired([1.0, 1.0])["p"] == 0.0  # no spread: t infinite


def test_prob_improvement_as_the_script():
    assert rs.prob_improvement(_arrays(A), _arrays(B)) == pytest.approx(
        (0.5583333333333333, 0.44166666666666665, 0.6666666666666666), abs=1e-12
    )


def _run(seed: int, antes: list[int]) -> Run:
    e = pd.DataFrame(
        {
            "episode": range(len(antes)),
            "ante": antes,
            "won": [int(a >= 8) for a in antes],
            "final_round": [3 * a for a in antes],
        }
    )
    return Run(f"g_s{seed}", Path("g"), e, {"group": "g", "seed": seed})


def test_group_summary_and_compare():
    a = [_run(s, x) for s, x in reversed(list(enumerate(A)))]  # seed order restored
    b = [_run(s, x) for s, x in enumerate(B)]
    ga, gb = rs.group_summary(a), rs.group_summary(b)
    assert list(ga["per_seed"]) == [0, 1, 2]
    assert ga["iqm"] == pytest.approx(4.966666666666667)
    assert ga["ante"] == pytest.approx(np.mean([np.mean(x) for x in A]))
    c = rs.compare(ga, gb)
    assert c["seeds"] == [0, 1, 2]
    assert c["poi"] == pytest.approx((0.5583333333333333, 0.44166666666666665, 0.6666666666666666))
    row = rs.compare_row("a - b", c)
    assert row[0] == "a - b" and row[3] == f"{c['ante']['better']}/3"
    capped = rs.group_summary([_run(0, [9, 12, 3])], cap=8)
    assert capped["ante"] == pytest.approx((8 + 8 + 3) / 3)
    assert rs.group_summary([_run(0, [])]) is None
    assert rs.group_row("x", None) == ["x", "0", "-", "-", "-", "-"]
