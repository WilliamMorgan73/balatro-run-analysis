"""The Order's offers by xMult jokers held, as part 1 of balatro-shop-rl's
scripts/order_context.py."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import order
from balatro_analysis.loader import Run

V17 = Path(__file__).parent / "fixtures" / "v17"
APP = str(Path(__file__).parent.parent / "app.py")

OFFERS = [  # key, buy_legal, bought, jokers held, q_buy_adv
    ("j_order", True, True, ["j_baron"], 0.5),
    ("j_order", True, False, ["j_baron", "j_duo"], None),
    ("j_order", True, False, [], -0.25),
    ("j_order", False, True, [], 1.0),  # not legal: left out
    ("j_egg", True, True, [], 1.0),  # not The Order
    ("j_order", True, True, ["j_order", "j_joker"], 0.0),  # The Order itself doesn't count
]


def _offers() -> list[dict]:
    keys = ("key", "buy_legal", "bought", "jokers", "q_buy_adv")
    return [dict(zip(keys, o, strict=True)) for o in OFFERS]


def _run(name: str, seed: int, rows: list[dict]) -> Run:
    e = pd.DataFrame({"episode": [0], "ante": [1], "won": [0]})
    return Run(name, Path(name), e, {"group": "g", "seed": seed}, offers=pd.DataFrame(rows))


def test_from_logs():
    got = order.from_logs(_run("a", 0, _offers())).set_index("bucket")
    assert got.at["all", "n"] == 4
    assert got.at["xmult 0", "n"] == 2 and got.at["xmult 0", "rate"] == 0.5
    assert got.at["xmult 0", "adv"] == pytest.approx(-0.125)
    assert got.at["xmult 1", "n"] == 1 and got.at["xmult 1", "rate"] == 1.0
    assert got.at["xmult 2+", "n"] == 1 and np.isnan(got.at["xmult 2+", "adv"])
    assert got.at["hand-type xmult yes", "n"] == 1
    assert len(order.XMULT) == 35 and order.HAND_XMULT <= order.XMULT


def test_order_table_over_seeds():
    a = _run("a", 0, _offers())
    b = _run("b", 1, [dict(_offers()[2], bought=True), dict(_offers()[0], bought=False)])
    t, diff = order.order_table([a, b])
    t = t.set_index("held")
    assert t.at["all", "offers"] == 6
    assert t.at["xmult 0", "buy rate"] == pytest.approx((0.5 + 1.0) / 2)
    assert diff["n"] == 2 and diff["mean"] == pytest.approx(((1 - 0.5) + (0 - 1)) / 2)
    assert order.order_table([a])[1] is None  # one seed: no paired test


def test_app_order_panel(tmp_path):
    root = tmp_path / "runs"
    shutil.copytree(V17, root)
    offers = root / "run_v17" / "test" / "offers.jsonl"
    offers.write_text("".join(json.dumps(o | {"episode": 0, "step": 0}) + "\n" for o in _offers()))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(root)
    at.session_state["main_tab"] = "Economy"
    at.session_state["econ_tab"] = "Shop offers"
    at.run()
    assert not at.exception
    assert any("The Order" in h.value for h in at.subheader)
