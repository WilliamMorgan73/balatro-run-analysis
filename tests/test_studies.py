"""Runs from balatro-shop-rl's run manifest, and its studies' tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import _with_test
from streamlit.testing.v1 import AppTest

from balatro_analysis import latex, studies
from balatro_analysis.loader import Run, load_run

APP = str(Path(__file__).parent.parent / "app.py")


def _run(name, group, seed, antes, won=None):
    e = pd.DataFrame(
        {
            "episode": range(len(antes)),
            "seed": [f"TEST_{i}" for i in range(len(antes))],
            "deck": ["b_red", "b_checkered"] * (len(antes) // 2) + ["b_red"] * (len(antes) % 2),
            "stake": 1,
            "ante": antes,
            "won": won if won is not None else [0] * len(antes),
            "final_round": [3 * a for a in antes],
        }
    )
    return Run(name, Path(name), e, {"group": group, "seed": seed})


@pytest.fixture
def two_groups():
    on = [_run(f"on_s{s}", "on", s, [4 + s % 2, 5, 6, 3]) for s in range(3)]
    off = [_run(f"off_s{s}", "off", s, [4, 5, 5, 3]) for s in (0, 2)]  # seed 1 missing
    return on + off


def test_paired_by_seed_drops_unpaired_seeds(two_groups):
    on, off = two_groups[:3], two_groups[3:]
    d = studies.paired_by_seed(on, off)
    assert d["n"] == 2 and d["unpaired"] == [1]
    # seed 0: on mean 4.5 vs off 4.25; seed 2: same
    assert d["diff"] == pytest.approx(0.25)
    assert d["better"] == 2
    assert d["p"] == 0.0  # identical differences: no spread
    win = studies.paired_by_seed(on, off, "won", scale=100)
    assert win["diff"] == 0


def test_paired_by_seed_t_interval():
    a = [_run(f"a{s}", "a", s, [5 + d]) for s, d in enumerate([0.1, 0.3, 0.2])]
    b = [_run(f"b{s}", "b", s, [5]) for s in range(3)]
    d = studies.paired_by_seed(a, b)
    assert d["diff"] == pytest.approx(0.2)
    half = 4.303 * 0.1 / np.sqrt(3)  # t(2) 97.5% x SD / sqrt(n)
    assert d["low"] == pytest.approx(0.2 - half, abs=1e-3)
    assert d["high"] == pytest.approx(0.2 + half, abs=1e-3)
    assert d["p"] == pytest.approx(0.0742, abs=1e-3)  # t = 3.46 on 2 df


def test_group_and_pair_tables(two_groups):
    g = studies.group_table(two_groups, ["off", "on", "absent"])
    assert list(g["group"]) == ["off", "on"]
    assert list(g["seeds"]) == [2, 3]
    assert g.loc[g["group"] == "off", "ante"].item() == pytest.approx(4.25)
    assert (g["ante_iqm_low"] <= g["ante_iqm"]).all() and (
        g["ante_iqm"] <= g["ante_iqm_high"]
    ).all()
    p = studies.pair_table(two_groups, [["on", "off"], ["on", "absent"]])
    assert len(p) == 1
    row = p.iloc[0]
    assert (row["a"], row["b"], row["pairs"], row["a_better"]) == ("on", "off", 2, "2/2")
    assert row["unpaired_seeds"] == "1"
    assert 0.5 <= row["p_improvement"] <= 1
    assert "\\toprule" in latex.study_groups(g, "Groups", "x-groups")
    assert "on $-$ off" in latex.study_pairs(p, "Pairs", "x-pairs")


def test_where_keeps_matching_episodes(two_groups):
    r = studies.where(two_groups[0], "deck", "b_checkered")
    assert set(r.episodes["deck"]) == {"b_checkered"} and len(r.episodes) == 2


# --- manifest ----------------------------------------------------------------


def _manifest(path: Path, runs_root: Path, runs: dict[str, tuple[str, int]], **extra) -> Path:
    """A manifest (balatro-shop-rl's format, version 1) for ``runs``: name -> (group,
    seed), all under ``runs_root``."""
    groups: dict[str, list[str]] = {}
    for name, (group, _) in runs.items():
        groups.setdefault(group, []).append(name)
    data = {
        "manifest_version": 1,
        "base": ".",
        "runs": [
            {
                "name": n,
                "path": str((runs_root / n).relative_to(path.parent)),
                "group": g,
                "seed": s,
            }
            for n, (g, s) in runs.items()
        ],
        "groups": groups,
        "studies": {},
        "problems": [],
        **extra,
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def study_dir(group_runs_dir):
    """group_runs_dir (a: 3 seeds, b: 2) with test games, a manifest beside it naming
    a study comparing a and b, and a problem for it."""
    for d in sorted(group_runs_dir.iterdir()):
        _with_test(d, ante_shift=3 if d.name.startswith("b") else 0)
    runs = {f"a_s{s}": ("a", s) for s in range(3)} | {f"b_s{s}": ("b", s) for s in range(2)}
    _manifest(
        group_runs_dir / "manifest.json",
        group_runs_dir,
        runs,
        studies={
            "ab": {
                "title": "A against B",
                "groups": ["a", "b"],
                "pairs": [["b", "a"]],
                "split_by": "deck",
            }
        },
        problems=[
            {"kind": "missing_seeds", "study": "ab", "group": "b", "message": "b: seed 2 missing"}
        ],
    )
    return group_runs_dir


def test_load_manifest(study_dir):
    m = studies.load_manifest(study_dir / "manifest.json")
    assert m.run_dir("a_s0") == (study_dir / "a_s0").resolve()
    assert m.study_runs("ab") == ["a_s0", "a_s1", "a_s2", "b_s0", "b_s1"]
    assert m.study_problems("ab") == ["b: seed 2 missing"]
    assert load_run(m.run_dir("b_s1")).test is not None


def test_unknown_manifest_version(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"manifest_version": 99, "runs": []}))
    with pytest.raises(ValueError):
        studies.load_manifest(tmp_path / "m.json")


def _app(study_dir: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["manifest_path"] = str(study_dir / "manifest.json")
    at.run()
    return at


def test_app_lists_runs_from_the_manifest(study_dir):
    at = _app(study_dir)
    assert not at.exception
    assert sorted(at.session_state["picked_runs"]) == ["a_s0", "a_s1", "a_s2", "b_s0", "b_s1"]
    assert at.sidebar.segmented_control(key="phase").value == "Test"


def test_app_study_picks_runs_and_shows_tables(study_dir):
    at = _app(study_dir)
    at.sidebar.multiselect(key="picked_runs").set_value(["a_s0"]).run()
    at.sidebar.selectbox(key="study").set_value("ab").run()
    assert not at.exception
    assert sorted(at.session_state["picked_runs"]) == ["a_s0", "a_s1", "a_s2", "b_s0", "b_s1"]
    assert at.session_state["compare_by"] == "Groups"
    assert any("b: seed 2 missing" in c.value for c in at.sidebar.caption)
    assert any(h.value == "A against B" for h in at.subheader)
    frames = [d.value for d in at.dataframe]
    pairs = next(f for f in frames if "Seed pairs" in f.columns)
    assert pairs.iloc[0]["Seed pairs"] == 2
    assert pairs.iloc[0]["Δ ante (A - B)"] == pytest.approx(3.0)  # b's test games: +3 ante
    assert any(e.label.startswith("Deck") for e in at.expander)
    assert not any("Not the test games" in w.value for w in at.warning)


def test_app_without_manifest_scans_the_folder(group_runs_dir):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(group_runs_dir)
    at.run()
    assert not at.exception
    assert not [s for s in at.sidebar.selectbox if s.label == "Study"]


def test_app_unreadable_manifest_falls_back(group_runs_dir, tmp_path):
    (tmp_path / "bad.json").write_text("{")
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(group_runs_dir)
    at.session_state["manifest_path"] = str(tmp_path / "bad.json")
    at.run()
    assert not at.exception
    assert any("Couldn't read the run manifest" in w.value for w in at.sidebar.warning)
    assert len(at.session_state["picked_runs"]) == 5


def test_study_caps_endless_antes_at_8(group_runs_dir):
    """b's runs are endless and go past ante 8: a study mixing them with a's normal
    runs caps ante at 8 by default, so both compare like for like."""
    for d in sorted(group_runs_dir.iterdir()):
        b = d.name.startswith("b")
        _with_test(d, ante_shift=10 if b else 0)
        if b:
            config = json.loads((d / "config.json").read_text())
            (d / "config.json").write_text(json.dumps(config | {"endless": True}))
    runs = {f"a_s{s}": ("a", s) for s in range(3)} | {f"b_s{s}": ("b", s) for s in range(2)}
    _manifest(
        group_runs_dir / "manifest.json",
        group_runs_dir,
        runs,
        studies={"ab": {"groups": ["a", "b"], "pairs": [["b", "a"]]}},
    )
    at = _app(group_runs_dir)
    at.sidebar.selectbox(key="study").set_value("ab").run()
    assert not at.exception
    assert at.toggle(key="study_cap_ab").value is True
    groups = next(d.value for d in at.dataframe if "IQM ante" in d.value.columns)
    assert groups.set_index("Group").at["b", "Ante"] == 8
    at.toggle(key="study_cap_ab").set_value(False).run()
    groups = next(d.value for d in at.dataframe if "IQM ante" in d.value.columns)
    assert groups.set_index("Group").at["b", "Ante"] > 8
