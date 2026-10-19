"""The jackdaw example scripts: kept in sync with the jackdaw checkout, and their
output loads in the dashboard."""

import shutil
import subprocess
from pathlib import Path

import pytest

from balatro_analysis.loader import load_run

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "jackdaw"
JACKDAW = ROOT.parent / "jackdaw-balatro"
SCRIPTS = sorted(p.relative_to(EXAMPLES) for p in EXAMPLES.rglob("*.py"))

needs_jackdaw = pytest.mark.skipif(
    not (JACKDAW / "jackdaw").is_dir(), reason="no jackdaw-balatro checkout next to this repo"
)


def test_examples_exist():
    assert [str(p) for p in SCRIPTS] == [
        "scripts/train_random_shop.py",
        "scripts/verify_shop_rl.py",
    ]


@needs_jackdaw
@pytest.mark.parametrize("rel", SCRIPTS, ids=str)
def test_examples_match_jackdaw(rel):
    """Fails when the copies here and in ../jackdaw-balatro drift apart."""
    theirs = JACKDAW / rel
    if not theirs.exists():
        pytest.skip(f"{rel} not pasted into jackdaw yet")
    assert (EXAMPLES / rel).read_bytes() == theirs.read_bytes(), (
        f"examples/jackdaw/{rel} differs from {theirs}; copy the newer one over"
    )


@pytest.mark.slow
@needs_jackdaw
def test_training_output_loads(tmp_path):
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    script = EXAMPLES / "scripts" / "train_random_shop.py"
    cmd = [
        "uv", "run", "--project", str(JACKDAW), "python", str(script),
        "--episodes", "3", "--play", "smart", "--decks", "b_red,b_blue",
        "--out-dir", str(tmp_path), "--run-name", "example", "--log-every", "3",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)

    run = load_run(tmp_path / "example", tmp_path)
    assert len(run.episodes) == 3
    assert set(run.episodes["deck"]) <= {"b_red", "b_blue"}
    assert run.config["play_agent"] == "SmartPlayAgent"
    assert run.steps is not None and len(run.steps) == run.episodes["shop_steps"].sum()
    assert run.rounds is not None and len(run.rounds) >= 3
    assert run.config["log_version"] == 6
    assert (run.episodes["ante"] == run.episodes["max_ante"]).all()  # max_ante is the ante
    assert (run.episodes["ante"] >= run.rounds.groupby("episode")["ante"].max()).all()
    for a, legal in zip(run.steps["action"], run.steps["legal"], strict=True):
        assert a in legal and 16 in legal  # chosen action legal; leaving always is
    beaten = run.rounds[run.rounds["beaten"] & run.rounds["earn_total"].notna()]
    assert len(beaten) and (beaten["earn_total"] > 0).all()
    assert run.shops is not None and set(run.shops["reason"]) >= {"enter"}
