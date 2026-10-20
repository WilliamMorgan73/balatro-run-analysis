"""A run written by balatro-shop-rl loads in the dashboard (the run format is the
contract between the two repos; see docs/run-format.md)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from balatro_analysis.loader import load_run

SHOP_RL = Path(__file__).parent.parent.parent / "balatro-shop-rl"


@pytest.mark.slow
@pytest.mark.skipif(
    not (SHOP_RL / "src" / "shop_rl").is_dir(),
    reason="no balatro-shop-rl checkout next to this repo",
)
def test_training_output_loads(tmp_path):
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    cmd = [
        "uv", "run", "--project", str(SHOP_RL), "python", str(SHOP_RL / "scripts" / "train.py"),
        "--episodes", "3", "--play", "smart", "--decks", "b_red,b_blue",
        "--out-dir", str(tmp_path), "--run-name", "example", "--log-every", "3",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)

    run = load_run(tmp_path / "example", tmp_path)
    assert len(run.episodes) == 3
    assert set(run.episodes["deck"]) <= {"b_red", "b_blue"}
    assert run.config["play_agent"] == "SmartPlayAgent"
    assert run.config["log_version"] == 6
    assert run.config["jackdaw_commit"] and "shop_rl_commit" in run.config
    assert run.steps is not None and len(run.steps) == run.episodes["shop_steps"].sum()
    assert run.rounds is not None and len(run.rounds) >= 3
    assert json.loads((tmp_path / "example" / "config.json").read_text())["n_actions"] == 17
