import shutil
from pathlib import Path

import pytest

from balatro_analysis.loader import Run, load_run

RUNS_DIR = Path(__file__).parent / "fixtures" / "runs"


@pytest.fixture
def runs_dir() -> Path:
    return RUNS_DIR


@pytest.fixture
def run() -> Run:
    return load_run(RUNS_DIR / "run_small", RUNS_DIR)


@pytest.fixture
def episodes_only_dir(tmp_path: Path) -> Path:
    """A run in the old format: episodes.csv only."""
    d = tmp_path / "old_run"
    d.mkdir()
    shutil.copy(RUNS_DIR / "run_small" / "episodes.csv", d / "episodes.csv")
    return tmp_path
