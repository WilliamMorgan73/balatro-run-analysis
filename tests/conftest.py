import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from balatro_analysis.loader import Run, load_run

RUNS_DIR = Path(__file__).parent / "fixtures" / "runs"

# Colours painted into the fake atlases, so tests can check which cell was cut
JOKER_COLOUR = (255, 0, 0, 255)  # j_joker at pos (0, 0)
SOUL_COLOUR = (0, 0, 255, 255)  # j_caino soul layer at (3, 9)
DECK_COLOUR = (0, 255, 0, 255)  # b_black at pos (3, 2)


@pytest.fixture(autouse=True)
def no_real_game(monkeypatch, tmp_path):
    """Never pick up the developer's real Balatro install in tests."""
    monkeypatch.setenv("BALATRO_PATH", str(tmp_path / "no-balatro-here"))


@pytest.fixture
def runs_dir() -> Path:
    return RUNS_DIR


@pytest.fixture
def run() -> Run:
    return load_run(RUNS_DIR / "run_small", RUNS_DIR)


@pytest.fixture
def episodes_only_dir(tmp_path: Path) -> Path:
    """A run in the oldest format: episodes.csv only, no deck/stake columns."""
    d = tmp_path / "old_run"
    d.mkdir()
    e = pd.read_csv(RUNS_DIR / "run_small" / "episodes.csv").drop(columns=["deck", "stake"])
    e.to_csv(d / "episodes.csv", index=False)
    return tmp_path


def _atlas(scale: int, cells: dict[tuple[int, int], tuple]) -> bytes:
    w, h = 71 * scale, 95 * scale
    img = Image.new("RGBA", (10 * w, 16 * h), (0, 0, 0, 0))
    for (x, y), colour in cells.items():
        img.paste(Image.new("RGBA", (w, h), colour), (x * w, y * h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_game(tmp_path: Path) -> Path:
    """A Balatro install dir whose Balatro.exe zip holds tiny solid-colour atlases."""
    game = tmp_path / "Balatro"
    game.mkdir()
    with zipfile.ZipFile(game / "Balatro.exe", "w") as z:
        z.writestr(
            "resources/textures/2x/Jokers.png",
            _atlas(2, {(0, 0): JOKER_COLOUR, (3, 9): SOUL_COLOUR}),
        )
        z.writestr("resources/textures/2x/Enhancers.png", _atlas(2, {(3, 2): DECK_COLOUR}))
    return game


@pytest.fixture
def copied_runs(tmp_path: Path) -> Path:
    dst = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, dst)
    return dst
