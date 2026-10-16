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


def _with_eval(run_dir: Path, ante_shift: int = 0) -> None:
    """Give ``run_dir`` an eval/ folder: run_small's 3 episodes replayed on
    EVAL_0..2 in two eval rounds (after 0 and 10 training episodes)."""
    shutil.copytree(RUNS_DIR / "run_small", run_dir)
    e = pd.read_csv(RUNS_DIR / "run_small" / "episodes.csv")
    rounds = []
    for rnd, done in enumerate((0, 10)):
        r = e.copy()
        r["episode"] = rnd * len(e) + r.index
        r["seed"] = [f"EVAL_{i}" for i in range(len(e))]
        r["ante"] = r["ante"] + rnd + ante_shift
        r["eval_round"] = rnd
        r["train_episodes"] = done
        rounds.append(r)
    (run_dir / "eval").mkdir()
    pd.concat(rounds).to_csv(run_dir / "eval" / "episodes.csv", index=False)
    for name in ("steps.jsonl", "rounds.jsonl", "shops.jsonl"):
        shutil.copy(RUNS_DIR / "run_small" / name, run_dir / "eval" / name)


@pytest.fixture
def eval_runs_dir(tmp_path: Path) -> Path:
    """Two runs with eval episodes on the same seeds; run_b is one ante better."""
    root = tmp_path / "eval_runs"
    _with_eval(root / "run_a")
    _with_eval(root / "run_b", ante_shift=1)
    return root


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
        z.writestr("resources/textures/1x/Jokers.png", _atlas(1, {(0, 0): JOKER_COLOUR}))
        z.writestr("localization/en-us.lua", FAKE_LOCALIZATION)
        z.writestr("card.lua", FAKE_CARD_LUA)
    return game


# Shaped like the game's files (written for these tests, not copied from the game)
FAKE_LOCALIZATION = """
return {
    descriptions={
        Joker={
            j_joker={
                name="Joker",
                text={
                    "{C:mult}+#1#{} Mult",
                },
            },
            j_greedy_joker={
                name="Greedy Joker",
                text={
                    "Played {C:diamonds}#2#{} cards",
                    "give {C:mult}+#1#{} Mult",
                },
            },
            j_cavendish={
                name="Cavendish",
                text={
                    "{X:mult,C:white} X#1# {} Mult",
                    "{C:green}#2# in #3#{} chance <gone>",
                },
            },
            j_castle={
                name="Castle",
                text={
                    "Gains {C:chips}+#1#{} per {V:1}#2#{} discard",
                },
            },
            j_misprint={
                name="Misprint",
                text={
                    "",
                },
            },
        },
    },
    misc={
        suits_singular={
            Clubs="Club",
            Diamonds="Diamond",
        },
    },
}
"""

FAKE_CARD_LUA = """
function Card:generate_UIBox_ability_table()
        if self.ability.name == 'Joker' then loc_vars = {self.ability.mult}
        elseif self.ability.name == 'Greedy Joker' or self.ability.name == 'Lusty Joker' then
            loc_vars = {self.ability.extra.s_mult, localize(self.ability.extra.suit, 'suits_singular')}
        elseif self.ability.name == 'Cavendish' then loc_vars = {self.ability.extra.Xmult, ''..(G.GAME and G.GAME.probabilities.normal or 1), self.ability.extra.odds}
        elseif self.ability.name == 'Castle' then loc_vars = {self.ability.extra.chip_mod, localize(G.GAME.current_round.castle_card.suit, 'suits_singular'), colours = {G.C.SUITS.Spades}}
        end
end
"""


@pytest.fixture
def copied_runs(tmp_path: Path) -> Path:
    dst = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, dst)
    return dst
