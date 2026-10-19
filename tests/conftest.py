import io
import json
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
BLIND_COLOUR = (255, 255, 0, 255)  # bl_hook chip at (0, 7)
TAROT_COLOUR = (255, 0, 255, 255)  # c_hermit at (9, 0) in Tarots.png


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


@pytest.fixture
def group_runs_dir(tmp_path: Path) -> Path:
    """Group "a" (3 seeds, run_small's antes +0/+1/+2) and group "b" (2 seeds, +3/+4)."""
    root = tmp_path / "group_runs"
    for group, shifts in (("a", (0, 1, 2)), ("b", (3, 4))):
        for seed, shift in enumerate(shifts):
            d = root / f"{group}_s{seed}"
            shutil.copytree(RUNS_DIR / "run_small", d)
            e = pd.read_csv(d / "episodes.csv")
            e["ante"] += shift
            e.to_csv(d / "episodes.csv", index=False)
            config = json.loads((d / "config.json").read_text())
            (d / "config.json").write_text(json.dumps(config | {"group": group, "seed": seed}))
    return root


def _atlas(
    scale: int, cells: dict[tuple[int, int], tuple], cell: tuple[int, int] = (71, 95)
) -> bytes:
    w, h = cell[0] * scale, cell[1] * scale
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
        z.writestr("resources/textures/2x/Tarots.png", _atlas(2, {(9, 0): TAROT_COLOUR}))
        z.writestr("resources/textures/1x/Jokers.png", _atlas(1, {(0, 0): JOKER_COLOUR}))
        z.writestr(
            "resources/textures/2x/BlindChips.png", _atlas(2, {(0, 7): BLIND_COLOUR}, (34, 34))
        )
        z.writestr("localization/en-us.lua", FAKE_LOCALIZATION)
        z.writestr("card.lua", FAKE_CARD_LUA)
        z.writestr("back.lua", FAKE_BACK_LUA)
        z.writestr("functions/common_events.lua", FAKE_COMMON_EVENTS)
        z.writestr("game.lua", FAKE_GAME_LUA)
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
        Back={
            b_red={
                name="Red Deck",
                text={
                    "{C:red}+#1#{} discard",
                    "every round",
                },
            },
            b_black={
                name="Black Deck",
                text={
                    "{C:attention}+#1#{} Joker slot",
                    "",
                    "{C:blue}-#2#{} hand",
                },
            },
            b_nebula={
                name="Nebula Deck",
                text={
                    "Start run with the",
                    "{C:planet,T:v_telescope}#1#{} voucher",
                },
            },
        },
        Blind={
            bl_hook={
                name="The Hook",
                text={
                    "Discards 2 random",
                    "cards per hand played",
                },
            },
            bl_ox={
                name="The Ox",
                text={
                    "Playing a {C:attention}#1#{}",
                    "sets money to $0",
                },
            },
        },
        Voucher={
            v_telescope={
                name="Telescope",
                text={
                    "Celestial packs",
                },
            },
        },
    },
        Tarot={
            c_hermit={
                name="The Hermit",
                text={
                    "Doubles money",
                    "{C:inactive}(Max of {C:money}$#1#{C:inactive})",
                },
            },
            c_magician={
                name="The Magician",
                text={
                    "Enhances {C:attention}#1#{} selected",
                    "cards to {C:attention}#2#s",
                },
            },
        },
        Planet={
            c_mercury={
                name="Mercury",
                text={
                    "{S:0.8}({S:0.8,V:1}lvl.#1#{S:0.8}){} Level up",
                    "{C:attention}#2#",
                    "{C:mult}+#3#{} Mult and",
                    "{C:chips}+#4#{} chips",
                },
            },
        },
        Enhanced={
            m_lucky={
                name="Lucky Card",
                text={
                    "lucky",
                },
            },
        },
        Other={
            p_arcana_normal={
                name="Arcana Pack",
                text={
                    "Choose {C:attention}#1#{} of up to",
                    "{C:attention}#2#{C:tarot} Tarot{} cards",
                },
            },
        },
    misc={
        poker_hands={
            Pair="Pair",
        },
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

FAKE_BACK_LUA = """
function Back:generate_UI(other, ui_scale, min_dims, challenge)
        if name_to_check == 'Blue Deck' then loc_args = {effect_config.hands}
        elseif name_to_check == 'Red Deck' then loc_args = {effect_config.discards}
        elseif name_to_check == 'Black Deck' then loc_args = {effect_config.joker_slot, -effect_config.hands}
        elseif name_to_check == 'Nebula Deck' then loc_args = {localize{type = 'name_text', key = 'v_telescope', set = 'Voucher'}, -1}
        end
end
"""


# generate_card_ui's per-set sections (the unlock branch above them must be skipped)
FAKE_COMMON_EVENTS = """
function generate_card_ui(_c, full_UI_table, specific_vars, card_type, badges, hide_desc)
    if _c.locked then
        if _c.name == 'The Hermit' then loc_vars = {_c.unlock_condition.extra}
        end
    end
    if _c.set == 'Other' then
    elseif _c.set == 'Booster' then
        local desc_override = 'p_arcana_normal'
        if _c.name == 'Arcana Pack' then desc_override = 'p_arcana_normal'; loc_vars = {_c.config.choose, _c.config.extra}
        end
    elseif _c.set == 'Planet' then
        loc_vars = {
            G.GAME.hands[_c.config.hand_type].level,localize(_c.config.hand_type, 'poker_hands'), G.GAME.hands[_c.config.hand_type].l_mult, G.GAME.hands[_c.config.hand_type].l_chips,
            colours = {G.C.UI.TEXT_DARK}
        }
    elseif _c.set == 'Tarot' then
       if _c.name == "The Hermit" then loc_vars = {_c.config.extra}
       elseif _c.name == "The Magician" or _c.name == "The Empress" then loc_vars = {_c.config.max_highlighted, localize{type = 'name_text', set = 'Enhanced', key = _c.config.mod_conv}}
       end
    end
end

function other_function()
end
"""

FAKE_GAME_LUA = """
        hands = {
            ["Pair"] =              {visible = true,    order = 11,mult = 2,   chips = 10,  s_mult = 2,   s_chips = 10,  level = 1, l_mult = 1, l_chips = 15, played = 0},
        },
"""


@pytest.fixture
def copied_runs(tmp_path: Path) -> Path:
    dst = tmp_path / "runs"
    shutil.copytree(RUNS_DIR, dst)
    return dst
