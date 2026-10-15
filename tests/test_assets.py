import io
import subprocess

import pytest
from conftest import DECK_COLOUR, JOKER_COLOUR, SOUL_COLOUR
from PIL import Image

from balatro_analysis import catalog, picker
from balatro_analysis.sprites import Sprites, find_balatro, game_archive

# --- catalog ---------------------------------------------------------------


def test_names():
    assert catalog.name("j_greedy_joker") == "Greedy Joker"
    assert catalog.name("j_caino") == "Canio"
    assert catalog.name("b_black") == "Black Deck"
    assert catalog.name("p_buffoon_normal_1") == "Buffoon Pack"
    assert catalog.name("bl_hook") == "The Hook"
    assert catalog.name("x_not_real_key") == "Not Real Key"  # unknown: prettified
    assert catalog.name(None) == ""


def test_stake_name():
    assert catalog.stake_name("2") == "Red Stake"
    assert catalog.stake_name("unknown") == "Unknown stake"


# --- sprites ---------------------------------------------------------------


def _pixel(png: bytes, xy=(10, 10)):
    return Image.open(io.BytesIO(png)).convert("RGBA").getpixel(xy)


def test_game_archive_from_dir_or_file(fake_game):
    assert game_archive(fake_game) == fake_game / "Balatro.exe"
    assert game_archive(fake_game / "Balatro.exe") == fake_game / "Balatro.exe"
    assert game_archive(fake_game / "missing") is None


def test_find_balatro_env(monkeypatch, fake_game):
    monkeypatch.setenv("BALATRO_PATH", str(fake_game))
    assert find_balatro() == fake_game / "Balatro.exe"


def test_find_balatro_steam_library(monkeypatch, tmp_path, fake_game):
    monkeypatch.delenv("BALATRO_PATH")
    steam = tmp_path / "steam"
    library = tmp_path / "games"
    (steam / "steamapps").mkdir(parents=True)
    (library / "steamapps" / "common").mkdir(parents=True)
    fake_game.rename(library / "steamapps" / "common" / "Balatro")
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n "0"\n {{\n  "path"\t\t"{library}"\n }}\n}}\n'
    )
    expected = library / "steamapps" / "common" / "Balatro" / "Balatro.exe"
    assert find_balatro((str(steam),)) == expected


def test_sprites_crop_and_soul(fake_game):
    s = Sprites(fake_game / "Balatro.exe")
    joker = s.png("j_joker")
    assert Image.open(io.BytesIO(joker)).size == (142, 190)
    assert _pixel(joker) == JOKER_COLOUR
    assert _pixel(s.png("j_caino")) == SOUL_COLOUR  # soul layer composited on top
    assert _pixel(s.png("b_black")) == DECK_COLOUR
    assert s.png("j_not_a_joker") is None
    assert s.png(None) is None
    assert s.data_uri("j_joker").startswith("data:image/png;base64,")


# --- picker ----------------------------------------------------------------


class _Result:
    def __init__(self, code, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def test_picker_prefers_kdialog_on_kde(monkeypatch):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(picker.shutil, "which", lambda t: f"/usr/bin/{t}")
    cmds = picker._commands("/tmp", "t")
    assert [c[0] for c in cmds[:2]] == ["/usr/bin/kdialog", "/usr/bin/zenity"]
    assert cmds[-1][1] == "-c"  # tkinter fallback last


def test_picker_returns_choice(monkeypatch):
    monkeypatch.setattr(picker.subprocess, "run", lambda *a, **k: _Result(0, "/chosen/dir\n"))
    assert picker.pick_directory("/tmp") == "/chosen/dir"


def test_picker_cancel(monkeypatch):
    monkeypatch.setattr(picker.subprocess, "run", lambda *a, **k: _Result(1))
    assert picker.pick_directory("/tmp") is None


def test_picker_falls_through_then_raises(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[0])
        if len(calls) == 1:
            raise OSError("no display")
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(picker.subprocess, "run", run)
    with pytest.raises(picker.PickerUnavailable):
        picker.pick_directory("/tmp")
    assert len(calls) == len(picker._commands("/tmp", "t"))


def test_name_handles_missing_values():
    assert catalog.name(float("nan")) == ""
    assert not catalog.is_joker(float("nan"))
