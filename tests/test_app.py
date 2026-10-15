from pathlib import Path

from streamlit.testing.v1 import AppTest

from balatro_analysis import picker

APP = str(Path(__file__).parent.parent / "app.py")


def _run(runs_dir: Path, game: Path | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if game is not None:
        at.session_state["game_path"] = str(game)
    at.run()
    at.text_input(key="runs_dir").set_value(str(runs_dir)).run()
    return at


def test_app_renders_all_tabs(runs_dir):
    at = _run(runs_dir)
    assert not at.exception
    assert len(at.tabs) == 7
    assert at.sidebar.multiselect[0].value == ["run_small"]
    assert len(at.get("image")) == 0  # no game -> names only


def test_app_icons_with_game(runs_dir, fake_game):
    at = _run(runs_dir, fake_game)
    assert not at.exception
    # deck cards (2 decks) + most-bought jokers + explorer deck + final jokers
    assert len(at.get("image")) >= 5


def test_app_deck_filter(runs_dir):
    at = _run(runs_dir)
    decks = next(m for m in at.sidebar.multiselect if m.label == "Decks")
    assert decks.value == ["b_blue", "b_red"]
    decks.set_value(["b_blue"]).run()
    assert not at.exception
    episode = next(s for s in at.selectbox if s.label == "Episode")
    assert episode.options == ["1  (Blue Deck, ante 1, 4 shop steps)"]


def test_app_episode_explorer_switch(runs_dir):
    at = _run(runs_dir)
    episode = next(s for s in at.selectbox if s.label == "Episode")
    episode.set_value(1).run()
    assert not at.exception


def test_app_browse_button(runs_dir, monkeypatch):
    at = _run(Path("/nonexistent"))
    monkeypatch.setattr(
        picker.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": f"{runs_dir}\n", "stderr": ""})(),
    )
    at.button(key="browse_runs").click().run()
    assert not at.exception
    assert at.session_state["runs_dir"] == str(runs_dir)
    assert at.sidebar.multiselect[0].value == ["run_small"]


def test_app_old_format_run(episodes_only_dir):
    at = _run(episodes_only_dir)
    assert not at.exception
    assert any("needs steps.jsonl" in i.value for i in at.info)


def test_app_no_runs(tmp_path):
    at = _run(tmp_path)
    assert not at.exception
    assert at.sidebar.warning
