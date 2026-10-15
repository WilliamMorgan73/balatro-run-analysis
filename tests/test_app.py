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
    assert len(at.tabs) == 8
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


def test_names_toggle(runs_dir):
    at = _run(runs_dir)
    episode = next(s for s in at.selectbox if s.label == "Episode")
    assert "Red Deck" in episode.options[0]
    at.toggle(key="pretty").set_value(False).run()
    assert not at.exception
    episode = next(s for s in at.selectbox if s.label == "Episode")
    assert episode.options[0] == "0  (b_red, ante 2, 4 shop steps)"
    decks = next(m for m in at.sidebar.multiselect if m.label == "Decks")
    assert decks.format_func("b_red") == "b_red"


def test_names_query_param(runs_dir):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["names"] = "keys"
    at.run()
    assert at.toggle(key="pretty").value is False


def test_export_buttons_present(runs_dir):
    at = _run(runs_dir)
    assert not at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "Data (ZIP of CSVs)" in labels
    assert any(lbl.startswith("All ") and lbl.endswith("charts (ZIP)") for lbl in labels)
    assert labels.count("PNG") == labels.count("SVG") >= 5


def test_stale_modules_reloaded(runs_dir, monkeypatch):
    """A server started before a package change must not keep old modules
    (the 'unexpected keyword argument decks' bug)."""
    from balatro_analysis import metrics

    at = _run(runs_dir)
    real = metrics.filter_episodes

    def old_filter(run, start, end):  # the pre-deck-filter signature
        raise AssertionError("stale module used")

    monkeypatch.setattr(metrics, "filter_episodes", old_filter)
    import balatro_analysis

    del balatro_analysis._app_mtime  # like a server started before this guard existed
    at.run()
    assert not at.exception
    assert metrics.filter_episodes.__code__ is not old_filter.__code__
    assert metrics.filter_episodes.__name__ == real.__name__


def test_joker_collection_tab(runs_dir, fake_game):
    at = _run(runs_dir, fake_game)
    assert not at.exception
    assert any(c.value.startswith("Showing 150 of 150 jokers") for c in at.caption)
    at.text_input(key="jk_query").set_value("greedy").run()
    assert any(c.value.startswith("Showing 1 of 150") for c in at.caption)
    html = next(h.proto.body for h in at.get("html"))
    assert "Greedy Joker" in html and "Played" in html  # name + effect text from the game
    at.toggle(key="jk_used").set_value(True).run()
    at.text_input(key="jk_query").set_value("").run()
    assert any(c.value.startswith("Showing 3 of 150") for c in at.caption)


def test_joker_collection_without_game(runs_dir):
    at = _run(runs_dir)
    assert not at.exception
    assert any("need the game" in c.value for c in at.caption)
