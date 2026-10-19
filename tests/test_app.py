import json
import re
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from balatro_analysis import picker

APP = str(Path(__file__).parent.parent / "app.py")
TABS = [
    "Overview",
    "Decks",
    "Actions",
    "Economy",
    "Jokers",
    "Joker collection",
    "Rounds",
    "Episode explorer",
]
ECON_TABS = [
    "Money flow",
    "Spending",
    "Interest & saving",
    "Rerolls & prices",
    "Economy vs outcome",
    "Cash-out income",
    "Shop offers",
]

ACTIONS_TABS = [
    "Overview",
    "Context",
    "Sequences",
    "Over training",
    "Availability",
    "Actions vs outcome",
]


def _run(
    runs_dir: Path,
    game: Path | None = None,
    tab: str | None = None,
    econ: str | None = None,
    actions: str | None = None,
) -> AppTest:
    """Run the app on ``runs_dir``; tabs render lazily, so pick the one to inspect."""
    at = AppTest.from_file(APP, default_timeout=60)
    if game is not None:
        at.session_state["game_path"] = str(game)
    if tab:
        at.session_state["main_tab"] = tab
    if econ:
        at.session_state["econ_tab"] = econ
    if actions:
        at.session_state["actions_tab"] = actions
    at.run()
    at.text_input(key="runs_dir").set_value(str(runs_dir)).run()
    return at


def _episode_box(at: AppTest):
    return next(s for s in at.selectbox if s.label == "Episode")


def test_app_renders_default_tab(runs_dir):
    at = _run(runs_dir)
    assert not at.exception
    assert [t.label for t in at.tabs][: len(TABS)] == TABS
    assert at.sidebar.multiselect[0].value == ["run_small"]
    assert len(at.get("image")) == 0  # no game -> names only


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders(runs_dir, fake_game, tab):
    at = _run(runs_dir, fake_game, tab=tab)
    assert not at.exception


def test_only_open_tab_renders(runs_dir):
    """Lazy tabs: the Economy charts aren't built while Overview is open."""
    overview = _run(runs_dir)
    economy = _run(runs_dir, tab="Economy")
    assert len(overview.get("plotly_chart")) < len(economy.get("plotly_chart"))


def _html(at: AppTest) -> str:
    return "".join(h.proto.body for h in at.get("html"))


def test_app_icons_with_game(runs_dir, fake_game):
    decks = _html(_run(runs_dir, fake_game, tab="Decks"))
    assert decks.count('<dialog class="dt-dialog"') == 2  # one modal per deck
    assert decks.count("data:image/png;base64,") >= 2  # deck art
    assert "+1 discard" in re.sub(r"<[^>]+>", "", decks)  # Red Deck's effect text from the game
    explorer = _html(_run(runs_dir, fake_game, tab="Episode explorer"))
    assert 'id="ex-deck-b_red"' in explorer  # episode 0's deck
    assert explorer.count('data-dialog="ex-jk-') == 2  # its two final jokers


def test_app_modals_need_javascript(runs_dir):
    at = _run(runs_dir, tab="Decks")
    html = next(h for h in at.get("html") if "dt-dialog" in h.proto.body)
    assert html.proto.unsafe_allow_javascript


def test_app_jokers_tab_tiles_open_modals(runs_dir):
    html = _html(_run(runs_dir, tab="Jokers"))
    assert 'data-dialog="top-jk-0-j_joker"' in html
    assert "bought <b>2&times;</b>" in html


def test_app_boss_blind_grid(copied_runs, fake_game):
    rounds = copied_runs / "run_small" / "rounds.jsonl"
    hook = {"episode": 0, "ante": 1, "round": 3, "blind_key": "bl_hook", "target": 600,
            "chips": 900, "beaten": True}  # fmt: skip
    rounds.write_text(rounds.read_text() + json.dumps(hook) + "\n")
    at = _run(copied_runs, fake_game, tab="Rounds")
    assert not at.exception
    html = _html(at)
    assert 'id="boss-bl_hook"' in html and "Discards 2 random" in html
    no_boss = _run(copied_runs.parent / "missing", tab="Rounds")
    assert not no_boss.exception


def test_app_deck_filter(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    decks = next(m for m in at.sidebar.multiselect if m.label == "Decks")
    assert decks.value == ["b_blue", "b_red"]
    decks.set_value(["b_blue"]).run()
    assert not at.exception
    assert _episode_box(at).options == ["1  (Blue Deck, ante 1, 4 shop steps)"]


def test_app_episode_explorer_switch(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    _episode_box(at).set_value(1).run()
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
    at = _run(episodes_only_dir, tab="Economy")
    assert not at.exception
    assert any("needs steps.jsonl" in i.value for i in at.info)


def test_app_no_runs(tmp_path):
    at = _run(tmp_path)
    assert not at.exception
    assert at.sidebar.warning


def test_app_skips_run_still_starting(tmp_path):
    """A run whose episodes.csv is still empty doesn't break the others."""
    from conftest import RUNS_DIR

    shutil.copytree(RUNS_DIR / "run_small", tmp_path / "run_small")
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "episodes.csv").touch()
    at = _run(tmp_path)
    assert not at.exception
    assert any("training" in w.value for w in at.sidebar.warning)
    assert at.sidebar.multiselect[0].value == ["run_small", "training"]
    assert [t.label for t in at.tabs][: len(TABS)] == TABS


def test_names_toggle(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    assert "Red Deck" in _episode_box(at).options[0]
    at.toggle(key="pretty").set_value(False).run()
    assert not at.exception
    assert _episode_box(at).options[0] == "0  (b_red, ante 2, 4 shop steps)"
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
    assert any(lbl.startswith("Charts on this tab") for lbl in labels)
    assert labels.count("PNG") == labels.count("SVG") >= 2


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
    at = _run(runs_dir, fake_game, tab="Joker collection")
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
    at = _run(runs_dir, tab="Joker collection")
    assert not at.exception
    assert any("need the game" in c.value for c in at.caption)


# --- Economy -------------------------------------------------------------


@pytest.mark.parametrize("econ", ECON_TABS)
def test_economy_subtabs_render(runs_dir, fake_game, econ):
    at = _run(runs_dir, fake_game, tab="Economy", econ=econ)
    assert not at.exception
    assert not any("needs a run made with" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) >= 1


def test_economy_kpis(runs_dir):
    at = _run(runs_dir, tab="Economy")
    kpis = {m.label: m.value for m in at.metric}
    # fixture: 3 episodes, $11 + $9 + $0 spent; one visit sold a joker for $2
    assert kpis["$ spent / episode"] == "$6.67"
    assert kpis["$ from selling / ep"] == "$0.67"
    assert kpis["Mean $ entering shop"] == "$10.33"


@pytest.mark.parametrize("econ", ["Cash-out income", "Shop offers"])
def test_economy_new_data_note_for_old_runs(tmp_path, econ):
    """Runs from before log_version 2 get a clear note, not an error."""
    import json
    import shutil

    from conftest import RUNS_DIR

    old = tmp_path / "old_logs"
    shutil.copytree(RUNS_DIR / "run_small", old)
    (old / "shops.jsonl").unlink()
    rounds = [json.loads(line) for line in (old / "rounds.jsonl").read_text().splitlines()]
    (old / "rounds.jsonl").write_text(
        "".join(
            json.dumps({k: v for k, v in r.items() if not k.startswith("earn_")}) + "\n"
            for r in rounds
        )
    )
    at = _run(tmp_path, tab="Economy", econ=econ)
    assert not at.exception
    assert any("needs a run made with" in i.value for i in at.info)


# --- Actions -------------------------------------------------------------


@pytest.mark.parametrize("game", [False, True])
@pytest.mark.parametrize("sub", ACTIONS_TABS)
def test_actions_subtabs_render(runs_dir, fake_game, sub, game):
    at = _run(runs_dir, fake_game if game else None, tab="Actions", actions=sub)
    assert not at.exception
    assert not any("needs a run made with" in i.value for i in at.info)
    if sub != "Actions vs outcome":  # the fixture has too few episodes to correlate
        assert len(at.get("plotly_chart")) >= 1


def test_actions_context_splits(runs_dir):
    at = _run(runs_dir, tab="Actions", actions="Context")
    split = next(s for s in at.selectbox if s.label == "Split by")
    for option in split.options:
        split.set_value(option).run()
        assert not at.exception
        assert len(at.get("plotly_chart")) == 1


def test_actions_kpis(runs_dir):
    at = _run(runs_dir, tab="Actions")
    kpis = {m.label: m.value for m in at.metric}
    # fixture: 3 visits of 2, 2 and 4 decisions, all starting with a buy
    assert kpis["Decisions per visit"] == "2.67"
    assert kpis["Visits leaving at once"] == "0.0%"
    assert kpis["Visits with a buy"] == "100.0%"
    assert kpis["Most common first action"] == "Buy card 1"


def test_actions_availability_note_for_old_runs(tmp_path):
    """Runs from before log_version 3 have no legal actions: a note, not an error."""
    import json
    import shutil

    from conftest import RUNS_DIR

    old = tmp_path / "old_logs"
    shutil.copytree(RUNS_DIR / "run_small", old)
    steps = old / "steps.jsonl"
    rows = [json.loads(line) for line in steps.read_text().splitlines()]
    steps.write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "legal"}) + "\n" for r in rows)
    )
    at = _run(tmp_path, tab="Actions", actions="Availability")
    assert not at.exception
    assert any("log_version 3" in i.value for i in at.info)


def test_app_compares_two_runs(runs_dir, tmp_path):
    for name in ("run_small", "run_copy"):
        shutil.copytree(runs_dir / "run_small", tmp_path / name)
    at = _run(tmp_path)
    at.sidebar.multiselect[0].set_value(["run_copy", "run_small"]).run()
    assert not at.exception
    assert any(s.label == "Compare against baseline" for s in at.selectbox)


def test_app_eval_section_and_mode(eval_runs_dir):
    at = _run(eval_runs_dir)
    assert not at.exception
    assert any(r.label == "Eval metric" for r in at.radio)
    at.sidebar.radio(key="phase").set_value("Evaluation").run()
    assert not at.exception
    assert any(s.label == "Eval round" for s in at.sidebar.selectbox)
    assert any("eval episodes" in i.value for i in at.info)


def test_app_no_eval_switch_without_eval(runs_dir):
    at = _run(runs_dir)
    assert not [r for r in at.sidebar.radio if r.label == "Episodes"]


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders_in_eval_mode(eval_runs_dir, tab):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["main_tab"] = tab
    at.session_state["phase"] = "Evaluation"
    at.run()
    at.text_input(key="runs_dir").set_value(str(eval_runs_dir)).run()
    assert not at.exception


def test_app_groups_mode(group_runs_dir):
    at = _run(group_runs_dir)
    assert not at.exception
    at.radio(key="compare_by").set_value("Groups").run()
    assert not at.exception
    assert any(s.label == "Compare against baseline group" for s in at.selectbox)


def test_app_no_groups_toggle_without_groups(runs_dir):
    at = _run(runs_dir)
    assert not [r for r in at.radio if r.label == "Compare"]
