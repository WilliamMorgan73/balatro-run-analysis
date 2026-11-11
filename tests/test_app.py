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
    "Shop items",
    "Joker collection",
    "Rounds",
    "Hands",
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
    explorer: str | None = None,
) -> AppTest:
    """Run the app on ``runs_dir``; tabs render lazily, so pick the one to inspect."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(runs_dir)  # never the developer's real runs folder
    if game is not None:
        at.session_state["game_path"] = str(game)
    if tab:
        at.session_state["main_tab"] = tab
    if econ:
        at.session_state["econ_tab"] = econ
    if actions:
        at.session_state["actions_tab"] = actions
    if explorer:
        at.session_state["explorer_view"] = explorer
    at.run()
    return at


def _episode(at: AppTest) -> int:
    """The episode open in the explorer."""
    return at.session_state["explorer_episode"]


def _header(at: AppTest) -> str:
    return next(m.value for m in at.markdown if m.value.startswith("**Episode "))


def _stepping(at: AppTest) -> tuple[int, int]:
    """(position, count) of the open episode among those Previous/Next step through."""
    pos, count = re.search(r"\((\d+) of (\d+)\)", _header(at)).groups()
    return int(pos), int(count)


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
    assert "@font-face" in _html(_run(runs_dir, fake_game))  # the game's pixel font
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
    assert _episode(at) == 1 and _stepping(at) == (1, 1)
    assert any("Showing Blue Deck" in c.value for c in at.sidebar.caption)


def test_app_episode_explorer_previous_next(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    assert _episode(at) == 0 and _stepping(at) == (1, 3)
    assert at.button(key="ep_prev").disabled
    at.button(key="ep_next").click().run()
    assert not at.exception
    assert _episode(at) == 1 and _stepping(at) == (2, 3)
    at.button(key="ep_prev").click().run()
    assert _episode(at) == 0


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
    assert "Red Deck" in _header(at)
    at.toggle(key="pretty").set_value(False).run()
    assert not at.exception
    assert "b_red, 1, seed TEST_0" in _header(at)
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


def test_reload_order_covers_the_package():
    """Every module is reloaded, after the package modules it imports (purchases
    was once left out, so the new items module called the old one)."""
    import ast

    text = Path(APP).read_text()
    func = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "reload_order"
    )
    package = Path(APP).parent / "src" / "balatro_analysis"
    ns = {"ast": ast, "PACKAGE_DIR": package}
    exec("from __future__ import annotations\n" + ast.get_source_segment(text, func), ns)
    order = ns["reload_order"]()
    assert set(order) == {p.stem for p in package.glob("*.py")} - {"__init__"}
    assert order.index("purchases") < order.index("items")
    assert order.index("stats") < order.index("economy")
    assert order.index("loader") < order.index("metrics")


def test_old_module_reloaded_when_only_app_changed(runs_dir, monkeypatch):
    """The NCC bug: a server reloaded after one pull (stamping the package), then a
    second pull changed only app.py.  An old purchases (no ``offers``) left loaded
    must still be replaced, though no package file changed since."""
    from balatro_analysis import purchases

    at = _run(runs_dir)
    monkeypatch.delattr(purchases, "offers")
    monkeypatch.setattr(purchases, "_app_mtime", 0.0, raising=False)  # loaded from an old file
    at.run()
    assert not at.exception
    assert hasattr(purchases, "offers")


def test_stale_modules_reloaded(runs_dir, monkeypatch):
    """A server started before a package change must not keep old modules
    (the 'unexpected keyword argument decks' bug)."""
    from balatro_analysis import metrics

    at = _run(runs_dir)
    real = metrics.filter_episodes

    def old_filter(run, start, end):  # the pre-deck-filter signature
        raise AssertionError("stale module used")

    monkeypatch.setattr(metrics, "filter_episodes", old_filter)
    del metrics._app_mtime  # like a server started before this guard existed
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
    at.sidebar.segmented_control(key="phase").set_value("Evaluation").run()
    assert not at.exception
    assert any(s.label == "Eval round" for s in at.sidebar.selectbox)
    assert any("eval episodes" in i.value for i in at.info)


def test_app_no_eval_switch_without_eval(runs_dir):
    at = _run(runs_dir)
    assert not [r for r in at.sidebar.segmented_control if r.label == "Episodes"]


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


def test_every_chart_has_a_description():
    """chart() takes a required description; check every call passes one, even in
    branches the rendering tests don't reach."""
    import ast

    calls = [
        n
        for n in ast.walk(ast.parse(Path(APP).read_text()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "chart"
    ]
    assert len(calls) > 30
    for c in calls:
        desc = c.args[2] if len(c.args) > 2 else None
        assert desc is not None, f"chart() at app.py:{c.lineno} has no description"
        if isinstance(desc, ast.Constant):
            assert desc.value.strip(), f"empty description at app.py:{c.lineno}"


def test_chart_description_shown(runs_dir):
    at = _run(runs_dir, tab="Overview")
    assert not at.exception
    assert any("Share of episodes that ended at each ante" in c.value for c in at.caption)


@pytest.mark.parametrize("pretty", [True, False])
def test_explorer_item_icons_open_details(runs_dir, fake_game, pretty):
    """Shop decisions and Blinds played are HTML tables whose item icons open the
    same modals as elsewhere; the episode finder keeps its sortable icon table."""
    at = _run(runs_dir, fake_game, tab="Episode explorer", explorer="Money & blinds")
    if not pretty:
        at.toggle(key="pretty").set_value(False).run()
    assert not at.exception
    (episodes,) = [d.value for d in at.main.dataframe]  # shop/blinds aren't dataframes
    icons = "Final jokers icons" if pretty else "final_jokers_icons"
    assert episodes[icons].str.startswith("data:image/png;base64,", na=True).all()
    assert episodes[icons].notna().any()

    html = [h.proto.body for h in at.get("html")]
    shop = next(h for h in html if 'data-dialog="ex-shop-' in h)
    blinds = next(h for h in html if 'data-dialog="ex-blind-' in h)
    for table in (shop, blinds):
        assert 'data-dialog="ex-' in table and '<dialog class="dt-dialog"' in table
        assert "['j_" not in table  # never raw key lists
    assert 'id="ex-shop-j_joker"' in shop  # the joker's collection modal
    assert ("Buy card 1" if pretty else "buy_card_0") in shop


def test_explorer_drops_icons_on_long_tables(copied_runs, fake_game):
    """Past ICON_ROWS_MAX rows the episode finder shows joker names only: a strip
    per row of a 30,000-episode run took several GB (an NCC job was OOM-killed)."""
    csv = copied_runs / "run_small" / "episodes.csv"
    header, *rows = csv.read_text().splitlines()
    first = rows[0].split(",")
    extra = [",".join([str(len(rows) + i), *first[1:]]) for i in range(1001)]
    csv.write_text("\n".join([header, *rows, *extra]) + "\n")
    at = _run(copied_runs, fake_game, tab="Episode explorer")
    assert not at.exception
    (episodes,) = [d.value for d in at.main.dataframe]
    assert len(episodes) > 1000
    assert "Final jokers icons" not in episodes and "Final jokers" in episodes
    assert any("Icons are shown for up to 1,000 rows" in c.value for c in at.caption)


# --- Finding episodes and runs (#17) ------------------------------------------


def test_suggestion_opens_episode(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    at.button(key="ep_next").click().run()
    at.button(key="ep_next").click().run()
    assert _episode(at) == 2
    pills = next(b for b in at.get("button_group") if b.label == "Suggested episodes")
    assert "Best: 0" in pills.options and "Closest loss: 0" in pills.options
    pills.set_value("Best: 0").run()
    assert not at.exception
    assert _episode(at) == 0


def test_episode_filters_narrow_the_choice(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    at.multiselect(key="ep_decks").set_value(["b_blue"]).run()
    assert not at.exception
    assert _episode(at) == 1 and _stepping(at) == (1, 1)
    at.multiselect(key="ep_jokers").set_value(["j_joker"]).run()  # blue deck ep ended with Banner
    assert any("No episodes match" in c.value for c in at.caption)
    assert _stepping(at)[1] == 3  # falls back to every episode


def test_deep_link_opens_episode(runs_dir):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(runs_dir)
    at.query_params["run"] = "run_small"
    at.query_params["episode"] = "2"
    at.run()
    assert not at.exception
    assert at.session_state["main_tab"] == "Episode explorer"
    assert _episode(at) == 2
    at.button(key="ep_prev").click().run()
    assert at.query_params["episode"] == "1"  # the URL follows the selection


def test_find_runs_keeps_hidden_selection(runs_dir, tmp_path):
    for name in ("run_small", "run_copy"):
        shutil.copytree(runs_dir / "run_small", tmp_path / name)
    at = _run(tmp_path)
    runs = at.sidebar.multiselect[0]
    assert runs.value == ["run_copy", "run_small"]
    at.text_input(key="run_search").set_value("copy").run()
    assert not at.exception
    assert at.sidebar.multiselect[0].value == ["run_copy", "run_small"]  # nothing dropped
    assert any("hidden by the filters" in c.value for c in at.sidebar.caption)
    shown = next(d.value for d in at.sidebar.dataframe)  # stats of the runs shown
    assert shown.index.tolist() == ["run_copy"]
    next(b for b in at.sidebar.button if b.label == "Clear").click().run()
    assert not at.exception
    assert at.sidebar.multiselect[0].value == []
    next(b for b in at.sidebar.button if b.label == "All shown").click().run()
    assert at.sidebar.multiselect[0].value == ["run_copy"]  # only what the search shows


# --- Hand log (eval only, log_version 7) ---------------------------------------


def _eval_app(eval_runs_dir: Path, tab: str, game: Path | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["runs_dir"] = str(eval_runs_dir)
    at.session_state["main_tab"] = tab
    at.session_state["phase"] = "Evaluation"
    if game is not None:
        at.session_state["game_path"] = str(game)
    at.run()
    return at


def test_hands_tab_needs_evaluation(eval_runs_dir):
    at = _run(eval_runs_dir, tab="Hands")
    assert not at.exception
    assert any("**Test** or **Evaluation**" in i.value for i in at.info)
    assert not at.get("plotly_chart")


def test_hands_tab_in_evaluation(eval_runs_dir):
    at = _eval_app(eval_runs_dir, "Hands")
    assert not at.exception
    assert len(at.get("plotly_chart")) == 4  # hands, score, per blind, deck


def test_replay_plays_a_blind(eval_runs_dir, fake_game):
    at = _eval_app(eval_runs_dir, "Episode explorer", fake_game)
    assert not at.exception
    assert _episode(at) == 3  # the last eval round's first episode, which has a hand log
    html = _html(at)
    assert html.count('class="rp-act"') == 3  # small blind: a discard and two plays
    assert "Flush" in html and "data:image/png;base64," in html  # card art from the game
    assert 'data-dialog="ex-held-' in html  # jokers held, opening their collection card
    assert ">lvl.1<" in html
    assert any("As held at the first play" in c.value for c in at.caption)
    blinds = at.segmented_control(key="ex_blind")
    blinds.set_value(2).run()
    assert not at.exception
    assert _html(at).count('class="rp-act"') == 1
    assert "Pair 2" in "".join(c.value for c in at.caption)  # hand levels under the deck


def test_replay_without_hand_log_says_why(runs_dir):
    at = _run(runs_dir, tab="Episode explorer")
    assert any("only recorded for test and evaluation" in i.value for i in at.info)


def test_shop_view(runs_dir, fake_game):
    at = _run(runs_dir, fake_game, tab="Episode explorer", explorer="Shop")
    assert not at.exception
    html = _html(at)
    assert "Shop after round" in html and "bought" in html


def test_run_sheet_opens_best_episode(runs_dir):
    at = _run(runs_dir)
    assert "Mean ante" in _html(at)
    at.button(key="sheet_best_run_small").click().run()
    assert not at.exception
    assert at.session_state["main_tab"] == "Episode explorer" and _episode(at) == 0


def test_overview_with_a_run_behind_on_eval_rounds(eval_runs_dir):
    """A run still training hasn't played the last eval round yet: it has no
    episodes under the default eval round, and every tab must still render."""
    e = eval_runs_dir / "run_b" / "eval" / "episodes.csv"
    lines = e.read_text().splitlines()
    e.write_text("\n".join(lines[:4]) + "\n")  # header + eval round 0 only
    for tab in TABS:
        at = _eval_app(eval_runs_dir, tab)
        assert not at.exception, (tab, at.exception)
    at = _eval_app(eval_runs_dir, "Overview")
    assert any("not shown: run_b" in c.value for c in at.caption)


def _subheaders(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader]


def test_overview_iqm_and_power_for_runs(eval_runs_dir):
    at = _eval_app(eval_runs_dir, "Overview")
    assert not at.exception
    assert any("Interquartile mean (IQM)" in c.value for c in at.caption)
    assert "Can these runs tell agents apart?" in _subheaders(at)
    labels = [m.label for m in at.metric]
    assert "Detectable now" in labels and any(lbl.startswith("Games needed") for lbl in labels)
    assert any("Paired on games" in c.value for c in at.caption)
    # run_b is exactly one ante better on every shared seed: any difference shows
    assert next(m for m in at.metric if m.label == "Detectable now").value == "0.00 ante"
    assert not [w for w in at.warning if "reliably detect" in w.value]
    assert any("Probability of improvement" in c.value for c in at.caption)
    assert any("Performance profile (Agarwal" in c.value for c in at.caption)


def test_overview_iqm_and_power_for_groups(group_runs_dir):
    at = _run(group_runs_dir)
    at.radio(key="compare_by").set_value("Groups").run()
    assert not at.exception
    assert any("Interquartile mean (IQM)" in c.value for c in at.caption)
    labels = [m.label for m in at.metric]
    assert "Detectable now" in labels and "With unlimited games" in labels
    assert any(lbl.startswith("Seeds needed") for lbl in labels)
    # seeds a whole ante apart: 2-3 seeds can't detect a quarter of an ante
    assert any("these groups can reliably detect" in w.value for w in at.warning)
    at.number_input(key="power_target").set_value(50.0).run()
    assert not at.exception
    assert not [w for w in at.warning if "reliably detect" in w.value]
    assert any("Probability of improvement" in c.value for c in at.caption)
    profile = next(c.value for c in at.caption if c.value.startswith("Performance profile"))
    assert "one line per group" in profile


def test_app_loads_picked_runs_within_memory_budget(runs_dir, tmp_path, monkeypatch):
    """Runs beyond the memory budget aren't loaded (one run always is), and the
    sidebar says which and why; unpicked runs aren't loaded at all."""
    for name in ("run_a", "run_b", "run_c"):
        shutil.copytree(runs_dir / "run_small", tmp_path / name)
    monkeypatch.setenv("BALATRO_ANALYSIS_RUN_MEMORY", "0.000001")  # 1 KB: room for none
    at = _run(tmp_path)
    assert not at.exception
    warning = next(w.value for w in at.sidebar.warning if "within memory" in w.value)
    assert "run_b, run_c" in warning and "run_a" not in warning
    at.sidebar.multiselect[0].set_value(["run_c"]).run()
    assert not at.exception
    assert not any("within memory" in w.value for w in at.sidebar.warning)
