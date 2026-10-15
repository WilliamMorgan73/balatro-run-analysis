from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def _run(runs_dir: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(runs_dir)).run()
    return at


def test_app_renders_all_tabs(runs_dir):
    at = _run(runs_dir)
    assert not at.exception
    assert len(at.tabs) == 6
    assert at.sidebar.multiselect[0].value == ["run_small"]


def test_app_episode_explorer_switch(runs_dir):
    at = _run(runs_dir)
    at.selectbox(key="explorer_run")  # exists
    explorer_episode = next(s for s in at.selectbox if s.label == "Episode")
    explorer_episode.set_value(1).run()
    assert not at.exception


def test_app_old_format_run(episodes_only_dir):
    at = _run(episodes_only_dir)
    assert not at.exception
    assert any("needs steps.jsonl" in i.value for i in at.info)


def test_app_no_runs(tmp_path):
    at = _run(tmp_path)
    assert not at.exception
    assert at.sidebar.warning
