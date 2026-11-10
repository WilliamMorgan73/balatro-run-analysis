"""Streamlit dashboard for balatro-shop-rl training runs.

uv run streamlit run app.py -- --runs-dir ../balatro-shop-rl/runs
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import html
import importlib
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PACKAGE_DIR = Path(__file__).parent / "src" / "balatro_analysis"
RELOAD_ORDER = [
    "cards",
    "catalog",
    "names",
    "loader",
    "runcache",
    "memory",
    "find",
    "metrics",
    "hands",
    "economy",
    "actions",
    "plots",
    "sprites",
    "game_text",
    "theme",
    "details",
    "replay",
    "collection",
    "picker",
    "export",
]


def package_mtime() -> float:
    files = (p for p in PACKAGE_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return max(p.stat().st_mtime for p in files)


def reload_if_changed() -> None:
    """Streamlit re-runs this script on edit but keeps imported package modules,
    so a long-running server would mix new app code with old modules. Reload
    the package (in dependency order) when its files changed since it was
    stamped below, or when it was never stamped (a server started on older code)."""
    pkg = sys.modules.get("balatro_analysis")
    if pkg is None or getattr(pkg, "_app_mtime", None) == package_mtime():
        return
    for name in RELOAD_ORDER:
        mod = sys.modules.get(f"balatro_analysis.{name}")
        if mod is not None:
            importlib.reload(mod)


reload_if_changed()

from balatro_analysis import (  # noqa: E402
    actions,
    cards,
    collection,
    details,
    economy,
    endless,
    export,
    find,
    hands,
    latex,
    loader,
    memory,
    metrics,
    plots,
    purchases,
    replay,
    runcache,
    stats,
    studies,
    theme,
)
from balatro_analysis.game_text import RARITY, GameText  # noqa: E402
from balatro_analysis.loader import Run, TableBudgetError, episodes_bytes, find_runs  # noqa: E402
from balatro_analysis.names import Names, split_items  # noqa: E402
from balatro_analysis.picker import PickerUnavailable, pick_directory  # noqa: E402
from balatro_analysis.sprites import Sprites, find_balatro, game_archive  # noqa: E402

sys.modules["balatro_analysis"]._app_mtime = package_mtime()  # see reload_if_changed

DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "balatro-shop-rl" / "runs"
MIN_VISITS = 5  # money-curve points need at least this many shop visits
MIN_LEGAL = 20  # preference needs an action to have been legal this often


def cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    # balatro-shop-rl's scripts/run_manifest.py: runs across folders, and studies
    ap.add_argument("--manifest", type=Path, default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args


def cli_runs_dir() -> Path:
    return cli_args().runs_dir


def cli_manifest() -> str:
    """The manifest given, else the runs folder's ``manifest.json`` if it has one."""
    args = cli_args()
    if args.manifest is not None:
        return str(args.manifest)
    default = args.runs_dir / studies.MANIFEST
    return str(default) if default.is_file() else ""


@st.cache_data(show_spinner=False)
def cached_manifest(path: str, mtime: float) -> studies.Manifest:
    """``studies.load_manifest``; mtime: re-read once it's regenerated."""
    return studies.load_manifest(path)


FILTERED_KEPT = 2  # filtered copies kept per run (each up to a full copy of it)
# Tables longer than this show item names without icon strips: a strip is a ~20 KB
# PNG per row, so a 30,000-episode table took several GB to build and send.
ICON_ROWS_MAX = 1000


# The cached Runs themselves (not copies): cache_data would unpickle every run
# (hundreds of MB) on each rerun.  Nothing writes to them; tabs filter copies.  A
# plain dict rather than caching ``cached_run``, which may draw the progress bar.
# Values are (run, the pkg_mtime it was last served under, its memory in bytes).
# The runs held stay within ``memory.run_budget()`` (see ``load_runs``).
@st.cache_resource
def run_store() -> collections.OrderedDict[tuple, tuple[Run, float, int]]:
    return collections.OrderedDict()


@st.cache_resource
def run_locks() -> tuple[threading.Lock, dict[tuple, threading.Lock]]:
    """A lock per run key, so sessions loading the same run at once read it once
    (the second waits for the first); different runs still load in parallel.
    The first lock guards the store and the dict of locks."""
    return threading.Lock(), {}


@st.cache_resource
def run_sizes() -> dict[str, int]:
    """Each run's memory (by path) when last loaded, without its tables, to judge
    before loading it again whether it fits.  Runs never loaded are guessed from
    their episodes.csv files."""
    return {}


@st.cache_resource
def shown_keys() -> set:
    """Store keys of the runs the latest rerun shows: their tables are kept when
    making room for another table, so a view over several runs doesn't thrash."""
    return set()


def run_size_guess(path: str) -> int:
    """Memory the run at ``path`` will take once loaded (tables are read later)."""
    return run_sizes().get(path) or memory.estimate(episodes_bytes(path))


def held_size(entry: tuple) -> int:
    """Memory a stored run takes now: as loaded, plus the tables read since."""
    return entry[2] + memory.estimate(entry[0].read_bytes())


def _sub_runs(run: Run) -> list[Run]:
    return [run, *run.splits()]  # with the replays read so far


def _served(store: collections.OrderedDict, key: tuple, pkg_mtime: float) -> Run:
    """The stored run for ``key``, its memoised frames dropped if the package
    changed since it was last served (they were built by the old code)."""
    store.move_to_end(key)
    entry = store[key]
    if entry[1] != pkg_mtime:
        for r in _sub_runs(entry[0]):
            r._memo.clear()
        store[key] = (entry[0], pkg_mtime, entry[2])
    return entry[0]


def _make_room(store: collections.OrderedDict, budget: int | None, keep: set) -> None:
    """Drop the least recently used runs until the store is within ``budget``,
    never one in ``keep`` (the runs this rerun shows).  Guarded by the caller."""
    if budget is None:
        return
    used = sum(held_size(v) for v in store.values())
    for k in [k for k in store if k not in keep]:
        if used <= budget:
            break
        used -= held_size(store.pop(k))


def make_room_for_table(run: Run, table: str, nbytes: int) -> None:
    """``loader.table_read_hook``: before a table is read, free what it needs from
    the memory set aside for runs: first the tables of runs not shown, then those
    runs.  Refuses (``TableBudgetError``) if the runs shown need more than that."""
    budget = memory.run_budget()
    if budget is None:
        return
    need = memory.estimate(nbytes)
    store = run_store()
    guard, _ = run_locks()
    with guard:
        used = sum(held_size(v) for v in store.values())
        keep = shown_keys()
        for k, v in list(store.items()):
            if used + need <= budget:
                return
            if k not in keep and v[0].read_bytes():
                used -= memory.estimate(v[0].unload())
        _make_room(store, budget - need, keep)
        used = sum(held_size(v) for v in store.values())
    if used + need > budget:
        raise TableBudgetError(run.name, table, need, budget)


loader.table_read_hook = make_room_for_table


def cached_run(
    path: str, root: str, mtime: float, pkg_mtime: float, progress=None, keep: set | None = None
) -> Run:
    # mtime: re-written runs reload, reading only what was appended if this run is
    # still held from before.  The key holds the loading code's version, not
    # pkg_mtime, so editing the rest of the package keeps loaded runs (dropping only
    # what was memoised on them).  Runs unchanged since a previous session come
    # from the disk cache.  ``keep``: keys of runs this rerun already holds, which
    # making room for this one mustn't drop; this run's key is added to it.
    keep = set() if keep is None else keep
    store, key = run_store(), (path, root, mtime, runcache.code_version())
    guard, locks = run_locks()
    with guard:
        if key in store:
            keep.add(key)
            return _served(store, key, pkg_mtime)
        lock = locks.setdefault(key, threading.Lock())
    with lock:
        with guard:  # loaded by another session while this one waited
            if key in store:
                keep.add(key)
                return _served(store, key, pkg_mtime)
            older = [v[0] for k, v in store.items() if k[:2] == key[:2] and k[3] == key[3]]
            # room for it before reading it, so the peak stays within the budget
            budget = memory.run_budget()
            if budget is not None:
                _make_room(store, budget - run_size_guess(path), keep)
        before = memory.process_rss()
        try:
            run = runcache.load_run_cached(
                path, root, runcache.cache_dir(), progress, older[-1] if older else None
            )
        except BaseException:
            with guard:
                locks.pop(key, None)
            raise
        after = memory.process_rss()
        # what reading it took, unless that is less than its logs suggest (memory
        # freed meanwhile, or another session loading at the same time)
        size = max(memory.estimate(episodes_bytes(path)), (after or 0) - (before or 0))
        if older:
            size = max(size, run_sizes().get(path, 0))  # extended in place: as before
        with guard:  # stored and unlocked together, so no session loads it twice
            for old in [k for k in store if k[:2] == key[:2]]:
                del store[old]  # an older copy of this run: don't hold both
            store[key] = (run, pkg_mtime, size)
            run_sizes()[path] = size
            locks.pop(key, None)
            keep.add(key)
            _make_room(store, memory.run_budget(), keep)
    return run


@st.cache_data(max_entries=4096, show_spinner=False)
def cached_summary(path: str, name: str, mtime: float) -> Run | None:
    """``find.run_summary``; mtime: re-read once the run has logged more."""
    return find.run_summary(path, name)


@st.cache_resource
def filter_lock() -> threading.Lock:
    return threading.Lock()


FILTERED = ("filtered",)  # where a run keeps its filtered copies, in its _memo


def filtered_run(
    run: Run, start: float, end: float, decks: tuple, stakes: tuple, rounds: tuple | None
) -> Run:
    """``metrics.filter_episodes``, kept across reruns: the same filters give the
    same Run, so what tabs memoise on it (``loader.memo``) is computed once.  The
    copies live on ``run`` (its ``FILTERED_KEPT`` latest), so they go when it
    leaves the run store rather than holding it in memory after."""
    args = (start, end, decks, stakes, rounds)
    with filter_lock():
        kept = run._memo.setdefault(FILTERED, collections.OrderedDict())
        if args in kept:
            kept.move_to_end(args)
            return kept[args]
    out = metrics.filter_episodes(
        run,
        start,
        end,
        decks=list(decks),
        stakes=list(stakes),
        eval_rounds=list(rounds) if rounds is not None else None,
    )
    if out is run:  # nothing filtered out: no copy to keep
        return run
    with filter_lock():
        kept[args] = out
        while len(kept) > FILTERED_KEPT:
            kept.popitem(last=False)
    return out


@st.cache_resource(show_spinner="Checking chart export…")
def chart_export_problem() -> str | None:
    """None if PNG/SVG rendering works here, else the reason (checked once per server)."""
    return export.export_problem()


# pkg_mtime in these keys: objects built by old code are never reused after a change


@st.cache_resource
def cached_sprites(archive: str, pkg_mtime: float, scale: int = 2) -> Sprites:
    return Sprites(Path(archive), scale=scale)


@st.cache_resource(show_spinner="Reading joker text from the game…")
def cached_game_text(archive: str, pkg_mtime: float) -> GameText | None:
    try:
        return GameText.load(Path(archive))
    except (KeyError, OSError, UnicodeError):  # unexpected game version/layout
        return None


def need(df, what: str, logs_missing: bool = False) -> bool:
    """Show a note and return False if a table is missing or empty."""
    if df is None or len(df) == 0:
        hint = " (needs steps.jsonl / rounds.jsonl)" if logs_missing else ""
        st.info(f"No {what}{hint}.")
        return False
    return True


CHARTS: dict[str, object] = {}
CHART_DESCRIPTIONS: dict[str, str] = {}  # same keys as CHARTS
INTEREST_EFFICIENCY = (
    "Interest actually paid at cash-out as a share of the most the interest cap allows, by "
    "ante (exact, from the cash-out breakdown). 100% means the agent always held enough money "
    "to hit the cap."
)


LOG_VERSIONS = {
    2: "cash-out breakdown and shops.jsonl",
    3: "legal actions per step",
    4: "eval episodes on fixed seeds (--eval-every)",
    5: "cash-out money from tags",
    6: "highest ante reached (max_ante); a sell's item_cost is its sale value",
    7: "eval hand and deck logs",
    8: "upstream Jackdaw engine",
    9: "held-out jokers",
    10: "logged offers of chosen jokers (offers.jsonl)",
    11: "DQN function tables",
    12: "round features",
    13: "relative suits",
    14: "a test set (test/)",
    15: "scaling jokers' values",
    16: "deck suits",
    17: "editions and stickers on shop items and held jokers, the shop at each decision, "
    "endless mode",
}


def need_new(df, what: str, version: int = 2) -> bool:
    """Like need(), for data only runs from the newer training script log."""
    if df is None or len(df) == 0:
        st.info(
            f"No {what}: needs a run made with the current training script "
            f"(log_version {version}: {LOG_VERSIONS[version]})."
        )
        return False
    return True


def chart(fig, name: str, description: str) -> None:
    """Show a figure and what it shows, register both for 'download all', and
    offer PNG/SVG of it."""
    key = export.slug(name)
    CHARTS[key] = fig
    CHART_DESCRIPTIONS[key] = description
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
    st.caption(description)
    with st.popover("⬇ Export chart"):
        if problem := chart_export_problem():
            st.error(f"Chart export unavailable: {problem}")
            return
        for fmt in ("png", "svg"):
            st.download_button(
                fmt.upper(),
                data=functools.partial(export.render, fig, fmt),
                file_name=f"{key}.{fmt}",
                mime="image/png" if fmt == "png" else "image/svg+xml",
                key=f"dl_{key}_{fmt}",
                on_click="ignore",
            )


def table(df, keys: dict[str, str] | None = None, **kwargs) -> None:
    """A dataframe with key columns and headers in the current naming mode."""
    st.dataframe(names.frame(df, keys), hide_index=True, width="stretch", **kwargs)


def icon_table(df, item_cols: list[str], keys: dict[str, str] | None = None, **kwargs) -> None:
    """``table`` with an icon strip before each item-list column in ``item_cols``
    (held jokers, consumables, ...).  The names stay next to it; without the game
    files, or with more than ``ICON_ROWS_MAX`` rows, there are only the names."""
    present = [c for c in item_cols if c in df]
    if not sprites or not present or len(df) > ICON_ROWS_MAX:
        if sprites and present:
            st.caption(f"Icons are shown for up to {ICON_ROWS_MAX:,} rows: filter to see them.")
        table(df, keys, **kwargs)
        return
    df = df.copy()
    config = dict(kwargs.pop("column_config", None) or {})
    for col in present:
        icons = f"{col}_icons"
        df.insert(
            df.columns.get_loc(col), icons, df[col].map(lambda v: sprites.strip_uri(split_items(v)))
        )
        config[names.column(icons)] = st.column_config.ImageColumn(
            "", help=names.column(col), width="medium"
        )
    table(df, keys, column_config=config, row_height=60, **kwargs)


def browse_runs_dir() -> None:
    """Button callback: open a native folder dialog and store the choice."""
    try:
        chosen = pick_directory(st.session_state.runs_dir, "Choose runs folder")
    except PickerUnavailable as exc:
        st.session_state.picker_error = str(exc)
        return
    if chosen:
        st.session_state.runs_dir = chosen


def browse_game() -> None:
    try:
        chosen = pick_directory(st.session_state.game_path or "~", "Choose Balatro install folder")
    except PickerUnavailable as exc:
        st.session_state.picker_error = str(exc)
        return
    if chosen:
        st.session_state.game_path = chosen


def icon_uri(key: str | None) -> str | None:
    return sprites.data_uri(key) if sprites else None


def modal_html(markup: str) -> None:
    """Tiles + click-for-details modals (they need a little script to open)."""
    st.html(markup, unsafe_allow_javascript=True)


@functools.cache  # once per script run: several tables and grids ask for it
def joker_cards() -> list[collection.JokerCard]:
    """Every joker as a collection card, with stats from the selected runs."""
    return collection.build_cards(
        metrics.joker_stats(runs),
        cached_game_text(str(archive), pkg_mtime) if archive else None,
        cached_sprites(str(archive), pkg_mtime, 1) if archive else None,
        metrics.joker_held_by_run(runs),
    )


def with_joker_values(df: pd.DataFrame, jokers: str) -> pd.DataFrame:
    """``df`` with its ``joker_state`` column (log_version 15: each held joker's
    values, if it scales) as readable text, "Green Joker +4 Mult", in
    ``joker_values`` right after the ``jokers`` column."""
    if "joker_state" not in df or jokers not in df:
        return df
    text = [
        cards.state_summary(_cell_list(k), _cell_list(s), names.item)
        for k, s in zip(df[jokers], df["joker_state"], strict=True)
    ]
    out = df.drop(columns="joker_state")
    out.insert(out.columns.get_loc(jokers) + 1, "joker_values", text)
    return out


def _cell_list(value) -> list | None:
    """A list cell as a list (Arrow-backed cells can be arrays); None if missing."""
    if value is None or isinstance(value, str | dict | float):
        return None
    return list(value)


def joker_grid(items: list[tuple[str, str]], prefix: str) -> None:
    """(joker key, caption HTML) pairs as clickable tiles that open each joker's modal."""
    by_key = {c.key: c for c in joker_cards()}
    shown = [(by_key[k], cap) for k, cap in items if k in by_key]
    modal_html(
        collection.mini_grid_html([c for c, _ in shown], names, prefix, [cap for _, cap in shown])
    )


def item_table(
    df: pd.DataFrame, item_cols: list[str], prefix: str, keys: dict[str, str] | None = None
) -> None:
    """A table whose item icons (jokers, consumables, vouchers, packs) open the same
    details modals as elsewhere: jokers their collection card, the rest their effect
    text and how often the selected runs bought, sold or used them."""
    cards = {c.key: c for c in joker_cards()}
    text = cached_game_text(str(archive), pkg_mtime) if archive else None
    counts = metrics.item_actions(runs)

    def dialog(key: str, did: str) -> str:
        if key in cards:
            return collection.joker_dialog(cards[key], names, did)
        stats = []
        if key in counts.index:
            c = counts.loc[key]
            stats = [
                (label, str(int(c[col])))
                for label, col in (("Times bought", "bought"), ("Times sold", "sold"),
                                   ("Times used in the shop", "used"))
                if c[col]
            ]  # fmt: skip
        return details.item_dialog(
            key, did, names, text.item(key) if text else None, icon_uri(key), stats
        )

    st.caption("Click an item for its details.")
    modal_html(details.item_table_html(df, item_cols, names, prefix, dialog, icon_uri, keys))


def deck_infos(keys: list[str]) -> list[details.DeckInfo]:
    deck_table = metrics.by_deck(runs)
    text = cached_game_text(str(archive), pkg_mtime) if archive else None
    return [
        details.deck_info(k, deck_table, text, icon_uri, metrics.deck_top_jokers(runs, k))
        for k in keys
    ]


st.set_page_config(page_title="Balatro Run Analysis", layout="wide")

if "runs_dir" not in st.session_state:
    st.session_state.runs_dir = str(cli_runs_dir())
if "manifest_path" not in st.session_state:
    st.session_state.manifest_path = cli_manifest()
if "game_path" not in st.session_state:
    found = find_balatro()
    st.session_state.game_path = str(found) if found else ""
if "pretty" not in st.session_state:
    st.session_state.pretty = st.query_params.get("names", "game") != "keys"
names = Names(pretty=st.session_state.pretty)  # the toggle itself is under Settings
pkg_mtime = package_mtime()
archive = game_archive(st.session_state.game_path) if st.session_state.game_path else None
sprites = cached_sprites(str(archive), pkg_mtime) if archive else None
st.html(theme.css(sprites.font_uri() if sprites else None))
st.title("Balatro run analysis")


def settings(expanded: bool = False) -> None:
    """Things set once and left alone: naming, where runs and the game are."""
    with st.expander("Settings", icon=":material/settings:", expanded=expanded):
        st.toggle(
            "In-game names",
            key="pretty",
            help="On: names as shown in the game (for presenting). "
            "Off: exact keys like j_greedy_joker (for analysis).",
        )
        st.text_input(
            "Run manifest",
            key="manifest_path",
            help="runs/manifest.json from balatro-shop-rl's scripts/run_manifest.py: runs "
            "in several folders (runs/ and archives) and named studies. Empty: every run "
            "under the runs folder below.",
        )
        st.text_input("Runs folder", key="runs_dir", help="Used when there's no manifest.")
        st.button("Browse…", on_click=browse_runs_dir, key="browse_runs")
        st.text_input("Balatro install", key="game_path", help="Folder or Balatro.exe/.love")
        st.button("Browse…", on_click=browse_game, key="browse_game")
        if err := st.session_state.pop("picker_error", None):
            st.error(f"{err}. Type the path instead.")
        if sprites:
            st.caption(f"Card art, icons and font from {archive}")
        else:
            st.caption("Balatro not found: showing names only.")


# The sidebar's Episodes switch: which of a run's episodes every tab shows (None:
# the training episodes themselves), in the order offered; Test first, the default
PHASE_SPLITS = {"Test": loader.TEST_DIR, "Evaluation": loader.EVAL_DIR, "Training": None}
PHASE_CAPTIONS = {
    "Test": "Results: the final policy on the fixed test games.",
    "Evaluation": "Learning: greedy eval rounds during training. Not for ranking runs.",
    "Training": "Diagnostics: the episodes trained on, with exploration.",
}
# "Replay: <tag>": a replay/<tag>/ split (the test games played again, maybe on
# other decks or stakes, or endless), offered for the tags every picked run has
REPLAY_PHASE = "Replay: "


def phase_split(phase: str) -> str | None:
    """The ``Run.split`` name behind an Episodes choice (None: training)."""
    if phase.startswith(REPLAY_PHASE):
        return f"{loader.REPLAY_DIR}/{phase.removeprefix(REPLAY_PHASE)}"
    return PHASE_SPLITS[phase]


def shared_replay_tags(run_list: list[Run]) -> list[str]:
    """The replay tags every run in ``run_list`` has."""
    tags = [set(r.replays.tags()) if r.replays is not None else set() for r in run_list]
    return sorted(set.intersection(*tags)) if tags else []


def replay_caption(info: dict | None) -> str:
    """What a replay played, from its replay.json."""
    if not info:
        return "The test games played again."
    parts = []
    if info.get("decks"):
        parts.append("decks " + ", ".join(names.deck(d) for d in info["decks"]))
    if info.get("stakes"):
        parts.append("stakes " + ", ".join(names.stake(x) for x in info["stakes"]))
    if info.get("endless"):
        parts.append("endless (play on past ante 8)")
    games = f"{info['games']} " if info.get("games") else ""
    return f"The {games}test games played again" + (": " + "; ".join(parts) if parts else "") + "."


def episodes_shown() -> str:
    """Which episodes the tabs show, for captions and the export's README."""
    if replay_tag is not None:
        return f"the test games replayed ({replay_tag})"
    if test_mode:
        return "the test games"
    if eval_mode:
        return f"eval round {eval_rounds[0]}" if eval_rounds else "every eval round"
    return f"episodes {ep_range[0]} to {ep_range[1]}"


def phase_switch(phases: list[str], captions: dict[str, str]) -> str:
    """The sidebar's Episodes switch over ``phases``, with a line on what it's for."""
    phase = st.segmented_control(
        "Episodes",
        phases,
        required=True,
        key="phase",
        help="**Test**: the final policy on one fixed set of games (500 for balatro-shop-rl, "
        "TEST_0..), the same for every run and never used during training. Report results "
        "from these. **Evaluation**: greedy episodes on the fixed eval seeds every "
        "--eval-every episodes, for learning curves; too few games per round to rank runs "
        "on. **Training**: the episodes trained on, with exploration, for diagnostics. "
        "**Replay**: the test games played again by balatro-shop-rl's replay_test.py "
        "(replay/<tag>/), e.g. with the current logging (retest) or on other decks or stakes. "
        "Test and eval runs play the same games, so comparisons pair game by game.",
    )
    st.caption(captions[phase])
    return phase


NO_STUDY = "None"


def study_picker(manifest: studies.Manifest, run_dirs: dict) -> str | None:
    """The sidebar's Study picker: choosing a study picks its runs and compares
    groups.  Returns the study chosen (None: none)."""

    def pick() -> None:
        chosen = st.session_state.study
        if chosen != NO_STUDY:
            st.session_state.picked_runs = [n for n in manifest.study_runs(chosen) if n in run_dirs]
            st.session_state.compare_by = "Groups"

    chosen = st.selectbox(
        "Study",
        [NO_STUDY, *manifest.studies],
        key="study",
        on_change=pick,
        format_func=lambda s: manifest.studies.get(s, {}).get("title", s),
        help="A comparison named in the manifest (balatro-shop-rl's analysis/studies.toml): "
        "picks its groups' runs, and the Overview shows its tables, paired by seed.",
    )
    if chosen == NO_STUDY:
        return None
    for problem in manifest.study_problems(chosen):
        st.caption(f":material/warning: {problem}")
    return chosen


# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    root = Path(st.session_state.runs_dir).expanduser()
    manifest = None
    if mpath := st.session_state.manifest_path.strip():
        try:
            mfile = Path(mpath).expanduser()
            manifest = cached_manifest(str(mfile), mfile.stat().st_mtime)
        except (OSError, ValueError, KeyError) as e:
            st.warning(f"Couldn't read the run manifest ({e}); listing the runs folder instead.")
    if manifest is not None:
        # runs named by the manifest, wherever they are; names are the run folders'
        run_dirs = {
            n: d for n in manifest.runs if (d := manifest.run_dir(n) / "episodes.csv").is_file()
        }
        run_dirs = {n: d.parent for n, d in run_dirs.items()}
        source = str(manifest.path)
        if missing := len(manifest.runs) - len(run_dirs):
            st.caption(f"{missing} run(s) in the manifest not found on disk.")
    else:
        run_paths = find_runs(root)
        run_dirs = {str(p.relative_to(root)) if p != root else p.name: p for p in run_paths}
        source = str(root)
    if not run_dirs:
        st.warning(f"No runs (folders with episodes.csv) under {root}. Set the runs folder below.")
        settings(expanded=True)
        st.stop()

    def run_root(name: str) -> str:
        """What the run's name is relative to (its own folder's parent for a manifest)."""
        return str(run_dirs[name].parent) if manifest is not None else str(root)

    def run_mtime(path: Path) -> float:
        files = [path / "episodes.csv", *(path / d / "episodes.csv" for d in loader.SPLIT_DIRS)]
        return max(f.stat().st_mtime for f in files if f.is_file())

    # Listing runs reads only their episodes.csv and config.json: only the runs
    # picked below are loaded
    run_index = find.run_index(
        {n: cached_summary(str(p), n, run_mtime(p)) for n, p in run_dirs.items()}
    )

    # Every run starts selected; runs that appear later (Reload runs) are added
    if st.session_state.get("_runs_root") != source:
        st.session_state.picked_runs = list(run_dirs)
    else:
        known = st.session_state.get("_known_runs", [])
        kept = [n for n in st.session_state.get("picked_runs", []) if n in run_dirs]
        st.session_state.picked_runs = kept + [n for n in run_dirs if n not in known]
    st.session_state._runs_root, st.session_state._known_runs = source, list(run_dirs)

    study = None
    if manifest is not None and manifest.studies:
        study = study_picker(manifest, run_dirs)

    shown_runs = find.find_runs_in(
        run_index,
        st.session_state.get("run_search", ""),
        st.session_state.get("run_sort", "Name"),
        st.session_state.get("run_shop") or None,
        st.session_state.get("run_play") or None,
        st.session_state.get("run_group") or None,
    )
    # Selected runs the filters hide stay selected (and say so) rather than vanish
    hidden = [n for n in st.session_state.picked_runs if n not in shown_runs]
    picked = st.multiselect("Runs", shown_runs + hidden, key="picked_runs")
    if hidden:
        st.caption(
            f"{len(hidden)} selected run{'s' if len(hidden) > 1 else ''} hidden by the "
            "filters in Find runs: " + ", ".join(hidden)
        )
    if st.button(
        "Reload runs",
        icon=":material/refresh:",
        help="Pick up new runs, and new episodes of runs still training: only what "
        "was logged since the last load is read",
    ):
        st.rerun()

    def pick_runs(names_: list[str]) -> None:
        st.session_state.picked_runs = names_

    with st.expander("Find runs", icon=":material/search:"):
        st.text_input("Search runs", key="run_search", placeholder="part of a run name")
        st.selectbox("Sort by", list(find.RUN_SORTS), key="run_sort")
        for label, col, key in (
            ("Shop agent", "shop_agent", "run_shop"),
            ("Play agent", "play_agent", "run_play"),
            ("Group", "group", "run_group"),
        ):
            values = sorted(run_index[col].unique())
            if len(values) > 1:
                st.multiselect(
                    label,
                    values,
                    key=key,
                    placeholder="All",
                    format_func=lambda v: v or "(no group)",
                )
        st.dataframe(
            run_index.set_index("run")
            .loc[shown_runs, ["mean_ante", "scored_on", "episodes", "group"]]
            .rename(columns=names.column),
            column_config={names.column("mean_ante"): st.column_config.NumberColumn(format="%.2f")},
            height=min(36 * (len(shown_runs) + 1) + 2, 300),
        )
        st.caption(
            "Mean ante and win rate are on the 500 test games where a run has them "
            "(*Scored on* test), otherwise on its training episodes, which aren't comparable."
        )
        b1, b2, b3 = st.columns(3)
        b1.button(
            "Best per group",
            on_click=pick_runs,
            args=(find.best_per_group(run_index[run_index["run"].isin(shown_runs)]),),
            help="The highest mean-ante run of each group shown (the best run if none are "
            "grouped). A diagnostic: picking the best seed by its score selects for luck, so "
            "report groups, not the best run.",
        )
        b2.button("All shown", on_click=pick_runs, args=(shown_runs,))
        b3.button("Clear", on_click=pick_runs, args=([],))
    if not picked:
        st.info("Pick at least one run above.")
        settings()
        st.stop()

    # A progress bar over the picked runs' log bytes, only shown once a run is
    # actually read (from disk or the disk cache): runs already in memory never report.
    load_slot = st.empty()
    run_bytes = {n: max(episodes_bytes(run_dirs[n]), 1) for n in picked}
    total_bytes = sum(run_bytes.values())
    bytes_before = 0
    held: set = set()  # store keys of the runs this rerun shows: kept while loading the rest

    def load(i: int, name: str) -> Run | None:
        """The run, or None if it has no episodes yet (still training?) or can't be read."""

        def progress(label: str, fraction: float) -> None:
            done = (bytes_before + fraction * run_bytes[name]) / total_bytes
            load_slot.progress(
                min(done, 1.0),
                text=f"Loading run {i}/{len(picked)}: {name} · {label} "
                f"({run_bytes[name] / 1e6:,.0f} MB)",
            )

        path = str(run_dirs[name])
        try:
            run = cached_run(
                path, run_root(name), run_mtime(run_dirs[name]), pkg_mtime, progress, held
            )
        except (ValueError, OSError):  # e.g. EmptyDataError: training hasn't logged yet
            return None
        return None if run.episodes.empty else run

    # Picked runs are loaded in order while they fit the memory budget; the rest
    # are left out (and named) rather than overrun the machine's or the job's memory
    budget = memory.run_budget()
    loaded, over_budget, used = {}, [], 0
    for i, n in enumerate(picked, 1):
        size = run_size_guess(str(run_dirs[n]))
        if budget is not None and loaded and used + size > budget:
            over_budget.append(n)
        else:
            loaded[n] = load(i, n)
            used += run_sizes().get(str(run_dirs[n]), size)
        bytes_before += run_bytes[n]
    load_slot.empty()
    keys = shown_keys()
    keys.clear()
    keys.update(held)

    runs = [r for r in loaded.values() if r is not None]
    not_ready = [n for n, r in loaded.items() if r is None]
    if not_ready:
        st.warning(
            f"Skipped {', '.join(not_ready)}: no episodes yet (still training?) or "
            "unreadable. **Reload runs** once it has logged some."
        )
    if over_budget:
        wanted = used + sum(run_size_guess(str(run_dirs[n])) for n in over_budget)
        st.warning(
            f"Not loaded, to stay within memory: {', '.join(over_budget)}. The runs "
            f"picked need about {wanted / 1e9:,.0f} GB "
            f"and {budget / 1e9:,.0f} GB is set aside for runs. Pick fewer runs, give the "
            f"dashboard more memory (Slurm `--mem`), or set `{memory.ENV}` (GB).",
            icon=":material/memory:",
        )
    if not runs:
        settings()
        st.stop()
    train_runs = runs

    # Test or eval episodes (greedy, fixed games) replace the training episodes in
    # every tab.  Test is the default: it's what results are reported on.
    # Replays play the test games again: shown like test games.
    with_split = {s: [r for r in runs if getattr(r, s) is not None] for s in loader.SPLIT_DIRS}
    phases = [p for p, s in PHASE_SPLITS.items() if s is not None and with_split[s]]
    phases += [REPLAY_PHASE + t for t in shared_replay_tags(runs)] + ["Training"]
    if st.session_state.get("phase") not in phases:  # Test if any run has it, else Training
        st.session_state.phase = phases[0] if phases[0] == "Test" else "Training"
    phase = "Training"  # no switch when there are only training episodes
    split = None
    replay_tag = None
    if len(phases) > 1:
        captions = dict(PHASE_CAPTIONS)
        for p in phases:
            if p.startswith(REPLAY_PHASE):
                first = runs[0].split(phase_split(p))
                captions[p] = replay_caption(first.replay if first is not None else None)
        phase = phase_switch(phases, captions)
        split = phase_split(phase)
        if phase.startswith(REPLAY_PHASE):
            replay_tag = phase.removeprefix(REPLAY_PHASE)
    eval_mode = split == loader.EVAL_DIR
    test_mode = split == loader.TEST_DIR or replay_tag is not None
    if split is not None:
        picked = [(r, r.split(split)) for r in runs]
        if missing := [r.name for r, s in picked if s is None]:
            st.caption(f"No {split} episodes, not shown: " + ", ".join(missing))
        runs = [s for _, s in picked if s is not None]

    filter_summary = st.empty()  # filled once the filters below are read
    with st.expander("Filter", icon=":material/filter_list:"):
        all_decks = sorted({d for r in runs for d in r.episodes["deck"]})
        all_stakes = sorted({s for r in runs for s in r.episodes["stake"]})
        decks = st.multiselect("Decks", all_decks, default=all_decks, format_func=names.deck)
        stakes = st.multiselect("Stakes", all_stakes, default=all_stakes, format_func=names.stake)

        max_ep = int(max(r.episodes["episode"].max() for r in runs))
        eval_rounds = None
        if test_mode:
            ep_range = (0, max_ep)
        elif eval_mode:
            ep_range = (0, max_ep)
            all_rounds = sorted({int(x) for r in runs for x in r.episodes["eval_round"]})
            eval_round = st.selectbox(
                "Eval round",
                ["All", *all_rounds],
                index=len(all_rounds),
                help="Eval round n is played after the training episodes shown in the "
                "Overview eval chart. The last round scores the final policy.",
            )
            eval_rounds = None if eval_round == "All" else [eval_round]
        else:
            ep_range = st.slider("Episodes", 0, max(max_ep, 1), (0, max_ep))
        window = st.slider(
            "Rolling window",
            1,
            max(2, max_ep + 1),
            min(50, max_ep + 1),
            help="Episodes averaged per point of the learning curves",
        )

    def _which(chosen: list, every: list, f, plural: str) -> str:
        if not chosen:
            return f"no {plural}"
        if len(chosen) == len(every):
            return f"all {plural}" if len(every) > 1 else f(every[0])
        return ", ".join(f(x) for x in chosen)

    scope = episodes_shown()
    filter_summary.caption(
        f"Showing {_which(decks, all_decks, names.deck, 'decks')}, "
        f"{_which(stakes, all_stakes, names.stake, 'stakes')}, {scope}."
    )

    settings()
    export_area = st.container()  # filled at the end, once every chart is registered


def filtered(r: Run, start: float, end: float, rounds: list[int] | None = None) -> Run:
    return filtered_run(
        r, start, end, tuple(decks), tuple(stakes), tuple(rounds) if rounds is not None else None
    )


runs = [filtered(r, *ep_range, eval_rounds) for r in runs]
RETEST = "retest"  # the replay tag of the test games again, with the current logging


def item_runs() -> tuple[list[Run], list[str]]:
    """The runs for item-level views (shop purchases, stickers): on the test games,
    each run's ``retest`` replay instead when it has one (the same games, logged with
    editions and stickers).  Also the names swapped."""
    if split != loader.TEST_DIR:
        return runs, []
    shown = {r.name: r for r in runs}
    out, swapped = [], []
    for orig in train_runs:
        if orig.name not in shown:
            continue
        retest = orig.split(f"{loader.REPLAY_DIR}/{RETEST}")
        if retest is None:
            out.append(shown[orig.name])
        else:
            out.append(filtered(retest, 0, float("inf")))
            swapped.append(orig.name)
    return out, swapped


def item_runs_caption(swapped: list[str]) -> None:
    if swapped:
        st.caption(
            f"From the **{RETEST}** replay (the same test games, logged with editions and "
            "stickers) for " + ", ".join(swapped) + "."
        )


def shop_purchases() -> None:
    """Buy rate of each kind of shop item, as balatro-shop-rl's special_buys.py."""
    item_list, swapped = item_runs()
    if not need(
        [r for r in item_list if r.has("shops") and r.has("steps")] or None,
        "shop and step logs",
        no_steps,
    ):
        return
    item_runs_caption(swapped)
    if old := [r.name for r in item_list if not purchases.has_editions(r)]:
        need_new(None, "editions, stickers or joker slots for " + ", ".join(old), version=17)
    c1, c2 = st.columns(2)
    by = c1.selectbox(
        "Split by",
        [None, "deck", "stake"],
        format_func=lambda c: "Nothing" if c is None else names.column(c),
        key="buys_by",
    )
    min_legal = c2.number_input(
        "Hide rows with fewer legal listings than",
        0,
        value=purchases.MIN_LEGAL,
        key="buys_min_legal",
    )
    rates = purchases.buy_rates(grouped(item_list), by)
    if not need(rates, "shop listings"):
        return
    for part in sorted(rates["part"].unique(), key=str):
        if by is not None:
            st.markdown(f"**{names.column(by)}: {names.of(by)(part)}**")
        table(purchases.rate_table(rates, part, int(min_legal)))
    st.caption(
        "Each listing (entering a shop, or a reroll) counts once per item. Buy rate = listings "
        "bought ÷ listings where buying the item was legal (affordable, with room) at some "
        "decision while it was on offer, per run, then mean ± SD over the group's runs; (n) = "
        "legal listings pooled. Picks from packs aren't shop buys. Slots full: no free joker "
        "slot at the listing. As balatro-shop-rl's scripts/special_buys.py."
    )
    st.download_button(
        "Buy rates (CSV)",
        data=rates.to_csv(index=False),
        file_name="buy_rates.csv",
        mime="text/csv",
        key="dl_buy_rates",
        on_click="ignore",
    )


def grouped(run_list: list[Run]) -> dict[str, list[Run]]:
    """``run_list`` by group when the Overview compares groups, else one per run."""
    group_map = stats.groups(run_list)
    if st.session_state.get("compare_by") == "Groups" and len(group_map) < len(run_list):
        return group_map
    return {r.name: [r] for r in run_list}


eval_runs = [filtered(r.eval, 0, float("inf")) for r in train_runs if r.eval is not None]
run_names = [r.name for r in runs]
by_name = {r.name: r for r in runs}
# whether runs have these logs, without reading them (tables are read when a view needs them)
no_steps = any(not r.has("steps") for r in runs)
no_rounds = any(not r.has("rounds") for r in runs)

EPISODE_FILTERS = ("ep_decks", "ep_stakes", "ep_antes", "ep_outcome", "ep_jokers")
TOP_JOKERS_DEFAULT = 15
top = TOP_JOKERS_DEFAULT  # the Jokers tab's slider; the export reads it even when that tab is shut
# Deep link ?run=<name>&episode=<n> opens that episode in the Episode explorer
if not st.session_state.get("_deep_link_read"):
    st.session_state._deep_link_read = True
    if st.query_params.get("run") in run_names:
        st.session_state.explorer_run = st.query_params["run"]
        st.session_state.main_tab = "Episode explorer"
        if (ep := st.query_params.get("episode", "")).isdigit():
            st.session_state.explorer_episode = int(ep)

# Tabs report which one is open, so only the open tab builds its charts
tabs = st.tabs(
    [
        "Overview",
        "Decks",
        "Actions",
        "Economy",
        "Jokers",
        "Joker collection",
        "Rounds",
        "Hands",
        "Episode explorer",
    ],
    key="main_tab",
    on_change="rerun",
)


def show_episode(run_name: str, episode: int) -> None:
    """Button callback: open ``episode`` of ``run_name`` in the Episode explorer."""
    for key in EPISODE_FILTERS:
        st.session_state.pop(key, None)
    st.session_state.explorer_run = run_name
    st.session_state.explorer_episode = episode
    st.session_state.main_tab = "Episode explorer"


RUN_SHEETS = 6  # run sheets shown before the rest fold away


def run_sheets() -> None:
    """Each selected run as the game's run-over screen, with buttons into its best
    and worst episode: where to start reading."""
    # A run can have no episodes under the filters, e.g. one still training that
    # hasn't played the chosen eval round yet
    sheet_runs = [r for r in runs if not r.episodes.empty]
    if empty := [r.name for r in runs if r.episodes.empty]:
        st.caption(
            "No episodes under the current filters, not shown: "
            + ", ".join(empty)
            + "."
            + (" Still training? Try an earlier eval round under Filter." if eval_mode else "")
        )
    if not sheet_runs:
        return
    ci = stats.summary_ci(sheet_runs)
    ci = ci[ci["outcome"] == "ante reached"].set_index("run") if len(ci) else ci
    means = {r.name: r.episodes["ante"].mean() for r in sheet_runs}
    best_run = max(means, key=means.get) if len(sheet_runs) > 1 else None
    ranked = sorted(sheet_runs, key=lambda r: -means[r.name])  # best first
    shown, rest = ranked[:RUN_SHEETS], ranked[RUN_SHEETS:]
    run_sheet_grid(shown, ci, means, best_run)
    if rest:
        with st.expander(f"The other {len(rest)} runs, by mean ante"):
            run_sheet_grid(rest, ci, means, best_run)


def run_sheet_grid(sheet_runs: list[Run], ci, means: dict, best_run: str | None) -> None:
    for start in range(0, len(sheet_runs), 3):
        for col, r in zip(st.columns(3), sheet_runs[start : start + 3], strict=False):
            e = r.episodes
            with col:
                st.html(
                    replay.run_sheet_html(
                        r.name,
                        r.config,
                        means[r.name],
                        (ci.at[r.name, "ci_low"], ci.at[r.name, "ci_high"])
                        if r.name in ci.index
                        else None,
                        e["won"].mean(),
                        int(e["ante"].max()),
                        len(e),
                        sorted(e["deck"].unique()),
                        names,
                        icon_uri,
                        hands.most_played_hand(r),
                        best=r.name == best_run,
                        replay=r.replay,
                    )
                )
                order = e.sort_values(["ante", "return"], kind="stable")["episode"]
                b1, b2 = st.columns(2)
                b1.button(
                    "Best episode",
                    icon=":material/trophy:",
                    on_click=show_episode,
                    args=(r.name, int(order.iloc[-1])),
                    key=f"sheet_best_{r.name}",
                    width="stretch",
                )
                b2.button(
                    "Worst episode",
                    icon=":material/heart_broken:",
                    on_click=show_episode,
                    args=(r.name, int(order.iloc[0])),
                    key=f"sheet_worst_{r.name}",
                    width="stretch",
                )


IQM_HELP = (
    "**Interquartile mean (IQM)**: the mean of the middle 50% of games, dropping the best "
    "and worst quarter, as Agarwal et al. (2021) recommend for RL results: steadier than the "
    "mean, less coarse than the median."
)


def iqm_table(by_group: bool) -> None:
    """IQM of ante and return with stratified bootstrap 95% CIs (``stats.iqm_summary``)."""
    iq = stats.iqm_summary(runs, by_group)
    if not len(iq):
        return
    shared = (iq["method"] == stats.SHARED_GAMES).any()
    st.caption(
        IQM_HELP
        + " 95% CI by stratified bootstrap: "
        + (
            "where a group's runs played the same games (test or eval episodes), training "
            "seeds are "
            "redrawn within each game, as rliable does, so the interval reflects seed-to-seed "
            "variation; otherwise episodes are redrawn within each run."
            if by_group and shared
            else "episodes are redrawn within each run."
        )
    )
    num = st.column_config.NumberColumn(format="%.3f")
    table(iq, column_config={names.column(c): num for c in ("mean", "iqm", "iqm_low", "iqm_high")})


POI_HELP = (
    "**Probability of improvement** (Agarwal et al. 2021): the chance that a game from this "
    "{unit} reaches a higher ante than a game from the baseline, ties counting half. 0.5 means "
    "no difference; a CI that excludes 0.5 is a reliable difference in either direction. Unlike "
    "a difference in means it says how often one agent gets further, not by how much. "
)


def improvement_table(by_group: bool, baseline: str) -> None:
    """``stats.improvement`` of each run / group over ``baseline``, with its caption."""
    poi = stats.improvement(runs, baseline, by_group)
    if not len(poi):
        return
    how = {
        "paired games": "paired: compared game by game on the games both played, games "
        "redrawn for the CI",
        stats.SHARED_GAMES: "on the games every run played, per game over every pair of "
        "seeds and averaged over games; CI by redrawing seeds within each game (rliable)",
        stats.WITHIN_RUNS: "unpaired (no shared games); CI by redrawing episodes within each run",
    }
    used = "; ".join(f"*{m}*: {how[m]}" for m in poi["method"].unique())
    st.caption(POI_HELP.format(unit="group" if by_group else "run") + used + ". 95% CI.")
    num = st.column_config.NumberColumn(format="%.3f")
    table(poi, column_config={names.column(c): num for c in ("p_improvement", "p_low", "p_high")})


def past_eight_panel(by_group: bool) -> None:
    """Endless runs (log_version 17): how far the won games went past ante 8, as
    balatro-shop-rl's scripts/endless_report.py."""
    endless_runs = [r for r in runs if endless.is_endless(r)]
    if not endless_runs:
        return
    sets = stats.groups(endless_runs) if by_group else {r.name: [r] for r in endless_runs}
    rows, counts = [], {}
    for name, members in sets.items():
        res = endless.past_eight(members)
        if res is None:
            continue
        rows.append(
            {
                "group" if by_group else "run": name,
                "games": res.games,
                "won": res.won,
                "mean highest ante of a win": res.mean_win_ante,
                "lowest seed": res.seed_low,
                "highest seed": res.seed_high,
            }
        )
        counts[name] = res.counts
    st.markdown("**Endless: past ante 8**")
    if not rows:
        st.caption("No won games among the endless runs shown.")
        return
    table = pd.DataFrame(rows)
    if not by_group:
        table = table.drop(columns=["lowest seed", "highest seed"])
    st.dataframe(table, hide_index=True, key="past_eight")
    st.caption(
        "Endless runs play a won game on until it is lost, so a win's highest ante can pass "
        "8. Mean highest ante of a win: each run's mean over its won games, then the mean "
        "over runs (seeds), with the range across them. Counts of won games by highest ante:"
    )
    st.dataframe(
        pd.DataFrame(counts).fillna(0).astype(int).rename_axis("highest ante").reset_index(),
        hide_index=True,
        key="past_eight_counts",
    )


def profile_chart(by_group: bool) -> None:
    """``stats.performance_profile`` of the selection as a chart."""
    prof = stats.performance_profile(runs, by_group)
    if not len(prof):
        return
    fig = plots.performance_profile(prof, names)
    if by_group:
        fig.update_layout(legend_title_text="Group")
    chart(
        fig,
        "performance profile",
        "Performance profile (Agarwal et al. 2021): for each ante k, the share of games that "
        "reached ante k or further, one line per "
        + ("group" if by_group else "run")
        + ". It starts at 100% and falls where runs end, so it shows where along the run "
        "each agent breaks down, which a single mean can't. A line above another at every k "
        "dominates it. Band: 95% stratified bootstrap CI ("
        + (
            "training seeds redrawn within each shared game where the group's runs played the "
            "same games, otherwise episodes within each run"
            if by_group
            else "episodes redrawn within the run"
        )
        + ").",
    )


def power_section(by_group: bool, baseline: str | None) -> None:
    """How small a difference in mean ante this selection can detect, and what it
    would take to detect a chosen one (``stats.power_runs`` / ``power_groups``)."""
    st.subheader("Can these runs tell agents apart?")
    target = st.number_input(
        "Difference in mean ante to detect",
        min_value=0.01,
        value=0.25,
        step=0.05,
        key="power_target",
        help="The smallest difference you'd care about between two agents.",
    )
    shown = "two-sided test, α = 0.05, 80% power (exact noncentral t)"
    if by_group:
        pw = stats.power_groups(runs, target)
        if not need(pw, "group with two or more runs to estimate seed-to-seed variation"):
            return
        st.caption(
            "Comparing two groups on the mean of their runs' mean ante, as the comparison above "
            "does. Ante varies from game to game (SD within a run) and from seed to seed (SD of "
            "the runs' true means, after removing game noise). **Detectable now**: the "
            "smallest difference found 80% of the time with this many seeds and games per seed. "
            "**Seeds needed**: per group, for your difference with this many games per seed; "
            "with unlimited games only seed-to-seed variation is left. Treats the groups as "
            f"independent (no pairing on shared test or eval games), so it's conservative. "
            f"{shown}."
        )
        worst = pw.loc[pw["mde"].idxmax()]
        cols = st.columns(3)
        cols[0].metric(
            "Detectable now",
            f"{worst['mde']:.2f} ante",
            help=f"Worst group: {worst['group']}",
        )
        needed = pw["seeds_needed"].dropna()
        cols[1].metric(
            f"Seeds needed for {target:g}",
            f"{int(needed.max())}" if len(needed) else f"> {stats.MAX_N:,}",
            help=f"With {int(worst['games_per_seed'])} games per seed, the largest over the groups",
        )
        floor = pw["seeds_needed_any_games"].dropna()
        cols[2].metric(
            "With unlimited games",
            f"{int(floor.max())} seeds" if len(floor) == len(pw) else "unknown",
            help="Seeds needed if every seed played infinitely many games. Unknown when a "
            "group's seed-to-seed variation is estimated as 0: too small to see with these games.",
        )
        if worst["mde"] > target:
            st.warning(
                f"With {int(worst['seeds'])} seeds and {int(worst['games_per_seed'])} games per "
                f"seed, the smallest difference these groups can reliably detect is "
                f"{worst['mde']:.2f} ante, more than the {target:g} you're after: a "
                "non-significant difference here says little."
            )
        num = st.column_config.NumberColumn(format="%.3f")
        table(pw, column_config={names.column(c): num for c in ("sd_game", "sd_seed", "mde")})
        return
    if baseline is None:
        st.info("Select two or more runs to see what they can tell apart.")
        return
    pw = stats.power_runs(runs, baseline, target)
    if not need(pw, "runs to compare with the baseline"):
        return
    st.caption(
        f"Each run against {baseline} on mean ante. Paired on games both played (same seed, "
        "deck and stake), which removes seed luck, otherwise two-sample. **Detectable now**: "
        "the smallest difference found 80% of the time with this many games. **Games needed**: "
        f"per run, for your difference. {shown}."
    )
    worst = pw.loc[pw["mde"].idxmax()]
    cols = st.columns(2)
    cols[0].metric("Detectable now", f"{worst['mde']:.2f} ante", help=f"Worst run: {worst['run']}")
    needed = pw["games_needed"].dropna()
    cols[1].metric(
        f"Games needed for {target:g}",
        f"{int(needed.max()):,}" if len(needed) == len(pw) else f"> {stats.MAX_N:,}",
        help="The largest over the runs",
    )
    if worst["mde"] > target:
        st.warning(
            f"With {int(worst['games'])} games the smallest difference these runs can reliably "
            f"detect is {worst['mde']:.2f} ante, more than the {target:g} you're after: a "
            "non-significant difference here says little."
        )
    num = st.column_config.NumberColumn(format="%.3f")
    table(pw, column_config={names.column(c): num for c in ("sd", "mde")})


@contextlib.contextmanager
def table_budget():
    """Show a warning, not a traceback, when a tab needs more logs than fit in the
    memory set aside for runs (``make_room_for_table``)."""
    try:
        yield
    except TableBudgetError as e:
        st.warning(
            f"Not enough memory for this view: reading {e.table} for {e.run} (about "
            f"{e.needed / 1e9:,.1f} GB) would go past the {e.budget / 1e9:,.1f} GB set aside "
            f"for runs, with the logs of the other runs shown. Pick fewer runs, give the "
            f"dashboard more memory (Slurm `--mem`), or set `{memory.ENV}` (GB).",
            icon=":material/memory:",
        )


STUDY_GROUP_COLUMNS = {
    "group": "Group",
    "seeds": "Seeds",
    "games": "Games",
    "win_pct": "Win %",
    "win_pct_sd": "Win % SD (seeds)",
    "final_round": "Final round",
    "ante": "Ante",
    "ante_iqm": "IQM ante",
    "ante_iqm_low": "IQM low",
    "ante_iqm_high": "IQM high",
}
STUDY_PAIR_COLUMNS = {
    "a": "A",
    "b": "B",
    "pairs": "Seed pairs",
    "ante_diff": "Δ ante (A - B)",
    "ante_low": "Δ ante low",
    "ante_high": "Δ ante high",
    "p": "p",
    "a_better": "Seeds A ahead",
    "win_diff": "Δ win (pts)",
    "win_low": "Δ win low",
    "win_high": "Δ win high",
    "p_improvement": "P(A > B)",
    "p_low": "P low",
    "p_high": "P high",
    "unpaired_seeds": "Seeds not paired",
}


def study_tables(study: str, spec: dict, runs: list[Run]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The study's group and pair tables over ``runs``, memoised on the first run
    (filtered runs are kept between reruns, so this is computed once per filter)."""
    key = f"study:{study}:" + ",".join(r.name for r in runs)
    return stats.memo_value(
        runs[0],
        key,
        lambda: (
            studies.group_table(runs, spec.get("groups", [])),
            studies.pair_table(runs, spec.get("pairs", [])),
        ),
    )


def show_study_tables(groups: pd.DataFrame, pairs: pd.DataFrame, key: str) -> None:
    f2, f3 = (
        st.column_config.NumberColumn(format="%.2f"),
        st.column_config.NumberColumn(format="%.3f"),
    )
    if len(groups):
        st.dataframe(
            groups.rename(columns=STUDY_GROUP_COLUMNS),
            hide_index=True,
            key=f"{key}_groups",
            column_config={
                c: (f3 if c.startswith("IQM") or c == "Ante" else f2)
                for c in STUDY_GROUP_COLUMNS.values()
                if c not in ("Group", "Seeds", "Games")
            },
        )
    if len(pairs):
        shown = (
            pairs
            if pairs["unpaired_seeds"].astype(bool).any()
            else pairs.drop(columns="unpaired_seeds")
        )
        st.dataframe(
            shown.rename(columns=STUDY_PAIR_COLUMNS),
            hide_index=True,
            key=f"{key}_pairs",
            column_config={
                c: (st.column_config.NumberColumn(format="%.3g") if c == "p" else f3)
                for c in STUDY_PAIR_COLUMNS.values()
                if c not in ("A", "B", "Seed pairs", "Seeds A ahead", "Seeds not paired")
            },
        )


def study_section(manifest: studies.Manifest, study: str, runs: list[Run]) -> None:
    """The chosen study's results on the episodes shown: one row per group, then
    each pair A - B matched by training seed; by ``split_by`` too if it has one."""
    spec = manifest.studies[study]
    in_study = set(spec.get("groups", []))
    members = [r for r in runs if stats.group_of(r) in in_study]
    st.subheader(spec.get("title", study))
    if not members:
        st.info("None of this study's runs are picked (or have these episodes).")
        return
    st.caption(
        f"On {episodes_shown()}. Groups: mean over seeds (win % ± SD across seeds), and the "
        "IQM of ante over seeds × games with a stratified-bootstrap 95% CI. Pairs: A - B "
        "per training seed (same seed, same training games), Student-t 95% CI and p over "
        "the seed pairs; P(A > B): the chance a game of A gets further than one of B, "
        "rliable's, with a bootstrap 95% CI." + (f" {spec['note']}" if spec.get("note") else "")
    )
    if not test_mode:
        st.warning(
            "Not the test games: set **Episodes** to *Test* for the results to report.",
            icon=":material/info:",
        )
    kinds = {endless.is_endless(r) for r in members}
    memo_key = study  # the tables' memo key
    if True in kinds:
        mixed = len(kinds) > 1
        if st.toggle(
            "Cap ante at 8",
            value=mixed,
            key=f"study_cap_{study}",
            help="Endless runs play a won game on past ante 8. Capped, every run's ante stops "
            "at 8, so endless and normal runs compare like for like (as endless_report.py's "
            "first table); on by default when the study mixes them.",
        ):
            members = [endless.capped(r) for r in members]
            memo_key = f"{study}:cap"
    groups, pairs = study_tables(memo_key, spec, members)
    show_study_tables(groups, pairs, f"study_{study}")
    tex = [
        latex.study_groups(groups, latex.escape(spec.get("title", study)), f"{study}-groups"),
        latex.study_pairs(
            pairs, latex.escape(spec.get("title", study)) + ", paired by seed", f"{study}-pairs"
        ),
    ]
    if col := spec.get("split_by"):
        values = sorted({v for r in members if col in r.episodes for v in r.episodes[col]})
        for v in values:
            sub = [studies.where(r, col, v) for r in members]
            sub = [r for r in sub if len(r.episodes)]
            if not sub:
                continue
            with st.expander(
                f"{names.column(col)}: {names.of(col)(v) if col in ('deck', 'stake') else v}"
            ):
                g, p = study_tables(f"{memo_key}:{col}={v}", spec, sub)
                show_study_tables(g, p, f"study_{study}_{v}")
                tex.append(
                    latex.study_pairs(
                        p,
                        latex.escape(f"{spec.get('title', study)}, {col} {v}"),
                        f"{study}-pairs-{v}",
                    )
                )
    c1, c2 = st.columns(2)
    c1.download_button(
        "Study tables (CSV)",
        data=groups.to_csv(index=False) + "\n" + pairs.to_csv(index=False),
        file_name=f"study_{study}.csv",
        mime="text/csv",
        key=f"dl_study_csv_{study}",
        on_click="ignore",
    )
    c2.download_button(
        "Study tables (LaTeX)",
        data="\n".join(tex),
        file_name=f"study_{study}.tex",
        mime="text/x-tex",
        key=f"dl_study_tex_{study}",
        on_click="ignore",
    )


# --- Overview -------------------------------------------------------------
with tabs[0], table_budget():
    if tabs[0].open:
        run_sheets()
        if test_mode:
            st.info(
                "Showing **test games**: each run's final policy on the same fixed games, "
                "never used in training. These are the results to report; the paired "
                "comparison below matches runs game by game."
            )
        elif eval_mode:
            st.info(
                "Showing **eval episodes** (greedy, fixed seeds), eval round "
                f"{eval_rounds[0] if eval_rounds else 'all'}. Runs played the same games, "
                "so the paired comparison below matches them game by game. For learning "
                "curves: with few games per round, rank runs on **Test** instead."
            )
        if study is not None:
            study_section(manifest, study, runs)
        group_map = stats.groups(runs)
        by_group = (
            len(group_map) < len(runs)
            and st.radio(
                "Compare",
                ["Runs", "Groups"],
                horizontal=True,
                key="compare_by",
                help="Groups: runs sharing --group (one config, several training seeds) "
                "are averaged, with CIs across seeds.",
            )
            == "Groups"
        )
        if by_group:
            num = st.column_config.NumberColumn(format="%.3f")
            pval = st.column_config.NumberColumn(format="%.2g")
            st.caption(
                "Each run's mean, then the mean over a group's runs with a Student-t 95% CI "
                "across runs: the interval shows how much results vary between training "
                "seeds, not how many episodes were played."
            )
            table(
                stats.group_summary(runs),
                column_config={
                    names.column(c): num for c in ("mean", "ci_low", "ci_high", "std_across_runs")
                },
            )
            with st.expander("Runs in each group"):
                table(
                    stats.run_means(runs),
                    column_config={names.column(c): num for c in stats.OUTCOMES.values()},
                )
            iqm_table(by_group=True)
            if len(group_map) > 1:
                baseline = st.selectbox(
                    "Compare against baseline group", list(group_map), key="baseline_group"
                )
                st.caption(
                    "Difference in the mean of the runs' means vs the baseline group, with "
                    "Welch's t-test and its 95% CI (each run is one observation)."
                )
                gcomp = stats.compare_groups(runs, baseline)
                table(
                    gcomp,
                    column_config={
                        names.column(c): pval if c.endswith("_p") else num
                        for c in gcomp.columns
                        if c.endswith(("_diff", "_low", "_high", "_p"))
                    },
                )
                improvement_table(by_group=True, baseline=baseline)
            power_section(by_group=True, baseline=None)
        else:
            summary = metrics.summary(runs)
            table(
                summary,
                {"decks": "items", "stakes": "stakes"},
                column_config={
                    names.column("mean ante"): st.column_config.NumberColumn(format="%.2f"),
                    names.column("win rate"): st.column_config.NumberColumn(format="percent"),
                    names.column("mean return"): st.column_config.NumberColumn(format="%.2f"),
                    names.column("shop steps / ep"): st.column_config.NumberColumn(format="%.1f"),
                    names.column("s / ep"): st.column_config.NumberColumn(format="%.3f"),
                },
            )
            ci = stats.summary_ci(runs)
            if len(ci):
                st.caption(
                    "95% confidence intervals of the mean: bootstrap for ante and return, "
                    "Wilson for win rate (stays sensible with no wins yet)."
                )
                num = st.column_config.NumberColumn(format="%.3f")
                table(
                    ci,
                    column_config={
                        names.column(c): num for c in ("mean", "ci_low", "ci_high", "std")
                    },
                )
            iqm_table(by_group=False)
            if len(runs) > 1:
                baseline = st.selectbox(
                    "Compare against baseline", [r.name for r in runs], key="baseline_run"
                )
                comparison = stats.compare(runs, baseline)
                st.caption(
                    "Ante reached vs the baseline. **Unpaired**: difference in mean ante with a "
                    "bootstrap 95% CI, Mann-Whitney U and its rank-biserial effect size "
                    "(+1: this run always gets further). Win rate by Fisher's exact test. "
                    "**Paired**: episodes played on the same seed (and deck/stake) in both runs, "
                    "mean per-seed difference and Wilcoxon signed-rank. Pairing removes "
                    "seed-to-seed luck, so prefer it when the runs share seeds."
                )
                num = st.column_config.NumberColumn(format="%.3f")
                pval = st.column_config.NumberColumn(format="%.2g")
                table(
                    comparison,
                    column_config={
                        names.column(c): num
                        for c in (
                            "mean_ante_diff",
                            "diff_ci_low",
                            "diff_ci_high",
                            "rank_biserial",
                            "win_rate_diff",
                            "paired_mean_diff",
                        )
                    }
                    | {names.column(c): pval for c in ("mann_whitney_p", "fisher_p", "wilcoxon_p")},
                )
                improvement_table(by_group=False, baseline=baseline)
            power_section(by_group=False, baseline=baseline if len(runs) > 1 else None)
        if eval_runs:
            st.subheader("Evaluation on fixed seeds")
            ecurve = stats.group_eval_curve(eval_runs) if by_group else stats.eval_curve(eval_runs)
            if need(ecurve, "eval episodes for this selection"):
                emetric = st.radio(
                    "Eval metric", ["mean ante", "win rate"], horizontal=True, key="eval_metric"
                )
                fig = plots.eval_curve(ecurve, emetric, names)
                if by_group:
                    fig.update_layout(legend_title_text="Group")
                chart(
                    fig,
                    f"eval {emetric}",
                    f"Eval {emetric} against the number of training episodes so far: each "
                    "point is one eval round of greedy episodes (no exploration) on the same "
                    "fixed seeds, so rounds are directly comparable and a learning agent's line "
                    "should rise. Band: 95% CI ("
                    + (
                        "each group's runs averaged, Student-t CI across runs"
                        if by_group
                        else "bootstrap for ante, Wilson for win rate"
                    )
                    + "). For the learning curve only: results come from the test games "
                    "(**Episodes** → *Test* in the sidebar).",
                )
        if split is None:
            if by_group:
                curve = stats.group_learning_curve(runs, window)
                band_help = "Student-t 95% CI across the group's runs"
            else:
                curve = metrics.learning_curve(runs, window)
                band_help = "mean ± 1.96 SE over the window"
            if need(curve, "episodes for this selection"):
                c1, c2 = st.columns([3, 1])
                metric = c1.radio("Curve", list(metrics.CURVE_METRICS), horizontal=True)
                band = c2.checkbox("95% band", value=True, help=band_help)
                fig = plots.learning_curve(curve, metric, names, band)
                if by_group:
                    fig.update_layout(legend_title_text="Group")
                chart(
                    fig,
                    f"learning curve {metric}",
                    f"Training {metric} as a rolling average over the last {window} episodes "
                    "(the smoothing window is in the sidebar), one line per "
                    + ("group" if by_group else "run")
                    + f". Band: {band_help}. A flat line means no learning; a random agent "
                    "should stay flat.",
                )
        if any(len(r.episodes) for r in runs):
            chart(
                plots.ante_distribution(metrics.ante_distribution(runs), names),
                "ante reached",
                "Share of episodes that ended at each ante (the highest ante reached), one bar "
                "per run. Bars further right mean the agent gets further; winning means "
                "clearing ante 8 (endless runs play a win on past it).",
            )
            past_eight_panel(by_group)
            profile_chart(by_group)
        with st.expander("Run configs"):
            for r in runs:
                st.markdown(f"**{r.name}**")
                st.json(r.config or {"note": "no config.json"}, expanded=False)

# --- Decks ----------------------------------------------------------------
with tabs[1], table_budget():
    if tabs[1].open:
        deck_table = metrics.by_deck(runs)
        if need(deck_table, "episodes for this selection"):
            st.caption("Click a deck for its effect and results.")
            decks_played = sorted(deck_table["deck"].unique())
            modal_html(details.deck_grid_html(deck_infos(decks_played), names))
            chart(
                plots.by_deck(deck_table, names),
                "mean ante by deck",
                "Mean ante reached for each deck and stake played, one bar per run. Higher "
                "stakes add harder rules, so compare decks within a stake; hover for the "
                "episode count, best ante and win rate.",
            )
            table(
                deck_table[
                    ["run", "deck", "stake", "episodes", "mean_ante", "max_ante", "win_rate"]
                ],
                column_config={
                    names.column("mean_ante"): st.column_config.NumberColumn(format="%.2f"),
                    names.column("win_rate"): st.column_config.NumberColumn(format="percent"),
                },
            )

# --- Actions --------------------------------------------------------------
with tabs[2], table_budget():
    if tabs[2].open:
        summary = actions.visit_summary(runs)
        if len(summary):
            top_row = summary.iloc[0]
            st.caption(
                f"Shop visits in **{top_row['run']}**"
                + (" (every selected run is in the table below)" if len(summary) > 1 else "")
            )
            cols = st.columns(4)
            cols[0].metric("Decisions per visit", f"{top_row['decisions_per_visit']:.2f}")
            cols[1].metric("Visits leaving at once", f"{top_row['leave_at_once']:.1%}")
            cols[2].metric("Visits with a buy", f"{top_row['with_buy']:.1%}")
            cols[3].metric(
                "Most common first action",
                names.action(top_row["top_first_action"]),
                f"{top_row['top_first_share']:.0%} of visits",
                delta_color="off",
                delta_arrow="off",
            )
            if len(summary) > 1:
                pct = st.column_config.NumberColumn(format="percent")
                table(
                    summary,
                    {"top_first_action": "action"},
                    column_config={
                        names.column(c): pct
                        for c in (
                            "leave_at_once",
                            "with_buy",
                            "with_sell",
                            "with_reroll",
                            "top_first_share",
                        )
                    }
                    | {
                        names.column("decisions_per_visit"): st.column_config.NumberColumn(
                            format="%.2f"
                        )
                    },
                )

        sub = st.tabs(
            [
                "Overview",
                "Context",
                "Sequences",
                "Over training",
                "Availability",
                "Actions vs outcome",
            ],
            key="actions_tab",
            on_change="rerun",
        )
        with sub[0]:
            if sub[0].open:
                fam = actions.family_share(runs)
                if need(fam, "shop actions"):
                    chart(
                        plots.family_share(fam, names),
                        "action family share",
                        "Share of all shop decisions by kind (buy card, buy pack, sell joker, "
                        "reroll, leave shop, ...), one bar per run. Each bar sums to 100%.",
                    )
                freq = metrics.action_frequency(runs)
                if need(freq, "shop actions"):
                    chart(
                        plots.action_frequency(freq, names),
                        "action share",
                        "Share of shop decisions that chose each of the 17 shop actions (slot "
                        "numbers count from 0, left to right), one bar per run.",
                    )
                visits = actions.decisions_per_visit(runs)
                if need(visits, "shop visits", no_steps):
                    chart(
                        plots.decisions_per_visit(visits, names),
                        "decisions per visit",
                        "How many decisions the agent made in one shop visit, including the "
                        "final leave-shop. 1 means it left at once without buying anything.",
                    )

        with sub[1]:
            if sub[1].open:
                left, right = st.columns(2)
                run_name = left.selectbox("Run", run_names, key="actions_run")
                split = right.selectbox("Split by", list(actions.SPLITS), key="actions_split")
                heat = actions.share_by(by_name[run_name], split)
                if need(heat, "per-step action data", by_name[run_name].steps is None):
                    chart(
                        plots.action_heatmap(
                            heat, f"Action share by {split.lower()}: {run_name}", split, names
                        ),
                        f"actions by {split.lower()} {run_name}",
                        f"Which action the agent chose given the {split.lower()}"
                        + (" (money before the decision)" if split == "Money" else "")
                        + ". Each column sums to 100%; darker cells are more common choices "
                        "in that situation.",
                    )

        with sub[2]:
            if sub[2].open:
                run_name = st.selectbox("Run", run_names, key="actions_seq_run")
                run = by_name[run_name]
                matrix = actions.transition_matrix(run)
                if need(matrix, "per-step action data", run.steps is None):
                    chart(
                        plots.transition_matrix(matrix, run_name, names),
                        f"transitions {run_name}",
                        "For each decision (row), how often each decision came straight after "
                        "it in the same shop visit. Rows sum to 100%; shows habits such as "
                        "reroll → reroll or buy → leave.",
                    )
                    pos = actions.share_by(run, "Position in visit")
                    chart(
                        plots.action_heatmap(
                            pos,
                            f"Action share by position in the visit: {run_name}",
                            "decision number in the visit",
                            names,
                        ),
                        f"actions by position {run_name}",
                        "Which action was chosen at the 1st, 2nd, 3rd ... decision of a shop "
                        "visit. Each column sums to 100%; shows what the agent does first and "
                        "how visits usually end.",
                    )
                    seqs = actions.top_sequences(run)
                    st.subheader("Most common whole visits")
                    seqs["sequence"] = [
                        " → ".join(names.action(a) for a in seq.split(" → "))
                        for seq in seqs["sequence"]
                    ]
                    table(
                        seqs,
                        column_config={
                            names.column("share"): st.column_config.NumberColumn(format="percent")
                        },
                    )

        with sub[3]:
            if sub[3].open:
                # Eval episodes are numbered round by round, so one block per eval round
                # lines the blocks up with the rounds
                n_rounds = max(r.episodes["eval_round"].nunique() for r in runs) if eval_mode else 0
                bins = st.slider(
                    "Episode bins",
                    2,
                    50,
                    n_rounds if n_rounds > 1 else 10,
                    key="actions_bins_eval" if eval_mode else "actions_bins",
                    help="Splits each run's episodes into this many equal blocks: 1000 "
                    "episodes and 10 bins gives blocks of 100. More bins show finer changes, "
                    "but each block has fewer decisions, so it's noisier.",
                )
                st.caption(
                    "Does the shop agent's behaviour change as it trains? Each column is a "
                    "block of episodes; the coloured bands split that block's shop decisions "
                    "by kind and add up to 100%."
                    + (
                        " These are **eval episodes**: with Eval round *All* under Filter and "
                        "one bin per eval round, each column is the greedy policy at that "
                        "point in training (round 0 is before any training)."
                        if eval_mode
                        else ""
                    )
                )
                over = actions.family_over_training(runs, bins)
                if need(over, "per-step action data", no_steps):
                    chart(
                        plots.family_over_training(over, names),
                        "action families over training",
                        "Share of each kind of shop decision (buy a card, pack or voucher, "
                        "sell a joker or consumable, use a consumable, reroll, leave) in each "
                        "block of episodes, one panel per run; the x axis is each block's first "
                        "episode. The random, heuristic and smart shops don't learn, so their "
                        "mix should stay roughly flat: any slow drift comes from the games "
                        "(better play reaches later antes with more money to spend), not the "
                        "agent. Q-learning starts fully random (epsilon 1.0, falling evenly to "
                        "0.05 over the first 80% of training episodes), so its early blocks look "
                        "like the random shop and the later ones show the habits it learned.",
                    )

        with sub[4]:
            if sub[4].open:
                avail = actions.availability(runs)
                if need_new(avail, "legal-action data", version=3):
                    missing = sorted(set(run_names) - set(avail["run"]))
                    if missing:
                        st.caption("Not shown (logged before log_version 3): " + ", ".join(missing))
                    chart(
                        plots.availability(avail, names),
                        "action availability",
                        "For each action: how often it was legal (available and affordable), "
                        "how often it was taken when legal, and the uniform-random rate, the "
                        "chance of taking it if the agent chose evenly among the legal actions "
                        "each time it was legal.",
                    )
                    chart(
                        plots.preference(avail[avail["legal_count"] >= MIN_LEGAL], names),
                        "action preference vs uniform",
                        "Taken rate ÷ uniform-random rate: 1 is random-like, above 1 the agent "
                        "favours the action, below 1 it avoids it. Error bars are 95% "
                        "intervals; q is the Benjamini-Hochberg adjusted p-value of a test "
                        "against uniform-random choice (q < 0.05: a real preference). Actions "
                        f"legal fewer than {MIN_LEGAL} times are left out (too few to tell).",
                    )
                    chart(
                        plots.legal_counts(actions.legal_counts(runs), names),
                        "legal actions",
                        "How many of the 17 shop actions were legal at each decision. Leaving "
                        "is always legal, so the minimum is 1; few legal actions means little "
                        "real choice.",
                    )

        with sub[5]:
            if sub[5].open:
                corr = actions.action_outcome_correlations(runs)
                if need(corr, "enough episodes with shop data", no_steps):
                    chart(
                        plots.action_outcome_correlations(corr, names),
                        "actions vs ante reached",
                        "Spearman ρ between how often each kind of decision is made per shop "
                        "visit and the ante reached (+1: more of it goes with deeper runs). "
                        "Per visit, because longer runs visit more shops. Correlation, not "
                        "cause. \\* = significant at q < 0.05 (p-values Benjamini-Hochberg "
                        "adjusted within each run); hover for p and q.",
                    )

# --- Economy --------------------------------------------------------------
with tabs[3], table_budget():
    if tabs[3].open:
        kpi = economy.kpis(runs)
        if len(kpi):
            top_row = kpi.iloc[0]
            st.caption(
                f"Key numbers for **{top_row['run']}**"
                + (" (every selected run is in the table below)" if len(kpi) > 1 else "")
            )
            cols = st.columns(7)
            for col, (label, field, fmt) in zip(
                cols,
                [
                    ("$ spent / episode", "spent_per_episode", "${:.2f}"),
                    ("$ from selling / ep", "sold_per_episode", "${:.2f}"),
                    ("$ on rerolls / ep", "reroll_per_episode", "${:.2f}"),
                    ("Mean $ entering shop", "mean_entry", "${:.2f}"),
                    ("Mean $ leaving shop", "mean_exit", "${:.2f}"),
                    ("Exits at interest cap", "exits_at_cap", "{:.1%}"),
                    ("Mean $ at end of run", "final_dollars", "${:.2f}"),
                ],
                strict=True,
            ):
                value = top_row[field]
                col.metric(label, "-" if pd.isna(value) else fmt.format(value))
            if len(kpi) > 1:
                money = st.column_config.NumberColumn(format="$%.2f")
                table(
                    kpi,
                    column_config={
                        names.column(c): money
                        for c in (
                            "spent_per_episode",
                            "sold_per_episode",
                            "reroll_per_episode",
                            "mean_entry",
                            "mean_exit",
                            "final_dollars",
                        )
                    }
                    | {
                        names.column("exits_at_cap"): st.column_config.NumberColumn(
                            format="percent"
                        )
                    },
                )

        visits = economy.shop_visits(runs)
        econ = economy.episode_economy(runs)
        sub = st.tabs(
            [
                "Money flow",
                "Spending",
                "Interest & saving",
                "Rerolls & prices",
                "Economy vs outcome",
                "Cash-out income",
                "Shop offers",
                "Shop purchases",
            ],
            key="econ_tab",
            on_change="rerun",
        )
        with sub[0]:
            if sub[0].open:
                if need(visits, "shop visits", no_steps):
                    curve = economy.money_by_round(runs)
                    chart(
                        plots.money_curve(curve[curve["visits"] >= MIN_VISITS], names),
                        "money curve",
                        "Median money entering and leaving the shop in each round, with the "
                        "middle 50% of visits shaded. The gap between the lines is what was "
                        f"spent. Rounds with fewer than {MIN_VISITS} shop visits are left off "
                        "(too few runs get that far to say much).",
                    )
                    chart(
                        plots.entry_exit_by_ante(visits, names),
                        "money entering vs leaving",
                        "Spread of money on entering and on leaving each shop visit, by ante "
                        "(box: middle 50% of visits, line: median, whiskers: the rest).",
                    )
                    money = metrics.money_by_ante(runs)
                    chart(
                        plots.money_by_ante(money, names),
                        "money before each decision",
                        "Spread of the money held just before every shop decision, by ante "
                        "(box: middle 50%, line: median).",
                    )
                if "final_dollars" in econ and econ["final_dollars"].notna().any():
                    chart(
                        plots.final_dollars(econ, names),
                        "money at end of run",
                        "Money held when each episode ended, as a share of episodes. Lots of "
                        "money left at the end was never turned into a stronger build.",
                    )

        with sub[1]:
            if sub[1].open:
                spend = economy.spend_by_category(runs)
                if need(spend, "purchases", no_steps):
                    chart(
                        plots.spend_share(spend, names),
                        "spending by category",
                        "Where the shop money went (jokers, packs, vouchers, rerolls, ...) as a "
                        "share of all shop spending, one bar per run.",
                    )
                    chart(
                        plots.spend_by_ante(economy.spend_by_category(runs, by_ante=True), names),
                        "spending mix by ante",
                        "The same split of shop spending, separately for each ante: shows "
                        "whether the agent's spending changes as the run goes on.",
                    )
                income = economy.income_by_source(runs)
                if need(income, "income"):
                    chart(
                        plots.income_sources(income, names),
                        "income by source",
                        "Mean money gained per episode from each source: selling and "
                        "money-making consumables in the shop, plus the cash-out parts (blind "
                        "reward, unused hands and discards, interest, jokers) for newer runs.",
                    )
                spend_ep = metrics.spend_per_episode(runs)
                if need(spend_ep, "spend data"):
                    chart(
                        plots.spend_per_episode(spend_ep, names),
                        "spend per episode",
                        "Total money spent in the shop over a whole episode, as a share of "
                        "episodes. Longer runs visit more shops, so they tend to spend more.",
                    )

        with sub[2]:
            if sub[2].open:
                if need(visits, "shop visits", no_steps):
                    chart(
                        plots.exit_brackets(visits, names),
                        "money left after shopping",
                        "Money left on leaving each shop, in $5 brackets, as a share of visits. "
                        "Interest is paid at cash-out at $1 per $5 held, up to the cap (dashed "
                        "line), so money above the cap earns nothing.",
                    )
                    chart(
                        plots.interest_by_ante(visits, names),
                        "estimated interest by ante",
                        "Mean interest the money left on leaving the shop would earn at the "
                        "next cash-out, by ante. It is estimated from the exit balance, so it "
                        "ignores money gained or spent during the round.",
                    )
                    parked = visits.groupby("run", as_index=False).agg(
                        visits=("exit", "size"),
                        exits_at_cap=("at_cap", "mean"),
                        mean_parked_above_cap=("parked_above_cap", "mean"),
                        mean_est_interest=("est_interest", "mean"),
                    )
                    table(
                        parked,
                        column_config={
                            names.column("exits_at_cap"): st.column_config.NumberColumn(
                                format="percent"
                            ),
                            names.column("mean_parked_above_cap"): st.column_config.NumberColumn(
                                format="$%.2f"
                            ),
                            names.column("mean_est_interest"): st.column_config.NumberColumn(
                                format="$%.2f"
                            ),
                        },
                    )
                eff = economy.interest_efficiency(runs)
                if need_new(eff, "exact interest"):
                    chart(
                        plots.interest_efficiency(eff, names),
                        "interest efficiency",
                        INTEREST_EFFICIENCY,
                    )

        with sub[3]:
            if sub[3].open:
                if need(visits, "shop visits", no_steps):
                    chart(
                        plots.rerolls_per_visit(visits, names),
                        "rerolls per visit",
                        "How many times the shop was rerolled in one visit, as a share of visits.",
                    )
                    chart(
                        plots.reroll_by_ante(economy.reroll_stats(runs), names),
                        "reroll spend by ante",
                        "Mean money spent on rerolls per shop visit, by ante. A reroll's price "
                        "goes up by $1 each time within a visit.",
                    )
                paid = economy.price_paid(runs)
                if need(paid, "purchases", no_steps):
                    chart(
                        plots.price_paid(paid, names),
                        "price paid",
                        "Spread of the price paid per purchase in each category (box: middle "
                        "50%, line: median).",
                    )
                    chart(
                        plots.joker_price_by_ante(paid, names),
                        "joker price by ante",
                        "Mean price paid per joker bought, by ante: whether the agent buys "
                        "pricier (usually rarer) jokers later in the run.",
                    )

        with sub[4]:
            if sub[4].open:
                corr = economy.outcome_correlations(econ)
                if need(corr, "enough episodes with shop data", no_steps):
                    chart(
                        plots.outcome_correlations(corr, names),
                        "economy vs ante reached",
                        "Spearman ρ between each economy habit and the ante reached (+1: more "
                        "of it goes with deeper runs). Totals grow with run length, so they "
                        "correlate almost by construction; compare the per-visit measures. "
                        "\\* = significant at q < 0.05 (p-values Benjamini-Hochberg "
                        "adjusted within each run); hover for p and q.",
                    )
                    choices = {
                        k: v
                        for k, v in economy.OUTCOME_METRICS.items()
                        if k in econ and econ[k].notna().any()
                    }
                    metric = st.selectbox(
                        "Habit", list(choices), format_func=choices.get, key="econ_outcome_metric"
                    )
                    chart(
                        plots.economy_by_outcome(econ, metric, choices[metric], names),
                        f"{choices[metric]} by ante reached",
                        f"Spread of {choices[metric].lower()} per episode, grouped by the ante "
                        "the episode reached (box: middle 50%, line: median). Pick the habit "
                        "above.",
                    )

        with sub[5]:
            if sub[5].open:
                breakdown = economy.income_breakdown(runs)
                if need_new(breakdown, "cash-out breakdown"):
                    chart(
                        plots.income_breakdown(breakdown, names),
                        "cash-out income",
                        "Mean cash-out money per beaten blind, split into its parts (blind "
                        "reward, unused hands, unused discards, interest, jokers, tags, minus "
                        "rent), by ante. These are the exact amounts the game paid.",
                    )
                    eff = economy.interest_efficiency(runs)
                    if need_new(eff, "exact interest"):
                        chart(
                            plots.interest_efficiency(eff, names),
                            "interest efficiency (cash-out)",
                            INTEREST_EFFICIENCY,
                        )

        with sub[6]:
            if sub[6].open:
                conversion = economy.offer_conversion(runs)
                if need_new(conversion, "shop offers"):
                    chart(
                        plots.offer_conversion(conversion, names),
                        "offer conversion",
                        "For each kind of shop item: the share of offers the agent could afford, "
                        "and the share it bought, of everything offered and of what it could "
                        "afford. A low rate when affordable means the agent passes them up.",
                    )
                    chart(
                        plots.offered_vs_paid(conversion, economy.price_paid(runs), names),
                        "price offered vs paid",
                        "Mean price of the items the shop offered next to the mean price of "
                        "those the agent bought, per category. Paying less than offered means "
                        "it picks the cheaper items.",
                    )
                    table(
                        conversion.round(3),
                        column_config={
                            names.column(c): st.column_config.NumberColumn(format="percent")
                            for c in ("affordable_share", "buy_rate", "buy_rate_when_affordable")
                        },
                    )
                    skipped = economy.skipped_jokers(runs)
                    st.subheader("Jokers most often affordable but not bought")
                    if need(skipped, "affordable jokers"):
                        joker_grid(
                            [
                                (s.key, f"skipped <b>{s.skipped}</b> of {s.affordable}")
                                for s in skipped.itertuples()
                            ],
                            "econ-jk",
                        )

        with sub[7]:
            if sub[7].open:
                shop_purchases()

# --- Jokers ---------------------------------------------------------------
with tabs[4], table_budget():
    if tabs[4].open:
        top = st.slider("Top N jokers", 5, 40, TOP_JOKERS_DEFAULT)
        run_name = st.selectbox("Run", run_names, key="jokers_run")
        run = by_name[run_name]
        bought_one = metrics.most_bought([run], top)
        st.subheader(f"Most-bought jokers: {run_name}")
        if need(bought_one, "joker purchases", run.steps is None):
            joker_grid(
                [(b.joker, f"bought <b>{b.bought}&times;</b>") for b in bought_one.itertuples()],
                "top-jk",
            )
        bought = metrics.most_bought(runs, top)
        if len(runs) > 1 and need(bought, "joker purchases", no_steps):
            chart(
                plots.most_bought(bought, names),
                "most bought jokers",
                "How many times each joker was bought across all episodes, for the most-bought "
                "jokers (count set by the slider), one bar per run.",
            )
        first = metrics.first_bought_ante(runs)
        if len(first):
            chart(
                plots.first_bought_ante(first, names),
                "first bought ante",
                "The ante in which each joker was first bought in an episode, as a share of "
                "first purchases: how early the agent builds its joker line-up.",
            )
        co = metrics.joker_cooccurrence(run, top)
        if need(co, "end-of-episode joker builds"):
            chart(
                plots.cooccurrence(co, run_name, names),
                f"joker cooccurrence {run_name}",
                "How many episodes ended holding both jokers in the row and the column; the "
                "diagonal is how often each joker was held at the end at all. Shows which "
                "jokers end up together.",
            )
        swaps = metrics.sell_then_buy(run)
        st.subheader("Sold → next joker bought")
        if need(swaps, "sell-then-buy pairs", run.steps is None):
            table(
                swaps.assign(
                    sold_icon=swaps["sold"].map(sprites.data_uri) if sprites else None,
                    bought_icon=swaps["bought"].map(sprites.data_uri) if sprites else None,
                )[["sold_icon", "sold", "bought_icon", "bought", "count"]],
                {"sold": "item", "bought": "item"},
                column_config={
                    names.column("sold_icon"): st.column_config.ImageColumn("", width="small"),
                    names.column("bought_icon"): st.column_config.ImageColumn("", width="small"),
                },
                row_height=60 if sprites else None,
            )

# --- Joker collection -----------------------------------------------------
with tabs[5], table_budget():
    if tabs[5].open:
        game_text = cached_game_text(str(archive), pkg_mtime) if archive else None
        cards = joker_cards()
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        query = c1.text_input("Search", placeholder="name, key or effect text", key="jk_query")
        rarities = c2.multiselect(
            "Rarity",
            list(RARITY),
            default=list(RARITY),
            format_func=lambda r: RARITY[r][0],
            key="jk_rarity",
        )
        sort = c3.selectbox("Sort by", list(collection.SORTS), key="jk_sort")
        only_used = c4.toggle("Only used in runs", key="jk_used")
        shown = collection.select(cards, query, rarities, only_used, sort)
        note = "" if game_text else " · art and effect text need the game (see Settings)"
        st.caption(
            f"Showing {len(shown)} of {len(cards)} jokers · stats from the selected runs"
            f" · click a joker for details{note}"
        )
        if shown:
            modal_html(collection.grid_html(shown, names))
        else:
            st.info("No jokers match.")

# --- Rounds ---------------------------------------------------------------
with tabs[6], table_budget():
    if tabs[6].open:
        st.subheader("Boss blinds")
        bosses = metrics.boss_stats(runs)
        if need(bosses, "boss blinds faced", no_rounds):
            st.caption("Most faced first · click a boss blind for its effect and results.")
            text = cached_game_text(str(archive), pkg_mtime) if archive else None
            modal_html(details.blind_grid_html(details.blind_infos(bosses, text, icon_uri), names))
        ratio = metrics.score_ratio(runs)
        if need(ratio, "round data", no_rounds):
            chart(
                plots.score_ratio(ratio, names),
                "score vs target",
                "Chips scored ÷ the blind's target for every blind played, by ante and blind "
                "type. At or above the dashed line (1) the blind was beaten; the further "
                "above, the more comfortably.",
            )
        dead = metrics.deaths(runs)
        if need(dead, "lost rounds", no_rounds):
            chart(
                plots.deaths(dead, names),
                "blind that ended the run",
                "Number of episodes lost on each blind, coloured by ante: where runs fail, "
                "and which boss blinds are hardest for the agent.",
            )

# --- Hands ----------------------------------------------------------------
NO_HAND_LOG_TRAINING = (
    "Hands are only recorded for test and evaluation episodes. Switch **Episodes** to "
    "**Test** or **Evaluation** in the sidebar to see them."
)
NO_HAND_LOG_EVAL = (
    "These episodes have no hand log. Runs from balatro-shop-rl with log_version 7 "
    "record every play and discard in eval/hands.jsonl (and test/hands.jsonl from "
    "log_version 14); train again with --eval-every."
)


def hand_log_missing(run_list: list[Run]) -> bool:
    """Say why there are no hands to show, if there aren't."""
    if any(hands.has_hands(r) for r in run_list):
        return False
    st.info(NO_HAND_LOG_TRAINING if split is None else NO_HAND_LOG_EVAL, icon=":material/style:")
    return True


with tabs[7], table_budget():
    if tabs[7].open and not hand_log_missing(runs):
        if missing := [r.name for r in runs if not hands.has_hands(r)]:
            st.caption("No hand log, not shown: " + ", ".join(missing))
        logged = [r for r in runs if hands.has_hands(r)]
        types = hands.hand_type_stats(logged)
        c1, c2 = st.columns(2)
        with c1:
            chart(
                plots.hand_type_share(types, names),
                "hands played",
                "Share of each run's plays by poker hand, best hand at the top. The play "
                "policy picks the hand, so this shows what the decks and jokers the shop "
                "agent built let it make.",
            )
        with c2:
            chart(
                plots.hand_type_score(types, names),
                "hand score vs target",
                "Mean score of a single play of each hand, as a share of the blind's target "
                "chips. 100% beats the blind in one hand.",
            )
        played = hands.blind_play(logged)
        chart(
            plots.blind_play(played, names),
            "hands and discards per blind",
            "Mean hands played (solid) and discards used (dashed) per blind, by ante. More "
            "hands per blind means the build is struggling to reach the target.",
        )
        deck = hands.deck_composition(logged)
        if len(deck):
            chart(
                plots.deck_composition(deck, names),
                "deck composition",
                "Mean number of enhanced, editioned and sealed cards in the deck at the start "
                "of each ante's blinds (from tarots, spectrals and Standard packs).",
            )

# --- Episode explorer -----------------------------------------------------


def open_episode(episode: int) -> None:
    """Show ``episode`` in the explorer, clearing filters that would hide it."""
    for key in (*EPISODE_FILTERS, "ep_suggest"):
        st.session_state.pop(key, None)
    st.session_state.explorer_episode = episode


def open_selected_row() -> None:
    rows = st.session_state.ep_table.selection.rows
    if rows:
        open_episode(st.session_state._ep_rows[rows[0]])


def open_suggestion() -> None:
    label = st.session_state.get("ep_suggest")
    if label in st.session_state.get("_suggestions", {}):
        open_episode(st.session_state._suggestions[label])


def step_episode(options: list[int], by: int) -> None:
    i = options.index(st.session_state.explorer_episode) + by
    st.session_state.explorer_episode = options[max(0, min(i, len(options) - 1))]


def card_uri(card) -> str | None:
    return sprites.card_uri(card) if sprites else None


with tabs[8], table_budget():
    if not tabs[8].open:
        for key in ("run", "episode"):  # the link belongs to the explorer
            st.query_params.pop(key, None)
    if tabs[8].open:
        ex_top = st.columns([1, 3], vertical_alignment="bottom")
        run_name = ex_top[0].selectbox("Run", run_names, key="explorer_run")
        run = by_name[run_name]
        eps = run.episodes.set_index("episode")
        if need(eps, "episodes for this selection"):
            ep_table = find.episode_table(run)
            suggestions = find.suggested_episodes(ep_table)
            st.session_state._suggestions = {
                f"{s.label}: {s.episode}": s.episode for s in suggestions
            }
            ex_top[1].pills(
                "Suggested episodes",
                list(st.session_state._suggestions),
                key="ep_suggest",
                on_change=open_suggestion,
                help="\n\n".join(f"**{s.label}**: {s.reason}" for s in suggestions),
            )

            with st.expander("Browse episodes", icon=":material/list:"):
                f1, f2, f3 = st.columns(3)
                deck_opts = sorted(ep_table["deck"].unique())
                stake_opts = sorted(ep_table["stake"].unique())
                f_decks = f1.multiselect(
                    "Deck", deck_opts, key="ep_decks", placeholder="All", format_func=names.deck
                )
                f_stakes = f2.multiselect(
                    "Stake", stake_opts, key="ep_stakes", placeholder="All", format_func=names.stake
                )
                outcome = f3.segmented_control(
                    "Outcome", find.OUTCOMES, key="ep_outcome", default=find.OUTCOMES[0],
                    required=True,
                )  # fmt: skip
                lo, hi = int(ep_table["ante"].min()), int(ep_table["ante"].max())
                antes = (
                    st.slider("Ante reached", lo, hi, (lo, hi), key="ep_antes") if lo < hi else None
                )
                held = sorted({j for js in ep_table.get("final_jokers", []) for j in js})
                f_jokers = st.multiselect(
                    "Ended holding",
                    held,
                    key="ep_jokers",
                    placeholder="Any jokers",
                    format_func=names.item,
                )
                shown_eps = find.filter_episode_table(
                    ep_table, f_decks or None, f_stakes or None, antes, outcome, f_jokers
                )
                st.caption(
                    f"{len(shown_eps)} of {len(ep_table)} episodes. Click a row to open it, "
                    "a header to sort."
                )
                st.session_state._ep_rows = shown_eps["episode"].tolist()
                icon_table(
                    shown_eps,
                    ["final_jokers"],
                    {"lost_to": "blind"},
                    column_config={
                        names.column("score_vs_target"): st.column_config.NumberColumn(
                            format="percent",
                            help="Lost episodes: score ÷ target on the blind that ended the run",
                        )
                    },
                    on_select=open_selected_row,
                    selection_mode="single-row",
                    key="ep_table",
                    height=280,
                )

            options = shown_eps["episode"].tolist()
            if not options:
                st.caption("No episodes match the filters: stepping through all of them.")
                options = eps.index.tolist()
            if st.session_state.get("explorer_episode") not in options:
                st.session_state.explorer_episode = options[0]
            episode = st.session_state.explorer_episode
            st.query_params.update(run=run_name, episode=str(episode))
            row = eps.loc[episode]

            with st.container(border=True):
                head = st.columns([1, 7], vertical_alignment="center")
                with head[0]:
                    modal_html(details.deck_grid_html(deck_infos([row["deck"]]), names, "ex-deck"))
                with head[1]:
                    nav = st.container(horizontal=True, vertical_alignment="center")
                    pos = options.index(episode)
                    nav.button(
                        "Previous",
                        icon=":material/chevron_left:",
                        on_click=step_episode,
                        args=(options, -1),
                        disabled=pos == 0,
                        key="ep_prev",
                    )
                    nav.markdown(
                        f"**Episode {episode}** ({pos + 1} of {len(options)}), "
                        f"{names.deck(row['deck'])}, {names.stake(row['stake'])}, "
                        f"seed {row.get('seed', '-')}"
                    )
                    nav.button(
                        "Next",
                        icon=":material/chevron_right:",
                        on_click=step_episode,
                        args=(options, 1),
                        disabled=pos == len(options) - 1,
                        key="ep_next",
                    )
                    cols = st.columns(4)
                    cols[0].metric("Ante reached", int(row["ante"]))
                    cols[1].metric("Won", "Yes" if row["won"] else "No")
                    cols[2].metric("Shop decisions", int(row["shop_steps"]))
                    cols[3].metric("Money spent", f"${int(row['dollars_spent'])}")

            final = [j for j in str(row.get("final_jokers", "")).split(";") if j]
            if final:
                st.subheader("Final jokers")
                joker_grid([(j, "") for j in final], "ex-jk")

            ex_tabs = st.tabs(
                ["Replay", "Shop", "Money & blinds"], key="explorer_view", on_change="rerun"
            )
            with ex_tabs[0]:
                if ex_tabs[0].open and not hand_log_missing([run]):
                    acts = hands.episode_hands(run, episode)
                    blinds_played = acts.drop_duplicates("round")[["round", "ante", "blind_key"]]
                    labels = {
                        int(b.round): f"{b.ante}: {names.blind(b.blind_key)}"
                        for b in blinds_played.itertuples()
                    }
                    if st.session_state.get("ex_blind") not in labels:
                        st.session_state.ex_blind = next(iter(labels))
                    blind_round = st.segmented_control(
                        "Blind (ante: blind)",
                        list(labels),
                        key="ex_blind",
                        format_func=labels.get,
                        required=True,
                    )
                    rounds_ = metrics.episode_rounds(run, episode)
                    beaten = None
                    if "beaten" in rounds_:
                        match = rounds_[rounds_["round"] == blind_round]["beaten"]
                        beaten = bool(match.iloc[0]) if len(match) else None
                    held, logged, held_states = hands.blind_jokers(run, episode, blind_round)
                    st.markdown("**Jokers held**")
                    if held:
                        joker_grid(
                            [
                                (j, html.escape(cards.state_label(j, cards.joker_state(st_))))
                                for j, st_ in zip(held, held_states, strict=False)
                            ],
                            "ex-held",
                        )
                    st.caption(
                        ("None. " if not held else "")
                        + (
                            "Under a scaling joker, its value at the time (Spare Trousers' "
                            "+Mult, Ice Cream's +Chips). "
                            if any(cards.joker_state(x) for x in held_states)
                            else ""
                        )
                        + (
                            "As held at the first play or discard of this blind; changes "
                            "during the blind, including scaling jokers' values, are noted on "
                            "the action. Click a joker for its effect."
                            if logged
                            else "What was held leaving the shop before this blind (this run "
                            "didn't log jokers per play, so changes during the blind, "
                            "like Riff-raff or Ice Cream, aren't shown)."
                        )
                    )
                    st.html(
                        replay.blind_html(
                            acts[acts["round"] == blind_round], names, card_uri, icon_uri, beaten
                        )
                    )
                    decks_ = hands.episode_decks(run, episode)
                    if blind_round in decks_.index:
                        d = decks_.loc[blind_round]
                        with st.expander(
                            f"Deck at the start of this blind ({len(d['cards'])} cards)",
                            icon=":material/playing_cards:",
                        ):
                            st.html(replay.deck_html(d["cards"], card_uri))
                            levels = d.get("hand_levels")
                            if isinstance(levels, dict) and levels:
                                st.caption(
                                    "Hand levels: "
                                    + ", ".join(f"{h} {lv}" for h, lv in levels.items())
                                )
            with ex_tabs[1]:
                if ex_tabs[1].open:
                    one = metrics.filter_episodes(run, episode, episode)
                    if need_new(one.shops, "shop offers in this episode"):
                        st.caption(
                            "Each shop as it was offered, on entering and after every reroll. "
                            "Raised items with an orange price were bought; other decisions "
                            "are listed under the offer they were taken on."
                        )
                        st.html(
                            replay.shop_html(
                                economy.offers(one), one.shops, one.steps, names, icon_uri
                            )
                        )
            with ex_tabs[2]:
                if ex_tabs[2].open:
                    timeline = metrics.episode_timeline(run, episode)
                    if need(timeline, "shop decisions in this episode", run.steps is None):
                        chart(
                            plots.episode_money(timeline, names),
                            f"episode {episode} money {run_name}",
                            "Money after each shop decision in this episode, in order. Hover a "
                            "point for the action and item; jumps between shop visits are the "
                            "round's cash-out.",
                        )
                        timeline = with_joker_values(timeline, "jokers_before")
                        order = [
                            "step", "ante", "round", "dollars_before", "action_name", "item_key"
                        ]  # fmt: skip
                        order += [c for c in timeline.columns if c not in order]
                        item_table(
                            timeline[order],
                            ["item_key", "jokers_before", "consumables_before"],
                            "ex-shop",
                        )
                    st.subheader("Blinds played")
                    rounds = metrics.episode_rounds(run, episode)
                    if need(rounds, "rounds in this episode", run.rounds is None):
                        if "jokers" in rounds:
                            st.caption(
                                "Jokers and consumables are what was held entering each blind "
                                "(after the shop before it); the first blind comes before any "
                                "shop."
                            )
                        item_table(
                            with_joker_values(rounds, "jokers"),
                            ["jokers", "consumables"],
                            "ex-blind",
                        )

# --- Export (sidebar, after every chart is registered) --------------------
with export_area, st.expander("Export"):
    selection = {
        "runs": ", ".join(run_names),
        "episodes": episodes_shown(),
        "decks": ", ".join(decks),
        "stakes": ", ".join(stakes),
        "rolling window": window,
        "top N jokers": top,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "Data (ZIP of CSVs)",
        data=functools.partial(
            export.export_zip,
            runs,
            selection,
            window,
            top,
            st.session_state.get("power_target", 0.25),
            eval_runs,
        ),
        file_name=f"balatro_runs_{stamp}.zip",
        mime="application/zip",
        key="dl_data",
        on_click="ignore",
        help="Filtered raw data per run + every table, with keys and in-game names.",
    )
    fmt = st.radio("Chart format", ["png", "svg", "both"], horizontal=True, key="chart_fmt")
    fmts = ["png", "svg"] if fmt == "both" else [fmt]
    if problem := chart_export_problem():
        st.caption(f"Chart export unavailable: {problem}")
    st.download_button(
        f"Charts on this tab: {len(CHARTS)} (ZIP)",
        data=functools.partial(export.charts_zip, dict(CHARTS), fmts, dict(CHART_DESCRIPTIONS)),
        disabled=problem is not None,
        file_name=f"balatro_charts_{stamp}.zip",
        mime="application/zip",
        key="dl_charts",
        on_click="ignore",
        help="Every chart on the open tab (and sub-tab), light theme, 1600x900.",
    )
