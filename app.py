"""Streamlit dashboard for jackdaw-balatro shop-RL training runs.

uv run streamlit run app.py -- --runs-dir ../jackdaw-balatro/runs
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PACKAGE_DIR = Path(__file__).parent / "src" / "balatro_analysis"
RELOAD_ORDER = [
    "catalog",
    "names",
    "loader",
    "runcache",
    "metrics",
    "economy",
    "actions",
    "plots",
    "sprites",
    "game_text",
    "details",
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
    collection,
    details,
    economy,
    export,
    find,
    metrics,
    plots,
    runcache,
    stats,
)
from balatro_analysis.game_text import RARITY, GameText  # noqa: E402
from balatro_analysis.loader import Run, find_runs  # noqa: E402
from balatro_analysis.names import Names, split_items  # noqa: E402
from balatro_analysis.picker import PickerUnavailable, pick_directory  # noqa: E402
from balatro_analysis.sprites import Sprites, find_balatro, game_archive  # noqa: E402

sys.modules["balatro_analysis"]._app_mtime = package_mtime()  # see reload_if_changed

DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "jackdaw-balatro" / "runs"
MIN_VISITS = 5  # money-curve points need at least this many shop visits
MIN_LEGAL = 20  # preference needs an action to have been legal this often


def cli_runs_dir() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.runs_dir


# cache_resource hands back the cached Run itself: cache_data would unpickle every
# run (hundreds of MB) on each rerun.  Nothing writes to it; tabs filter copies.
@st.cache_resource(show_spinner="Loading run…", max_entries=256)
def cached_run(path: str, root: str, mtime: float, pkg_mtime: float) -> Run:
    # mtime: re-written runs reload; pkg_mtime: never reuse Runs from old code.
    # Runs unchanged since a previous session come from the disk cache.
    return runcache.load_run_cached(path, root, runcache.cache_dir())


@st.cache_resource(max_entries=128)
def cached_filtered(
    _run: Run, key: tuple, start: float, end: float, decks: tuple, stakes: tuple, rounds
) -> Run:
    """``metrics.filter_episodes``, kept across reruns: the same filters give the
    same Run, so what tabs memoise on it (``loader.memo``) is computed once.
    ``key`` identifies ``_run`` (not hashed): its path, mtime and the code mtime."""
    rounds = list(rounds) if rounds is not None else None
    return metrics.filter_episodes(
        _run, start, end, decks=list(decks), stakes=list(stakes), eval_rounds=rounds
    )


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
    files there are only the names."""
    present = [c for c in item_cols if c in df]
    if not sprites or not present:
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
st.title("Balatro run analysis")

if "runs_dir" not in st.session_state:
    st.session_state.runs_dir = str(cli_runs_dir())
if "game_path" not in st.session_state:
    found = find_balatro()
    st.session_state.game_path = str(found) if found else ""
if "pretty" not in st.session_state:
    st.session_state.pretty = st.query_params.get("names", "game") != "keys"

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.toggle(
        "In-game names",
        key="pretty",
        help="On: names as shown in the game (for presenting). "
        "Off: exact keys like j_greedy_joker (for analysis).",
    )
    names = Names(pretty=st.session_state.pretty)

    st.text_input("Runs folder", key="runs_dir")
    st.button("Browse…", on_click=browse_runs_dir, key="browse_runs")
    if err := st.session_state.pop("picker_error", None):
        st.error(f"{err}. Type the path instead.")
    root = Path(st.session_state.runs_dir).expanduser()
    run_paths = find_runs(root)
    if not run_paths:
        st.warning(f"No runs (folders with episodes.csv) under {root}")
        st.stop()
    run_dirs = {str(p.relative_to(root)) if p != root else p.name: p for p in run_paths}
    pkg_mtime = package_mtime()

    def run_mtime(path: Path) -> float:
        files = [path / "episodes.csv", path / "eval" / "episodes.csv"]
        return max(f.stat().st_mtime for f in files if f.is_file())

    def load(name: str) -> Run | None:
        """The run, or None if it has no episodes yet (still training?) or can't be read."""
        try:
            run = cached_run(str(run_dirs[name]), str(root), run_mtime(run_dirs[name]), pkg_mtime)
        except (ValueError, OSError):  # e.g. EmptyDataError: training hasn't logged yet
            return None
        return None if run.episodes.empty else run

    loaded = {n: load(n) for n in run_dirs}
    run_index = find.run_index(loaded)

    # Every run starts selected; runs that appear later (Reload runs) are added
    if st.session_state.get("_runs_root") != str(root):
        st.session_state.picked_runs = list(run_dirs)
    else:
        known = st.session_state.get("_known_runs", [])
        kept = [n for n in st.session_state.get("picked_runs", []) if n in run_dirs]
        st.session_state.picked_runs = kept + [n for n in run_dirs if n not in known]
    st.session_state._runs_root, st.session_state._known_runs = str(root), list(run_dirs)

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
            "filters below: " + ", ".join(hidden)
        )

    def pick_runs(names_: list[str]) -> None:
        st.session_state.picked_runs = names_

    with st.expander("Find runs"):
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
            .loc[shown_runs, ["mean_ante", "episodes", "group"]]
            .rename(columns=names.column),
            column_config={names.column("mean_ante"): st.column_config.NumberColumn(format="%.2f")},
            height=min(36 * (len(shown_runs) + 1) + 2, 300),
        )
        b1, b2, b3 = st.columns(3)
        b1.button(
            "Best per group",
            on_click=pick_runs,
            args=(find.best_per_group(run_index[run_index["run"].isin(shown_runs)]),),
            help="The highest mean-ante run of each group shown (the best run if none are grouped)",
        )
        b2.button("All shown", on_click=pick_runs, args=(shown_runs,))
        b3.button("Clear", on_click=pick_runs, args=([],))
    if not picked:
        st.stop()

    runs = [loaded[n] for n in picked if loaded[n] is not None]
    not_ready = [n for n in picked if loaded[n] is None]
    if not_ready:
        st.warning(
            f"Skipped {', '.join(not_ready)}: no episodes yet (still training?) or "
            "unreadable. **Reload runs** once it has logged some."
        )
    if not runs:
        st.stop()
    train_runs = runs

    # Eval episodes (greedy, fixed seeds) replace the training episodes in every tab
    with_eval = [r for r in runs if r.eval is not None]
    eval_mode = bool(with_eval) and (
        st.radio(
            "Episodes",
            ["Training", "Evaluation"],
            horizontal=True,
            key="phase",
            help="Evaluation: greedy episodes on the fixed eval seeds (--eval-every). "
            "Every run plays the same eval games, so comparisons pair game by game.",
        )
        == "Evaluation"
    )
    if eval_mode:
        if missing := [r.name for r in runs if r.eval is None]:
            st.caption("No eval episodes, not shown: " + ", ".join(missing))
        runs = [r.eval for r in with_eval]

    all_decks = sorted({d for r in runs for d in r.episodes["deck"]})
    all_stakes = sorted({s for r in runs for s in r.episodes["stake"]})
    decks = st.multiselect("Decks", all_decks, default=all_decks, format_func=names.deck)
    stakes = st.multiselect("Stakes", all_stakes, default=all_stakes, format_func=names.stake)

    max_ep = int(max(r.episodes["episode"].max() for r in runs))
    eval_rounds = None
    if eval_mode:
        ep_range = (0, max_ep)
        all_rounds = sorted({int(x) for r in runs for x in r.episodes["eval_round"]})
        eval_round = st.selectbox(
            "Eval round",
            ["All", *all_rounds],
            index=len(all_rounds),
            help="Eval round n is played after the training episodes shown in the Overview "
            "eval chart. The last round scores the final policy.",
        )
        eval_rounds = None if eval_round == "All" else [eval_round]
    else:
        ep_range = st.slider("Episodes", 0, max(max_ep, 1), (0, max_ep))
    window = st.slider("Rolling window", 1, max(2, max_ep + 1), min(50, max_ep + 1))
    if st.button("Reload runs"):
        cached_run.clear()
        cached_filtered.clear()
        st.rerun()

    with st.expander("Game assets (icons)"):
        st.text_input("Balatro install", key="game_path", help="Folder or Balatro.exe/.love")
        st.button("Browse…", on_click=browse_game, key="browse_game")
        archive = game_archive(st.session_state.game_path) if st.session_state.game_path else None
        sprites = cached_sprites(str(archive), pkg_mtime) if archive else None
        if sprites:
            st.caption(f"Icons from {archive}")
        else:
            st.caption("Balatro not found: showing names only.")

    export_area = st.container()  # filled at the end, once every chart is registered


def filtered(r: Run, start: float, end: float, rounds: list[int] | None = None) -> Run:
    key = (str(r.path), run_mtime(run_dirs[r.name]), pkg_mtime)
    return cached_filtered(
        r,
        key,
        start,
        end,
        tuple(decks),
        tuple(stakes),
        tuple(rounds) if rounds is not None else None,
    )


runs = [filtered(r, *ep_range, eval_rounds) for r in runs]
eval_runs = [filtered(r.eval, 0, float("inf")) for r in train_runs if r.eval is not None]
run_names = [r.name for r in runs]
by_name = {r.name: r for r in runs}
no_steps = any(r.steps is None for r in runs)
no_rounds = any(r.rounds is None for r in runs)

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
        "Episode explorer",
    ],
    key="main_tab",
    on_change="rerun",
)

# --- Overview -------------------------------------------------------------
with tabs[0]:
    if tabs[0].open:
        if eval_mode:
            st.info(
                "Showing **eval episodes** (greedy, fixed seeds), eval round "
                f"{eval_rounds[0] if eval_rounds else 'all'}. Runs played the same games, "
                "so the paired comparison below matches them game by game."
            )
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
                    + "). To compare runs game by game on one round, set **Episodes** to "
                    "*Evaluation* in the sidebar.",
                )
        if not eval_mode:
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
                "clearing ante 8.",
            )
        with st.expander("Run configs"):
            for r in runs:
                st.markdown(f"**{r.name}**")
                st.json(r.config or {"note": "no config.json"}, expanded=False)

# --- Decks ----------------------------------------------------------------
with tabs[1]:
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
with tabs[2]:
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
                bins = st.slider("Episode bins", 2, 50, 10, key="actions_bins")
                over = actions.family_over_training(runs, bins)
                if need(over, "per-step action data", no_steps):
                    chart(
                        plots.family_over_training(over, names),
                        "action families over training",
                        "Share of each kind of decision within each block of episodes (block "
                        "count set by the slider), one panel per run. A random agent stays "
                        "flat; a learning one should drift.",
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
with tabs[3]:
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

# --- Jokers ---------------------------------------------------------------
with tabs[4]:
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
with tabs[5]:
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
        note = "" if game_text else " · art and effect text need the game (see Game assets)"
        st.caption(
            f"Showing {len(shown)} of {len(cards)} jokers · stats from the selected runs"
            f" · click a joker for details{note}"
        )
        if shown:
            modal_html(collection.grid_html(shown, names))
        else:
            st.info("No jokers match.")

# --- Rounds ---------------------------------------------------------------
with tabs[6]:
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

# --- Episode explorer -----------------------------------------------------
EPISODE_FILTERS = ("ep_decks", "ep_stakes", "ep_antes", "ep_outcome", "ep_jokers")


def open_episode(episode: int) -> None:
    """Show ``episode`` in the explorer, clearing filters that would hide it."""
    for key in EPISODE_FILTERS:
        st.session_state.pop(key, None)
    st.session_state.explorer_episode = episode


def open_selected_row() -> None:
    rows = st.session_state.ep_table.selection.rows
    if rows:
        open_episode(st.session_state._ep_rows[rows[0]])


with tabs[7]:
    if not tabs[7].open:
        for key in ("run", "episode"):  # the link belongs to the explorer
            st.query_params.pop(key, None)
    if tabs[7].open:
        run_name = st.selectbox("Run", run_names, key="explorer_run")
        run = by_name[run_name]
        eps = run.episodes.set_index("episode")
        if need(eps, "episodes for this selection"):
            ep_table = find.episode_table(run)
            suggestions = find.suggested_episodes(ep_table)
            st.markdown("**Suggested episodes** · hover for why")
            for start in range(0, len(suggestions), 4):
                for col, sug in zip(st.columns(4), suggestions[start : start + 4], strict=False):
                    col.button(
                        f"{sug.label}: {sug.episode}",
                        help=sug.reason,
                        on_click=open_episode,
                        args=(sug.episode,),
                        key=f"suggest_{sug.label}",
                        width="stretch",
                    )

            with st.expander("Filter episodes"):
                f1, f2, f3 = st.columns(3)
                deck_opts = sorted(ep_table["deck"].unique())
                stake_opts = sorted(ep_table["stake"].unique())
                f_decks = f1.multiselect(
                    "Deck", deck_opts, key="ep_decks", placeholder="All", format_func=names.deck
                )
                f_stakes = f2.multiselect(
                    "Stake", stake_opts, key="ep_stakes", placeholder="All", format_func=names.stake
                )
                outcome = f3.radio("Outcome", find.OUTCOMES, key="ep_outcome", horizontal=True)
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
                f"{len(shown_eps)} of {len(ep_table)} episodes · click a row to open it, "
                "a header to sort"
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
                st.caption("No episodes match the filters: choosing from all of them.")
                options = eps.index.tolist()
            if st.session_state.get("explorer_episode") not in options:
                st.session_state.explorer_episode = options[0]
            episode = st.selectbox(
                "Episode",
                options,
                key="explorer_episode",
                format_func=lambda e: (
                    f"{e}  ({names.deck(eps.loc[e, 'deck'])}, ante {eps.loc[e, 'ante']}, "
                    f"{eps.loc[e, 'shop_steps']} shop steps)"
                ),
            )
            st.query_params.update(run=run_name, episode=str(episode))
            row = eps.loc[episode]
            deck_col, stats_col = st.columns([1, 6])
            with deck_col:
                modal_html(details.deck_grid_html(deck_infos([row["deck"]]), names, "ex-deck"))
                st.caption(names.stake(row["stake"]))
            with stats_col:
                cols = st.columns(5)
                cols[0].metric("Ante reached", int(row["ante"]))
                cols[1].metric("Won", "yes" if row["won"] else "no")
                cols[2].metric("Shop decisions", int(row["shop_steps"]))
                cols[3].metric("$ spent", int(row["dollars_spent"]))
                cols[4].metric("Seed", str(row.get("seed", "-")))

            final = [j for j in str(row.get("final_jokers", "")).split(";") if j]
            st.subheader("Final jokers")
            if final:
                joker_grid([(j, "") for j in final], "ex-jk")
            else:
                st.caption("None")

            timeline = metrics.episode_timeline(run, episode)
            if need(timeline, "shop decisions in this episode", run.steps is None):
                chart(
                    plots.episode_money(timeline, names),
                    f"episode {episode} money {run_name}",
                    "Money after each shop decision in this episode, in order. Hover a point "
                    "for the action and item; jumps between shop visits are the round's "
                    "cash-out.",
                )
                order = ["step", "ante", "round", "dollars_before", "action_name", "item_key"]
                order += [c for c in timeline.columns if c not in order]
                item_table(
                    timeline[order], ["item_key", "jokers_before", "consumables_before"], "ex-shop"
                )
            st.subheader("Blinds played")
            rounds = metrics.episode_rounds(run, episode)
            if need(rounds, "rounds in this episode", run.rounds is None):
                if "jokers" in rounds:
                    st.caption(
                        "Jokers and consumables are what was held entering each blind (after "
                        "the shop before it); the first blind comes before any shop."
                    )
                item_table(rounds, ["jokers", "consumables"], "ex-blind")

# --- Export (sidebar, after every chart is registered) --------------------
with export_area, st.expander("Export"):
    selection = {
        "runs": ", ".join(run_names),
        "episodes": (
            f"eval round {eval_rounds[0] if eval_rounds else 'all'}"
            if eval_mode
            else f"{ep_range[0]}-{ep_range[1]}"
        ),
        "decks": ", ".join(decks),
        "stakes": ", ".join(stakes),
        "rolling window": window,
        "top N jokers": top,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "Data (ZIP of CSVs)",
        data=functools.partial(export.export_zip, runs, selection, window, top),
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
