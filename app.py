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
    catalog,
    collection,
    details,
    economy,
    export,
    metrics,
    plots,
    stats,
)
from balatro_analysis.game_text import RARITY, GameText  # noqa: E402
from balatro_analysis.loader import Run, find_runs, load_run  # noqa: E402
from balatro_analysis.names import Names  # noqa: E402
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


@st.cache_data(show_spinner="Loading run…")
def cached_run(path: str, root: str, mtime: float, pkg_mtime: float) -> Run:
    # mtime: re-written runs reload; pkg_mtime: never reuse Runs from old code
    return load_run(path, root)


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


def chart(fig, name: str) -> None:
    """Show a figure, register it for 'download all', and offer PNG/SVG of it."""
    key = export.slug(name)
    CHARTS[key] = fig
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
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
    picked = st.multiselect("Runs", list(run_dirs), default=list(run_dirs))
    if not picked:
        st.stop()
    pkg_mtime = package_mtime()

    def run_mtime(path: Path) -> float:
        files = [path / "episodes.csv", path / "eval" / "episodes.csv"]
        return max(f.stat().st_mtime for f in files if f.is_file())

    runs, not_ready = [], []
    for n in picked:
        try:
            run = cached_run(str(run_dirs[n]), str(root), run_mtime(run_dirs[n]), pkg_mtime)
        except (ValueError, OSError):  # e.g. EmptyDataError: training hasn't logged yet
            run = None
        if run is None or run.episodes.empty:
            not_ready.append(n)
        else:
            runs.append(run)
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
        st.cache_data.clear()
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

runs = [
    metrics.filter_episodes(r, *ep_range, decks=decks, stakes=stakes, eval_rounds=eval_rounds)
    for r in runs
]
eval_runs = [
    metrics.filter_episodes(r.eval, 0, float("inf"), decks=decks, stakes=stakes)
    for r in train_runs
    if r.eval is not None
]
run_names = [r.name for r in runs]
by_name = {r.name: r for r in runs}
no_steps = any(r.steps is None for r in runs)
no_rounds = any(r.rounds is None for r in runs)

TOP_JOKERS_DEFAULT = 15
top = TOP_JOKERS_DEFAULT  # the Jokers tab's slider; the export reads it even when that tab is shut
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
                chart(fig, f"eval {emetric}")
                st.caption(
                    "Greedy episodes on the same eval seeds after each block of training ("
                    + (
                        "each group's runs averaged, Student-t CI across runs"
                        if by_group
                        else "bootstrap CI for ante, Wilson for win rate"
                    )
                    + "). To compare runs game by game on one round, set **Episodes** to "
                    "*Evaluation* in the sidebar."
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
                chart(fig, f"learning curve {metric}")
        if any(len(r.episodes) for r in runs):
            chart(plots.ante_distribution(metrics.ante_distribution(runs), names), "ante reached")
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
            chart(plots.by_deck(deck_table, names), "mean ante by deck")
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
                    chart(plots.family_share(fam, names), "action family share")
                freq = metrics.action_frequency(runs)
                if need(freq, "shop actions"):
                    chart(plots.action_frequency(freq, names), "action share")
                visits = actions.decisions_per_visit(runs)
                if need(visits, "shop visits", no_steps):
                    chart(plots.decisions_per_visit(visits, names), "decisions per visit")

        with sub[1]:
            if sub[1].open:
                left, right = st.columns(2)
                run_name = left.selectbox("Run", run_names, key="actions_run")
                split = right.selectbox("Split by", list(actions.SPLITS), key="actions_split")
                heat = actions.share_by(by_name[run_name], split)
                if need(heat, "per-step action data", by_name[run_name].steps is None):
                    st.caption(
                        f"Each column sums to 100%: what the agent did given the {split.lower()}"
                        + (" (money before the decision)." if split == "Money" else ".")
                    )
                    chart(
                        plots.action_heatmap(
                            heat, f"Action share by {split.lower()}: {run_name}", split, names
                        ),
                        f"actions by {split.lower()} {run_name}",
                    )

        with sub[2]:
            if sub[2].open:
                run_name = st.selectbox("Run", run_names, key="actions_seq_run")
                run = by_name[run_name]
                matrix = actions.transition_matrix(run)
                if need(matrix, "per-step action data", run.steps is None):
                    chart(
                        plots.transition_matrix(matrix, run_name, names), f"transitions {run_name}"
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
                    st.caption(
                        "Share of each kind of decision within each block of episodes. A random "
                        "agent stays flat; a learning one should drift."
                    )
                    chart(plots.family_over_training(over, names), "action families over training")

        with sub[4]:
            if sub[4].open:
                avail = actions.availability(runs)
                if need_new(avail, "legal-action data", version=3):
                    missing = sorted(set(run_names) - set(avail["run"]))
                    if missing:
                        st.caption("Not shown (logged before log_version 3): " + ", ".join(missing))
                    st.caption(
                        "Uniform-random rate: the chance of picking the action if the agent "
                        "chose evenly among the legal actions each time it was legal. A "
                        "preference of 1 is random-like; above 1 the agent favours it. Error "
                        "bars are 95% intervals; q is the BH-adjusted p-value of a test "
                        "against uniform-random choice (q < 0.05: a real preference)."
                    )
                    chart(plots.availability(avail, names), "action availability")
                    st.caption(
                        f"Preference leaves out actions legal fewer than {MIN_LEGAL} times "
                        "(too few to tell)."
                    )
                    chart(
                        plots.preference(avail[avail["legal_count"] >= MIN_LEGAL], names),
                        "action preference vs uniform",
                    )
                    chart(plots.legal_counts(actions.legal_counts(runs), names), "legal actions")

        with sub[5]:
            if sub[5].open:
                corr = actions.action_outcome_correlations(runs)
                if need(corr, "enough episodes with shop data", no_steps):
                    st.caption(
                        "Spearman ρ between how often each kind of decision is made per shop "
                        "visit and the ante reached (+1: more of it goes with deeper runs). "
                        "Per visit, because longer runs visit more shops. "
                        "\\* = significant at q < 0.05 (p-values Benjamini-Hochberg "
                        "adjusted within each run); hover for p and q."
                    )
                    chart(plots.action_outcome_correlations(corr, names), "actions vs ante reached")

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
                    st.caption(
                        f"Rounds with fewer than {MIN_VISITS} shop visits are left off the "
                        "money curve (too few runs get that far to say much)."
                    )
                    chart(
                        plots.money_curve(curve[curve["visits"] >= MIN_VISITS], names),
                        "money curve",
                    )
                    chart(plots.entry_exit_by_ante(visits, names), "money entering vs leaving")
                    money = metrics.money_by_ante(runs)
                    chart(plots.money_by_ante(money, names), "money before each decision")
                if "final_dollars" in econ and econ["final_dollars"].notna().any():
                    chart(plots.final_dollars(econ, names), "money at end of run")

        with sub[1]:
            if sub[1].open:
                spend = economy.spend_by_category(runs)
                if need(spend, "purchases", no_steps):
                    chart(plots.spend_share(spend, names), "spending by category")
                    chart(
                        plots.spend_by_ante(economy.spend_by_category(runs, by_ante=True), names),
                        "spending mix by ante",
                    )
                income = economy.income_by_source(runs)
                if need(income, "income"):
                    chart(plots.income_sources(income, names), "income by source")
                spend_ep = metrics.spend_per_episode(runs)
                if need(spend_ep, "spend data"):
                    chart(plots.spend_per_episode(spend_ep, names), "spend per episode")

        with sub[2]:
            if sub[2].open:
                if need(visits, "shop visits", no_steps):
                    st.caption(
                        "Interest is paid at cash-out: $1 per $5 held, up to the cap. "
                        "Estimated here from the money left when leaving the shop."
                    )
                    chart(plots.exit_brackets(visits, names), "money left after shopping")
                    chart(plots.interest_by_ante(visits, names), "estimated interest by ante")
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
                    chart(plots.interest_efficiency(eff, names), "interest efficiency")

        with sub[3]:
            if sub[3].open:
                if need(visits, "shop visits", no_steps):
                    chart(plots.rerolls_per_visit(visits, names), "rerolls per visit")
                    chart(
                        plots.reroll_by_ante(economy.reroll_stats(runs), names),
                        "reroll spend by ante",
                    )
                paid = economy.price_paid(runs)
                if need(paid, "purchases", no_steps):
                    chart(plots.price_paid(paid, names), "price paid")
                    chart(plots.joker_price_by_ante(paid, names), "joker price by ante")

        with sub[4]:
            if sub[4].open:
                corr = economy.outcome_correlations(econ)
                if need(corr, "enough episodes with shop data", no_steps):
                    st.caption(
                        "Spearman ρ between each habit and the ante reached (+1: more of it "
                        "goes with deeper runs). Totals grow with run length, so they "
                        "correlate almost by construction; compare the per-visit measures. "
                        "\\* = significant at q < 0.05 (p-values Benjamini-Hochberg "
                        "adjusted within each run); hover for p and q."
                    )
                    chart(plots.outcome_correlations(corr, names), "economy vs ante reached")
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
                    )

        with sub[5]:
            if sub[5].open:
                breakdown = economy.income_breakdown(runs)
                if need_new(breakdown, "cash-out breakdown"):
                    chart(plots.income_breakdown(breakdown, names), "cash-out income")
                    eff = economy.interest_efficiency(runs)
                    if need_new(eff, "exact interest"):
                        chart(
                            plots.interest_efficiency(eff, names), "interest efficiency (cash-out)"
                        )

        with sub[6]:
            if sub[6].open:
                conversion = economy.offer_conversion(runs)
                if need_new(conversion, "shop offers"):
                    chart(plots.offer_conversion(conversion, names), "offer conversion")
                    chart(
                        plots.offered_vs_paid(conversion, economy.price_paid(runs), names),
                        "price offered vs paid",
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
            chart(plots.most_bought(bought, names), "most bought jokers")
        first = metrics.first_bought_ante(runs)
        if len(first):
            chart(plots.first_bought_ante(first, names), "first bought ante")
        co = metrics.joker_cooccurrence(run, top)
        if need(co, "end-of-episode joker builds"):
            chart(plots.cooccurrence(co, run_name, names), f"joker cooccurrence {run_name}")
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
            chart(plots.score_ratio(ratio, names), "score vs target")
        dead = metrics.deaths(runs)
        if need(dead, "lost rounds", no_rounds):
            chart(plots.deaths(dead, names), "blind that ended the run")

# --- Episode explorer -----------------------------------------------------
with tabs[7]:
    if tabs[7].open:
        run_name = st.selectbox("Run", run_names, key="explorer_run")
        run = by_name[run_name]
        eps = run.episodes.set_index("episode")
        if need(eps, "episodes for this selection"):
            episode = st.selectbox(
                "Episode",
                eps.index.tolist(),
                format_func=lambda e: (
                    f"{e}  ({names.deck(eps.loc[e, 'deck'])}, ante {eps.loc[e, 'ante']}, "
                    f"{eps.loc[e, 'shop_steps']} shop steps)"
                ),
            )
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
                chart(plots.episode_money(timeline, names), f"episode {episode} money {run_name}")
                timeline = timeline.assign(
                    icon=timeline["item_key"].map(
                        lambda k: sprites.data_uri(k) if sprites and catalog.is_joker(k) else None
                    )
                )
                order = [
                    "step",
                    "ante",
                    "round",
                    "dollars_before",
                    "action_name",
                    "icon",
                    "item_key",
                ]
                order += [c for c in timeline.columns if c not in order]
                table(
                    timeline[order],
                    column_config={
                        names.column("icon"): st.column_config.ImageColumn("", width="small")
                    },
                    row_height=60 if sprites else None,
                )
            st.subheader("Blinds played")
            rounds = metrics.episode_rounds(run, episode)
            if need(rounds, "rounds in this episode", run.rounds is None):
                table(rounds)

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
        data=functools.partial(export.charts_zip, dict(CHARTS), fmts),
        disabled=problem is not None,
        file_name=f"balatro_charts_{stamp}.zip",
        mime="application/zip",
        key="dl_charts",
        on_click="ignore",
        help="Every chart on the open tab (and sub-tab), light theme, 1600x900.",
    )
