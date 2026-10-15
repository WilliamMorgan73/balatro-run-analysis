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

import streamlit as st

PACKAGE_DIR = Path(__file__).parent / "src" / "balatro_analysis"
RELOAD_ORDER = ["catalog", "names", "loader", "metrics", "plots", "sprites", "picker", "export"]


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

from balatro_analysis import export, metrics, plots  # noqa: E402
from balatro_analysis.loader import Run, find_runs, load_run  # noqa: E402
from balatro_analysis.names import Names  # noqa: E402
from balatro_analysis.picker import PickerUnavailable, pick_directory  # noqa: E402
from balatro_analysis.sprites import Sprites, find_balatro, game_archive  # noqa: E402

sys.modules["balatro_analysis"]._app_mtime = package_mtime()  # see reload_if_changed

DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "jackdaw-balatro" / "runs"
ICON_WIDTH = 71  # Balatro's 1x card size; sprites are 2x for sharpness


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


@st.cache_resource
def cached_sprites(archive: str) -> Sprites:
    return Sprites(Path(archive))


def need(df, what: str, logs_missing: bool = False) -> bool:
    """Show a note and return False if a table is missing or empty."""
    if df is None or len(df) == 0:
        hint = " (needs steps.jsonl / rounds.jsonl)" if logs_missing else ""
        st.info(f"No {what}{hint}.")
        return False
    return True


CHARTS: dict[str, object] = {}


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


def icon_grid(items: list[tuple[str, str]], per_row: int = 8) -> None:
    """Show (key, caption) pairs as icons with captions, or captions alone without sprites."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (key, caption) in zip(cols, items[i : i + per_row], strict=False):
            png = sprites.png(key) if sprites else None
            with col:
                if png:
                    st.image(png, width=ICON_WIDTH)
                st.caption(caption)


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
    runs = [
        cached_run(
            str(run_dirs[n]), str(root), (run_dirs[n] / "episodes.csv").stat().st_mtime, pkg_mtime
        )
        for n in picked
    ]

    all_decks = sorted({d for r in runs for d in r.episodes["deck"]})
    all_stakes = sorted({s for r in runs for s in r.episodes["stake"]})
    decks = st.multiselect("Decks", all_decks, default=all_decks, format_func=names.deck)
    stakes = st.multiselect("Stakes", all_stakes, default=all_stakes, format_func=names.stake)

    max_ep = int(max(r.episodes["episode"].max() for r in runs))
    ep_range = st.slider("Episodes", 0, max(max_ep, 1), (0, max_ep))
    window = st.slider("Rolling window", 1, max(2, max_ep + 1), min(50, max_ep + 1))
    if st.button("Reload runs"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("Game assets (icons)"):
        st.text_input("Balatro install", key="game_path", help="Folder or Balatro.exe/.love")
        st.button("Browse…", on_click=browse_game, key="browse_game")
        archive = game_archive(st.session_state.game_path) if st.session_state.game_path else None
        sprites = cached_sprites(str(archive)) if archive else None
        if sprites:
            st.caption(f"Icons from {archive}")
        else:
            st.caption("Balatro not found: showing names only.")

    export_area = st.container()  # filled at the end, once every chart is registered

runs = [metrics.filter_episodes(r, *ep_range, decks=decks, stakes=stakes) for r in runs]
run_names = [r.name for r in runs]
by_name = {r.name: r for r in runs}
no_steps = any(r.steps is None for r in runs)
no_rounds = any(r.rounds is None for r in runs)

tabs = st.tabs(["Overview", "Decks", "Actions", "Economy", "Jokers", "Rounds", "Episode explorer"])

# --- Overview -------------------------------------------------------------
with tabs[0]:
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
    curve = metrics.learning_curve(runs, window)
    if need(curve, "episodes for this selection"):
        metric = st.radio("Curve", ["mean ante", "win rate", "mean return"], horizontal=True)
        chart(plots.learning_curve(curve, metric, names), f"learning curve {metric}")
        chart(plots.ante_distribution(metrics.ante_distribution(runs), names), "ante reached")
    with st.expander("Run configs"):
        for r in runs:
            st.markdown(f"**{r.name}**")
            st.json(r.config or {"note": "no config.json"}, expanded=False)

# --- Decks ----------------------------------------------------------------
with tabs[1]:
    deck_table = metrics.by_deck(runs)
    if need(deck_table, "episodes for this selection"):
        overall = (
            deck_table.assign(antes=deck_table["mean_ante"] * deck_table["episodes"])
            .groupby("deck", as_index=False)[["episodes", "antes"]]
            .sum()
        )
        icon_grid(
            [
                (
                    d.deck,
                    f"**{names.deck(d.deck)}**  \n{d.episodes} episodes  \n"
                    f"mean ante {d.antes / d.episodes:.2f}",
                )
                for d in overall.itertuples()
            ]
        )
        chart(plots.by_deck(deck_table, names), "mean ante by deck")
        table(
            deck_table[["run", "deck", "stake", "episodes", "mean_ante", "max_ante", "win_rate"]],
            column_config={
                names.column("mean_ante"): st.column_config.NumberColumn(format="%.2f"),
                names.column("win_rate"): st.column_config.NumberColumn(format="percent"),
            },
        )

# --- Actions --------------------------------------------------------------
with tabs[2]:
    freq = metrics.action_frequency(runs)
    if need(freq, "shop actions"):
        chart(plots.action_frequency(freq, names), "action share")
    run_name = st.selectbox("Run for heatmap", run_names, key="actions_run")
    heat = metrics.action_by_ante(by_name[run_name])
    if need(heat, "per-step action data", by_name[run_name].steps is None):
        chart(plots.action_by_ante(heat, run_name, names), f"actions by ante {run_name}")

# --- Economy --------------------------------------------------------------
with tabs[3]:
    money = metrics.money_by_ante(runs)
    if need(money, "per-step money data", no_steps):
        chart(plots.money_by_ante(money, names), "money by ante")
    spend = metrics.spend_per_episode(runs)
    if need(spend, "spend data"):
        chart(plots.spend_per_episode(spend, names), "spend per episode")

# --- Jokers ---------------------------------------------------------------
with tabs[4]:
    top = st.slider("Top N jokers", 5, 40, 15)
    run_name = st.selectbox("Run", run_names, key="jokers_run")
    run = by_name[run_name]
    bought_one = metrics.most_bought([run], top)
    st.subheader(f"Most-bought jokers: {run_name}")
    if need(bought_one, "joker purchases", run.steps is None):
        icon_grid(
            [
                (b.joker, f"**{names.item(b.joker)}**  \nbought {b.bought}x")
                for b in bought_one.itertuples()
            ]
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

# --- Rounds ---------------------------------------------------------------
with tabs[5]:
    ratio = metrics.score_ratio(runs)
    if need(ratio, "round data", no_rounds):
        chart(plots.score_ratio(ratio, names), "score vs target")
    dead = metrics.deaths(runs)
    if need(dead, "lost rounds", no_rounds):
        chart(plots.deaths(dead, names), "blind that ended the run")

# --- Episode explorer -----------------------------------------------------
with tabs[6]:
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
            deck_png = sprites.png(row["deck"]) if sprites else None
            if deck_png:
                st.image(deck_png, width=ICON_WIDTH)
            st.caption(f"**{names.deck(row['deck'])}**  \n{names.stake(row['stake'])}")
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
            icon_grid([(j, names.item(j)) for j in final])
        else:
            st.caption("None")

        timeline = metrics.episode_timeline(run, episode)
        if need(timeline, "shop decisions in this episode", run.steps is None):
            chart(plots.episode_money(timeline, names), f"episode {episode} money {run_name}")
            timeline = timeline.assign(
                icon=timeline["item_key"].map(
                    lambda k: sprites.data_uri(k) if sprites and str(k).startswith("j_") else None
                )
            )
            order = ["step", "ante", "round", "dollars_before", "action_name", "icon", "item_key"]
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
        "episodes": f"{ep_range[0]}-{ep_range[1]}",
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
        f"All {len(CHARTS)} charts (ZIP)",
        data=functools.partial(export.charts_zip, dict(CHARTS), fmts),
        disabled=problem is not None,
        file_name=f"balatro_charts_{stamp}.zip",
        mime="application/zip",
        key="dl_charts",
        on_click="ignore",
        help="Every chart currently shown, light theme, 1600x900.",
    )
