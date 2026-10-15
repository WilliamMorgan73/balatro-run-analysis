"""Streamlit dashboard for jackdaw-balatro shop-RL training runs.

uv run streamlit run app.py -- --runs-dir ../jackdaw-balatro/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from balatro_analysis import metrics, plots
from balatro_analysis.loader import Run, find_runs, load_run

DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "jackdaw-balatro" / "runs"


def cli_runs_dir() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.runs_dir


@st.cache_data(show_spinner="Loading run…")
def cached_run(path: str, root: str, mtime: float) -> Run:
    # mtime is part of the cache key so re-written runs reload
    return load_run(path, root)


def need(df, what: str, logs_missing: bool = False) -> bool:
    """Show a note and return False if a table is missing or empty."""
    if df is None or len(df) == 0:
        hint = " (needs steps.jsonl / rounds.jsonl)" if logs_missing else ""
        st.info(f"No {what}{hint}.")
        return False
    return True


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


st.set_page_config(page_title="Balatro Run Analysis", layout="wide")
st.title("Balatro run analysis")

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    root = Path(st.text_input("Runs directory", str(cli_runs_dir()))).expanduser()
    run_paths = find_runs(root)
    if not run_paths:
        st.warning(f"No runs (directories with episodes.csv) under {root}")
        st.stop()
    names = {str(p.relative_to(root)) if p != root else p.name: p for p in run_paths}
    picked = st.multiselect("Runs", list(names), default=list(names))
    if not picked:
        st.stop()
    runs = [
        cached_run(str(names[n]), str(root), (names[n] / "episodes.csv").stat().st_mtime)
        for n in picked
    ]
    max_ep = int(max(r.episodes["episode"].max() for r in runs))
    ep_range = st.slider("Episodes", 0, max(max_ep, 1), (0, max_ep))
    window = st.slider("Rolling window", 1, max(2, max_ep + 1), min(50, max_ep + 1))
    if st.button("Reload runs"):
        st.cache_data.clear()
        st.rerun()

runs = [metrics.filter_episodes(r, *ep_range) for r in runs]
run_names = [r.name for r in runs]
by_name = {r.name: r for r in runs}
no_steps = any(r.steps is None for r in runs)
no_rounds = any(r.rounds is None for r in runs)

tabs = st.tabs(["Overview", "Actions", "Economy", "Jokers", "Rounds", "Episode explorer"])

# --- Overview -------------------------------------------------------------
with tabs[0]:
    st.dataframe(
        metrics.summary(runs).style.format(
            {
                "mean ante": "{:.2f}",
                "win rate": "{:.1%}",
                "mean return": "{:.2f}",
                "shop steps / ep": "{:.1f}",
                "s / ep": "{:.3f}",
            },
            na_rep="-",
        ),
        hide_index=True,
        width="stretch",
    )
    curve = metrics.learning_curve(runs, window)
    metric = st.radio("Curve", ["mean ante", "win rate", "mean return"], horizontal=True)
    chart(plots.learning_curve(curve, metric))
    chart(plots.ante_distribution(metrics.ante_distribution(runs)))
    with st.expander("Run configs"):
        for r in runs:
            st.markdown(f"**{r.name}**")
            st.json(r.config or {"note": "no config.json"}, expanded=False)

# --- Actions --------------------------------------------------------------
with tabs[1]:
    freq = metrics.action_frequency(runs)
    if need(freq, "shop actions"):
        chart(plots.action_frequency(freq))
    run_name = st.selectbox("Run for heatmap", run_names, key="actions_run")
    table = metrics.action_by_ante(by_name[run_name])
    if need(table, "per-step action data", by_name[run_name].steps is None):
        chart(plots.action_by_ante(table, run_name))

# --- Economy --------------------------------------------------------------
with tabs[2]:
    money = metrics.money_by_ante(runs)
    if need(money, "per-step money data", no_steps):
        chart(plots.money_by_ante(money))
    spend = metrics.spend_per_episode(runs)
    if need(spend, "spend data"):
        chart(plots.spend_per_episode(spend))

# --- Jokers ---------------------------------------------------------------
with tabs[3]:
    top = st.slider("Top N jokers", 5, 40, 15)
    bought = metrics.most_bought(runs, top)
    if need(bought, "joker purchases", no_steps):
        chart(plots.most_bought(bought))
        chart(plots.first_bought_ante(metrics.first_bought_ante(runs)))
    run_name = st.selectbox("Run for build analysis", run_names, key="jokers_run")
    co = metrics.joker_cooccurrence(by_name[run_name], top)
    if need(co, "end-of-episode joker builds"):
        chart(plots.cooccurrence(co, run_name))
    swaps = metrics.sell_then_buy(by_name[run_name])
    st.subheader("Sold → next joker bought")
    if need(swaps, "sell-then-buy pairs", by_name[run_name].steps is None):
        st.dataframe(swaps, hide_index=True, width="stretch")

# --- Rounds ---------------------------------------------------------------
with tabs[4]:
    ratio = metrics.score_ratio(runs)
    if need(ratio, "round data", no_rounds):
        chart(plots.score_ratio(ratio))
    dead = metrics.deaths(runs)
    if need(dead, "lost rounds", no_rounds):
        chart(plots.deaths(dead))

# --- Episode explorer -----------------------------------------------------
with tabs[5]:
    run_name = st.selectbox("Run", run_names, key="explorer_run")
    run = by_name[run_name]
    eps = run.episodes.set_index("episode")
    episode = st.selectbox(
        "Episode",
        eps.index.tolist(),
        format_func=lambda e: (
            f"{e}  (ante {eps.loc[e, 'ante']}, {eps.loc[e, 'shop_steps']} shop steps)"
        ),
    )
    row = eps.loc[episode]
    cols = st.columns(4)
    cols[0].metric("Ante reached", int(row["ante"]))
    cols[1].metric("Won", "yes" if row["won"] else "no")
    cols[2].metric("Shop decisions", int(row["shop_steps"]))
    cols[3].metric("$ spent", int(row["dollars_spent"]))
    if row.get("final_jokers"):
        st.caption("Final jokers: " + row["final_jokers"].replace(";", ", "))

    timeline = metrics.episode_timeline(run, episode)
    if need(timeline, "shop decisions in this episode", run.steps is None):
        chart(plots.episode_money(timeline))
        st.dataframe(timeline, hide_index=True, width="stretch")
    st.subheader("Blinds played")
    rounds = metrics.episode_rounds(run, episode)
    if need(rounds, "rounds in this episode", run.rounds is None):
        st.dataframe(rounds, hide_index=True, width="stretch")
