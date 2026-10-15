"""Plotly figure builders. Each takes a frame from ``metrics`` plus a ``Names``
(in-game names vs exact keys) and returns a figure."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from balatro_analysis.names import Names

TEMPLATE = "plotly_white"


def _fig(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(template=TEMPLATE, margin={"l": 10, "r": 10, "t": 40, "b": 10}, **layout)
    return fig


def _labels(names: Names, *cols: str) -> dict[str, str]:
    """Axis/legend titles for ``cols`` in the current naming mode."""
    return {c: names.column(c) for c in cols}


def learning_curve(df: pd.DataFrame, metric: str, names: Names) -> go.Figure:
    fig = px.line(
        df,
        x="episode",
        y=metric,
        color="run",
        labels=_labels(names, "episode", "run"),
        title=f"Rolling {metric}",
    )
    return _fig(fig, hovermode="x unified")


def by_deck(df: pd.DataFrame, names: Names) -> go.Figure:
    df = df.assign(
        label=[
            f"{names.deck(d)} / {names.stake(s)}"
            for d, s in zip(df["deck"], df["stake"], strict=True)
        ]
    )
    fig = px.bar(
        df,
        x="label",
        y="mean_ante",
        color="run",
        barmode="group",
        hover_data=["episodes", "max_ante", "win_rate"],
        labels=_labels(names, "mean_ante", "run", "episodes", "max_ante", "win_rate"),
        title="Mean ante by deck and stake",
    )
    return _fig(fig, xaxis_title="")


def ante_distribution(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="ante",
        y="fraction",
        color="run",
        barmode="group",
        labels=_labels(names, "ante", "fraction", "run"),
        title="Ante reached",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_frequency(df: pd.DataFrame, names: Names) -> go.Figure:
    df = names.frame(df, {"action": "action"}, rename=False)
    fig = px.bar(
        df,
        x="action",
        y="share",
        color="run",
        barmode="group",
        hover_data=["count"],
        labels=_labels(names, "action", "share", "run", "count"),
        title="Share of shop decisions per action",
    )
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_by_ante(table: pd.DataFrame, run: str, names: Names) -> go.Figure:
    table = table.rename(index=names.action)
    fig = px.imshow(
        table,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": names.column("ante"),
            "y": names.column("action"),
            "color": names.column("share"),
        },
        title=f"Action share within each ante: {run}",
    )
    return _fig(fig)


def money_by_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="dollars_before",
        color="run",
        labels=_labels(names, "ante", "dollars_before", "run"),
        title="Money before each shop decision",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def spend_per_episode(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        df,
        x="dollars_spent",
        color="run",
        barmode="overlay",
        opacity=0.6,
        histnorm="probability",
        labels=_labels(names, "dollars_spent", "run"),
        title="Money spent in the shop per episode",
    )
    return _fig(fig, yaxis_title="fraction of episodes")


def most_bought(df: pd.DataFrame, names: Names) -> go.Figure:
    df = names.frame(df, rename=False)  # joker column -> names
    fig = px.bar(
        df,
        x="bought",
        y="joker",
        color="run",
        barmode="group",
        orientation="h",
        labels=_labels(names, "bought", "run", "joker"),
        title="Most-bought jokers",
    )
    fig.update_yaxes(categoryorder="total ascending", title="")
    return _fig(fig, height=max(350, 22 * df["joker"].nunique()))


def first_bought_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        df,
        x="ante",
        color="run",
        barmode="group",
        histnorm="probability",
        labels=_labels(names, "ante", "run"),
        title="Ante in which each joker was first bought",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="fraction of joker purchases")


def cooccurrence(matrix: pd.DataFrame, run: str, names: Names) -> go.Figure:
    matrix = matrix.rename(index=names.item, columns=names.item)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Purples",
        title=f"Jokers held together at end of episode: {run}",
    )
    return _fig(fig, height=max(400, 30 * len(matrix)))


def score_ratio(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="ratio",
        color="run",
        facet_col="blind type",
        category_orders={"blind type": ["Small", "Big", "Boss"]},
        labels=_labels(names, "ante", "ratio", "run", "blind type"),
        title="Score / blind target (≥ 1 beats the blind)",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="grey")
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def deaths(df: pd.DataFrame, names: Names) -> go.Figure:
    counts = df.groupby(["run", "ante", "blind_key"], as_index=False).size()
    counts = names.frame(counts, rename=False)
    fig = px.bar(
        counts,
        x="blind_key",
        y="size",
        color="run",
        barmode="group",
        facet_col="ante",
        labels=_labels(names, "ante", "run", "blind_key"),
        title="Blind that ended the run",
    )
    fig.update_xaxes(categoryorder="total descending", title="")
    return _fig(fig, yaxis_title=names.column("episodes"))


def episode_money(timeline: pd.DataFrame, names: Names) -> go.Figure:
    timeline = names.frame(timeline, rename=False)
    fig = px.line(
        timeline,
        x="step",
        y="dollars_after",
        markers=True,
        hover_data=["action_name", "item_key", "ante"],
        labels=_labels(names, "step", "dollars_after", "action_name", "item_key", "ante"),
        title="Money after each shop decision",
    )
    return _fig(fig)
