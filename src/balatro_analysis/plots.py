"""Plotly figure builders. Each takes a frame from ``metrics`` and returns a figure."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from balatro_analysis import catalog

TEMPLATE = "plotly_white"


def _fig(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(template=TEMPLATE, margin={"l": 10, "r": 10, "t": 40, "b": 10}, **layout)
    return fig


def learning_curve(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.line(df, x="episode", y=metric, color="run", title=f"Rolling {metric}")
    return _fig(fig, hovermode="x unified")


def by_deck(df: pd.DataFrame) -> go.Figure:
    df = df.assign(label=df["deck name"] + " / " + df["stake"].map(catalog.stake_name))
    fig = px.bar(
        df,
        x="label",
        y="mean_ante",
        color="run",
        barmode="group",
        hover_data=["episodes", "max_ante", "win_rate"],
        title="Mean ante by deck and stake",
    )
    return _fig(fig, xaxis_title="", yaxis_title="mean ante")


def ante_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="ante", y="fraction", color="run", barmode="group", title="Ante reached")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_frequency(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df,
        x="action",
        y="share",
        color="run",
        barmode="group",
        hover_data=["count"],
        title="Share of shop decisions per action",
    )
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_by_ante(table: pd.DataFrame, run: str) -> go.Figure:
    fig = px.imshow(
        table,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "ante", "y": "action", "color": "share"},
        title=f"Action share within each ante: {run}",
    )
    return _fig(fig)


def money_by_ante(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="dollars_before",
        color="run",
        title="Money before each shop decision",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="$")


def spend_per_episode(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="dollars_spent",
        color="run",
        barmode="overlay",
        opacity=0.6,
        histnorm="probability",
        title="Money spent in the shop per episode",
    )
    return _fig(fig, xaxis_title="$ spent", yaxis_title="fraction of episodes")


def most_bought(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df,
        x="bought",
        y="name",
        color="run",
        barmode="group",
        orientation="h",
        title="Most-bought jokers",
    )
    fig.update_yaxes(categoryorder="total ascending", title="")
    return _fig(fig, height=max(350, 22 * df["name"].nunique()))


def first_bought_ante(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="ante",
        color="run",
        barmode="group",
        histnorm="probability",
        title="Ante in which each joker was first bought",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="fraction of joker purchases")


def cooccurrence(matrix: pd.DataFrame, run: str) -> go.Figure:
    matrix = matrix.rename(index=catalog.name, columns=catalog.name)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Purples",
        title=f"Jokers held together at end of episode: {run}",
    )
    return _fig(fig, height=max(400, 30 * len(matrix)))


def score_ratio(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="ratio",
        color="run",
        facet_col="blind type",
        category_orders={"blind type": ["Small", "Big", "Boss"]},
        title="Score / blind target (≥ 1 beats the blind)",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="grey")
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def deaths(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["run", "ante", "blind_key"], as_index=False).size()
    fig = px.bar(
        counts,
        x="blind_key",
        y="size",
        color="run",
        barmode="group",
        facet_col="ante",
        title="Blind that ended the run",
    )
    fig.update_xaxes(categoryorder="total descending", title="")
    return _fig(fig, yaxis_title="episodes")


def episode_money(timeline: pd.DataFrame) -> go.Figure:
    fig = px.line(
        timeline,
        x="step",
        y="dollars_after",
        markers=True,
        hover_data=["action_name", "item_key", "ante"],
        title="Money after each shop decision",
    )
    return _fig(fig, yaxis_title="$")
