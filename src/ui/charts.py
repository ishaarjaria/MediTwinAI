"""
Chart builders. Pure functions: DataFrame in, Plotly figure out. All styling comes from
``src.ui.theme`` so charts match the rest of the UI.

Building a Plotly figure is the most expensive part of a page change (about 0.1 s of
property validation per chart), so the pure builders below are cached by their inputs with
``st.cache_data``: the same data and options return the same figure instantly, and any change
to the inputs (for example another date window) builds a new one. Charts that show What-If or
live form results are deliberately not cached.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.ui import theme

TEAL, BLUE, AMBER, CORAL, SLATE = (
    theme.SERIES[0], theme.SERIES[1], theme.SERIES[2], theme.SERIES[3], theme.SERIES[4],
)
_AXIS_TITLE = dict(size=11, color=theme.MUTED)


def _cached(fn):
    """Cache a pure chart builder by its arguments (bounded, no spinner)."""
    return st.cache_data(show_spinner=False, max_entries=64)(fn)


def _style(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(**theme.plotly_layout(height))
    return fig


def empty_figure(message: str, height: int = 240) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=theme.MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)


def _line(x, y, name, color, unit="", dash=None, width=2.2, fill=None):
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash, shape="linear"),
        fill=fill,
        hovertemplate=f"{name}: %{{y:,.1f}}{unit}<extra></extra>",
    )


# ----------------------------------------------------------------------
# Overview / demand
# ----------------------------------------------------------------------
@_cached
def opd_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame | None,
    height: int = 330,
    band_label: str = "Likely range",
) -> go.Figure:
    """
    Recent OPD history (solid blue) and the ML forecast (dashed teal) with its
    empirical likely range, separated by a shaded forecast zone and a 'Latest data' line.
    """
    if history.empty:
        return empty_figure("No OPD history in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(history["date"], history["opd_patients"], "Recorded outpatients", BLUE, width=2.2))
    if forecast is not None and not forecast.empty:
        last = pd.Timestamp(history["date"].iloc[-1])
        first_f, last_f = pd.Timestamp(forecast["date"].iloc[0]), pd.Timestamp(forecast["date"].iloc[-1])
        # Forecast zone and the 'latest data' marker (shapes avoid datetime-annotation quirks).
        fig.add_shape(type="rect", xref="x", yref="paper", y0=0, y1=1,
                      x0=last.isoformat(), x1=(last_f + pd.Timedelta(hours=12)).isoformat(),
                      fillcolor="rgba(46,125,115,.08)", line=dict(width=0), layer="below")
        fig.add_shape(type="line", xref="x", yref="paper", y0=0, y1=1,
                      x0=last.isoformat(), x1=last.isoformat(),
                      line=dict(color=theme.FAINT, width=1, dash="dot"))
        fig.add_annotation(x=first_f.isoformat(), xref="x", y=0.98, yref="paper", yanchor="top", xanchor="left",
                           text="Forecast", showarrow=False, font=dict(size=11, color=TEAL))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            mode="lines", fill="toself", fillcolor="rgba(46,125,115,.18)", line=dict(width=0),
            hoverinfo="skip", name=band_label, showlegend=True,
        ))
        # Join the forecast to the last observed point so the line reads as continuous.
        join_x = pd.concat([history["date"].tail(1), forecast["date"]])
        join_y = pd.concat([history["opd_patients"].tail(1), forecast["opd_forecast"]])
        fig.add_trace(go.Scatter(
            x=join_x, y=join_y, name="Expected outpatients", mode="lines+markers",
            line=dict(color=TEAL, width=2.6, dash="dot"),
            marker=dict(size=7, color=TEAL, line=dict(color=theme.SURFACE, width=1.5)),
            hovertemplate="Expected: %{y:,.0f}<extra></extra>",
        ))
    fig.update_yaxes(title_text="Patients per day", title_font=_AXIS_TITLE, rangemode="tozero")
    return _style(fig, height)


@_cached
def occupancy_trend(df: pd.DataFrame, height: int = 260) -> go.Figure:
    if df.empty:
        return empty_figure("No occupancy data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["bed_occupancy_percentage"], "Beds", TEAL, "%"))
    fig.add_trace(_line(df["date"], df["icu_occupancy_percentage"], "ICU", CORAL, "%"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return _style(fig, height)


@_cached
def demand_trend(df: pd.DataFrame, height: int = 300) -> go.Figure:
    if df.empty:
        return empty_figure("No demand data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["opd_patients"], "Outpatients", BLUE, width=1.3))
    fig.add_trace(_line(df["date"], df["opd_patients"].rolling(7, min_periods=1).mean(), "Outpatients, 7-day average", TEAL, width=2.8))
    fig.add_trace(_line(df["date"], df["emergency_patients"], "Emergency", AMBER, width=1.6))
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, height)


@_cached
def weekday_profile_chart(profile: pd.DataFrame, height: int = 260) -> go.Figure:
    if profile.empty:
        return empty_figure("No data.", height)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=profile["weekday"], y=profile["opd_patients"], name="Outpatients",
        marker=dict(color=BLUE, cornerradius=4),
        hovertemplate="%{x}: %{y:,.0f} outpatients<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=profile["weekday"], y=profile["emergency_patients"], name="Emergency",
        marker=dict(color=AMBER, cornerradius=4),
        hovertemplate="%{x}: %{y:,.0f} emergency<extra></extra>",
    ))
    fig.update_layout(barmode="group", hovermode="x unified", bargap=0.35)
    return _style(fig, height)


@_cached
def monthly_chart(monthly: pd.DataFrame, height: int = 260) -> go.Figure:
    if len(monthly) < 2:
        return empty_figure("Select a longer window to see monthly patterns.", height)
    fig = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["opd_patients"], marker=dict(color=TEAL, cornerradius=4),
        hovertemplate="%{x|%b %Y}: %{y:,.0f} avg outpatients<extra></extra>", name="Avg OPD",
    ))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(tickformat="%b")
    return _style(fig, height)


@_cached
def wait_scatter(df: pd.DataFrame, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_figure("No data in this window.", height)
    fig = go.Figure()
    for flag, name, color in ((0, "Weekday", BLUE), (1, "Weekend", AMBER)):
        part = df[df["is_weekend"] == flag]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["opd_patients"], y=part["average_waiting_time_minutes"], mode="markers", name=name,
            marker=dict(color=color, size=7, opacity=.75, line=dict(color=theme.SURFACE, width=.5)),
            customdata=part["date"].dt.strftime("%d %b %Y"),
            hovertemplate="%{customdata}<br>%{x:,.0f} outpatients · wait %{y:.1f} min<extra></extra>",
        ))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Outpatients that day", title_font=_AXIS_TITLE, showgrid=True, gridcolor=theme.GRID)
    fig.update_yaxes(title_text="Average waiting time (min)", title_font=_AXIS_TITLE)
    return _style(fig, height)


# ----------------------------------------------------------------------
# Capacity, flow, staffing, supplies
# ----------------------------------------------------------------------
@_cached
def flow_chart(df: pd.DataFrame, grain: str, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_figure("No admissions data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["admissions"], "Admissions", TEAL))
    fig.add_trace(_line(df["date"], df["discharges"], "Discharges", BLUE))
    fig.update_yaxes(title_text=f"Patients ({grain})", title_font=_AXIS_TITLE, rangemode="tozero")
    return _style(fig, height)


@_cached
def staffing_chart(df: pd.DataFrame, height: int = 280) -> go.Figure:
    if df.empty:
        return empty_figure("No staffing data in this window.", height)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(_line(df["date"], df["doctors_available"], "Doctors (left axis)", TEAL), secondary_y=False)
    fig.add_trace(_line(df["date"], df["nurses_available"], "Nurses (right axis)", BLUE), secondary_y=True)
    fig.update_yaxes(rangemode="tozero", secondary_y=False)
    fig.update_yaxes(rangemode="tozero", showgrid=False, secondary_y=True)
    return _style(fig, height)


@_cached
def stock_chart(df: pd.DataFrame, height: int = 260) -> go.Figure:
    if df.empty:
        return empty_figure("No medicine data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["medicine_stock_units"], "Stock (units)", TEAL))
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, height)


@_cached
def consumption_chart(df: pd.DataFrame, height: int = 260) -> go.Figure:
    if df.empty:
        return empty_figure("No medicine data in this window.", height)
    fig = go.Figure(go.Bar(
        x=df["date"], y=df["medicine_consumption_units"], marker=dict(color=AMBER),
        hovertemplate="Used: %{y:,.0f} units<extra></extra>", name="Consumption",
    ))
    fig.update_layout(bargap=0.2)
    return _style(fig, height)


@_cached
def wait_trend(df: pd.DataFrame, height: int = 260) -> go.Figure:
    if df.empty:
        return empty_figure("No waiting-time data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["average_waiting_time_minutes"], "Daily waiting time", BLUE, " min", width=1.3))
    fig.add_trace(_line(
        df["date"], df["average_waiting_time_minutes"].rolling(7, min_periods=1).mean(),
        "7-day average", TEAL, " min", width=2.8,
    ))
    fig.update_yaxes(rangemode="tozero", ticksuffix=" min")
    return _style(fig, height)


# ----------------------------------------------------------------------
# Anomalies and patient risk
# ----------------------------------------------------------------------
@_cached
def anomaly_chart(ops: pd.DataFrame, scores: pd.DataFrame | None, height: int = 300) -> go.Figure:
    if ops.empty:
        return empty_figure("No data in this window.", height)
    fig = go.Figure()
    fig.add_trace(_line(ops["date"], ops["opd_patients"], "Outpatients", BLUE, width=1.6))
    if scores is not None and not scores.empty:
        flagged = scores[scores["anomaly_flag"] == 1]
        flagged = flagged[flagged["date"].between(ops["date"].min(), ops["date"].max())]
        if not flagged.empty:
            fig.add_trace(go.Scatter(
                x=flagged["date"], y=flagged["opd_patients"], mode="markers", name="Flagged day",
                marker=dict(color=theme.RED, size=10, line=dict(color=theme.SURFACE, width=1.5)),
                customdata=flagged[["anomaly_reason", "anomaly_score"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>score %{customdata[1]:.3f}<extra></extra>",
            ))
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, height)


@_cached
def risk_distribution(patients: pd.DataFrame, height: int = 260) -> go.Figure:
    order = ["Low", "Medium", "High"]
    counts = patients["risk_level"].value_counts().reindex(order).fillna(0)
    colors = [TEAL, AMBER, theme.RED]
    fig = go.Figure(go.Bar(
        x=order, y=counts.values, marker=dict(color=colors, cornerradius=4),
        text=[f"{int(v):,}" for v in counts.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.INK),
        hovertemplate="%{x}: %{y:,.0f} patients<extra></extra>",
    ))
    fig.update_layout(bargap=0.45, showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, height)


@_cached
def risk_age_box(patients: pd.DataFrame, height: int = 260) -> go.Figure:
    fig = go.Figure()
    for level, color in (("Low", TEAL), ("Medium", AMBER), ("High", theme.RED)):
        part = patients[patients["risk_level"] == level]
        fig.add_trace(go.Box(y=part["age"], name=level, marker_color=color, line=dict(width=1.5), boxmean=False))
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(title_text="Age (years)", title_font=_AXIS_TITLE)
    return _style(fig, height)


# ----------------------------------------------------------------------
# Twin / what-if comparisons
# ----------------------------------------------------------------------
def paired_occupancy(
    labels: list[str], before: list[float], after: list[float],
    before_name: str, after_name: str, height: int = 220,
) -> go.Figure:
    """Grouped horizontal bars, 0-100%, e.g. current vs predicted occupancy."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=before, name=before_name, orientation="h",
        marker=dict(color=theme.SERIES[4], cornerradius=4),
        hovertemplate=f"{before_name}: %{{x:.1f}}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=after, name=after_name, orientation="h",
        marker=dict(color=TEAL, cornerradius=4),
        hovertemplate=f"{after_name}: %{{x:.1f}}%<extra></extra>",
    ))
    fig.update_layout(barmode="group", bargap=0.35, hovermode="y unified")
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=theme.GRID,
                     tickmode="array", tickvals=[0, 25, 50, 70, 85, 95, 100])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height)


# ----------------------------------------------------------------------
# Predictions page / patient risk
# ----------------------------------------------------------------------
def risk_probability_chart(probabilities: dict[str, float], height: int = 190) -> go.Figure:
    """Horizontal bars for the three class probabilities (Low / Medium / High)."""
    order = ["Low", "Medium", "High"]
    vals = [float(probabilities.get(k, 0.0)) for k in order]
    colors = [theme.SEVERITY["Low"][0], theme.SEVERITY["Medium"][0], theme.SEVERITY["High"][0]]
    fig = go.Figure(go.Bar(
        x=vals, y=order, orientation="h", marker=dict(color=colors),
        text=[f"{v:.0%}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return _style(fig, height)


@_cached
def model_comparison_chart(rows: list[dict], metric: str = "test_mae", title: str = "Test MAE (lower is better)",
                           height: int = 300) -> go.Figure:
    """Bars comparing models with their simple baselines; the selected model is highlighted."""
    usable = [r for r in rows if r.get(metric) is not None]
    if not usable:
        return empty_figure("No evaluated models to compare.", height)
    usable = sorted(usable, key=lambda r: r[metric], reverse=True)
    colors = [TEAL if r["selected"] else (BLUE if r["type"] == "model" else SLATE) for r in usable]
    fig = go.Figure(go.Bar(
        x=[r[metric] for r in usable], y=[r["model"] for r in usable], orientation="h",
        marker=dict(color=colors), text=[f"{r[metric]:.2f}" for r in usable], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title_text=title, title_font=_AXIS_TITLE, rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _style(fig, height)
