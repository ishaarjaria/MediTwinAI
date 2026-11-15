"""
Reusable presentation components (dark command-centre theme).

Builders return single-line HTML strings (no indentation or blank lines, so
Markdown never turns them into code blocks). All text is escaped. Streamlit
rendering is limited to the small ``md`` / ``card`` / ``plot`` helpers at the
bottom.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any, Iterable

from src.ui import theme
from src.ui.data import CAPACITY_MARKS, delta_arrow, fmt_num, fmt_signed


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


esc = _e  # public alias for views that assemble table cells


# ----------------------------------------------------------------------
# Atoms
# ----------------------------------------------------------------------
def info(text: str) -> str:
    """Small 'i' marker whose hover/focus tooltip explains a term in plain language."""
    return f'<span class="mt-info" tabindex="0" role="note" aria-label="{_e(text)}" data-tip="{_e(text)}">i</span>'


def badge(key: str, compact: bool = False) -> str:
    """Provenance badge: observed / ml / rule / unavailable."""
    label, ink, bg = theme.PROVENANCE.get(key, theme.PROVENANCE["unavailable"])
    if compact:
        label = theme.PROVENANCE_SHORT.get(key, label)
    title = theme.PROVENANCE.get(key, theme.PROVENANCE["unavailable"])[0]
    return f'<span class="mt-badge" title="{_e(title)}" style="color:{ink};background:{bg}">{_e(label)}</span>'


def pill(label: str, ink: str, bg: str) -> str:
    return f'<span class="mt-pill" style="color:{ink};background:{bg}">{_e(label)}</span>'


def status_pill(status: str) -> str:
    ink, bg = theme.STATUS.get(status, theme.STATUS["Normal"])
    return pill(status, ink, bg)


def severity_pill(label: str) -> str:
    """Severity pill with a shape glyph so colour is never the only cue."""
    ink, bg = theme.SEVERITY.get(label, theme.SEVERITY["Low"])
    glyph = theme.SEVERITY_GLYPH.get(label, "○")
    return (
        f'<span class="mt-pill sev" style="color:{ink};background:{bg}">'
        f'<span class="g" aria-hidden="true">{glyph}</span>{_e(label)}</span>'
    )


def chip(label: str, count: int, severity: str) -> str:
    ink, bg = theme.SEVERITY[severity]
    return f'<span class="mt-chip" style="color:{ink};background:{bg}"><b>{count}</b>{_e(label)}</span>'


def date_chip(text: str) -> str:
    return f'<span class="mt-chip-date">{_e(text)}</span>'


def occupancy_color(pct: float) -> str:
    """Bar colour following the twin's capacity ladder (presentation only)."""
    low, mid, high = CAPACITY_MARKS
    if pct >= high:
        return theme.SEVERITY["Critical"][0]
    if pct >= mid:
        return theme.SEVERITY["High"][0]
    if pct >= low:
        return theme.SEVERITY["Medium"][0]
    return theme.TEAL


def occupancy_bar(pct: float | None, height: int = 8) -> str:
    """Horizontal occupancy bar with tick marks at the capacity thresholds."""
    if pct is None or not math.isfinite(pct):
        return '<div class="mt-bar"></div>'
    pct = max(0.0, min(100.0, pct))
    ticks = "".join(
        f'<i style="position:absolute;left:{m}%;top:0;bottom:0;width:1px;background:rgba(20,43,42,.35)"></i>'
        for m in CAPACITY_MARKS
    )
    return (
        f'<div class="mt-bar" style="height:{height}px">'
        f'<span style="width:{pct:.1f}%;background:{occupancy_color(pct)}"></span>{ticks}</div>'
    )


def delta_html(delta: float | None, tone: str, decimals: int = 1, unit: str = "") -> str:
    if delta is None:
        return '<span class="mt-delta neutral">—</span>'
    arrow = delta_arrow(delta, decimals)
    text = fmt_signed(delta, decimals, unit)
    return f'<span class="mt-delta {tone}">{arrow} {_e(text)}</span>'


# ----------------------------------------------------------------------
# Shell
# ----------------------------------------------------------------------
def logo_svg(size: int = 36) -> str:
    """MediTwin mark: a pulse line and its faint twin on a deep-sage tile (pure SVG shapes)."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="MediTwin">'
        f'<rect width="36" height="36" rx="10" fill="{theme.PRIMARY}"/>'
        f'<path d="M7 22h5l2.6-7 4.2 13 3.4-10 2 4H29" stroke="{theme.SAGE}" stroke-opacity=".55" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M7 18h5l2.6-7 4.2 13 3.4-10 2 4H29" stroke="#FFFFFF" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def brand_html() -> str:
    return (
        f'<div class="mt-brand">{logo_svg()}<div>'
        '<div class="mt-brand-name">MediTwin</div>'
        '<div class="mt-brand-sub">Operations centre</div></div></div>'
    )


def sidebar_footer(data_through: str | None) -> str:
    """Data-status block (dataset date, synthetic indicator) plus the prototype disclaimer."""
    when = f'<div class="mt-status-line"><b>Dataset through</b> {_e(data_through)}</div>' if data_through else ""
    return (
        '<div class="mt-status-block">'
        '<div class="mt-status-row"><span class="mt-dot"></span>Synthetic data</div>'
        f'{when}'
        '<div class="mt-status-line">Synthetic dataset, not a live feed. Nothing updates in real time.</div></div>'
        '<div class="mt-side-disclaimer">Educational prototype. Not a clinical decision system; no diagnosis, '
        "treatment or triage.</div>"
    )


def user_block(name: str, role: str, initials: str) -> str:
    """Signed-in user chip for the sidebar."""
    return (
        f'<div class="mt-user"><div class="av">{_e(initials)}</div>'
        f'<div><div class="nm">{_e(name)}</div><div class="rl">{_e(role)}</div></div></div>'
    )


def page_header(title: str, subtitle: str = "", meta_html: str = "", eyebrow: str = "") -> str:
    eb = f'<div class="mt-eyebrow">{_e(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="mt-subtitle">{_e(subtitle)}</div>' if subtitle else ""
    meta = f'<div class="mt-meta">{meta_html}</div>' if meta_html else ""
    return f'<div class="mt-header"><div>{eb}<div class="mt-title">{_e(title)}</div>{sub}</div>{meta}</div>'


# ----------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------
def section(title: str, right_html: str = "") -> str:
    return f'<div class="mt-section-row"><div class="mt-section">{_e(title)}</div>{right_html}</div>'


def how_to_read(text: str) -> str:
    """Short plain-language note that explains a chart or table."""
    return f'<div class="mt-how"><b>How to read this.</b> {_e(text)}</div>'


def tile(
    label: str,
    value: str,
    unit: str = "",
    sub: str = "",
    bar_pct: float | None = None,
    foot_html: str = "",
    compact: bool = False,
    tip: str | None = None,
) -> str:
    """KPI tile. ``compact`` drops the equal-height minimum; ``tip`` adds an info tooltip."""
    unit_html = f'<span class="mt-tile-unit">{_e(unit)}</span>' if unit else ""
    bar = occupancy_bar(bar_pct) if bar_pct is not None else ""
    foot = f'<div class="mt-tile-foot">{foot_html}</div>' if foot_html else ""
    style = ' style="min-height:0"' if compact else ""
    tip_html = info(tip) if tip else ""
    return (
        f'<div class="mt-tile"{style}><div class="mt-tile-top"><div class="mt-tile-label">{_e(label)}</div>{tip_html}</div>'
        f'<div class="mt-tile-value">{_e(value)}{unit_html}</div>{bar}'
        f'<div class="mt-tile-sub">{sub}</div>{foot}</div>'
    )


def occupancy_ring(pct: float, label: str, caption: str = "", size: int = 112) -> str:
    """Donut gauge for a capacity share, with tick marks at the 70/85/95% status levels."""
    import math

    pct = max(0.0, min(100.0, float(pct)))
    r, cx = 44.0, 56.0
    circ = 2 * math.pi * r
    color = occupancy_color(pct)
    ticks = ""
    for mark in (70, 85, 95):
        a = math.radians(mark * 3.6 - 90)
        x1, y1 = cx + (r - 6) * math.cos(a), cx + (r - 6) * math.sin(a)
        x2, y2 = cx + (r + 6) * math.cos(a), cx + (r + 6) * math.sin(a)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{theme.FAINT}" stroke-width="1.4"/>'
    return (
        f'<div class="mt-ring" role="img" aria-label="{_e(label)} {pct:.0f} percent">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 112 112">'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="{theme.SURFACE_3}" stroke-width="9"/>'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{circ * pct / 100:.1f} {circ:.1f}" transform="rotate(-90 {cx} {cx})"/>{ticks}'
        f'<text x="{cx}" y="{cx + 2}" text-anchor="middle" class="ring-v">{pct:.0f}%</text>'
        f'<text x="{cx}" y="{cx + 18}" text-anchor="middle" class="ring-l">in use</text></svg>'
        f'<div class="ring-cap"><b>{_e(label)}</b><span>{_e(caption)}</span></div></div>'
    )


def hero(
    status: str,
    headline: str,
    summary: str,
    facts: list[tuple[str, str, str]],
    tip: str | None = None,
    rings: list[tuple[str, float, str]] | None = None,
) -> str:
    """Status banner: status word, one plain sentence, optional capacity rings and key facts."""
    ink, bg = theme.STATUS.get(status, theme.STATUS["Normal"])
    facts_html = "".join(
        f'<div class="mt-fact"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div><div class="s">{_e(s)}</div></div>'
        for k, v, s in facts
    )
    rings_html = (
        '<div class="mt-hero-rings">' + "".join(occupancy_ring(p, l, c) for l, p, c in rings) + "</div>"
        if rings else ""
    )
    tip_html = info(tip) if tip else ""
    return (
        f'<div class="mt-hero" style="--accent:{ink}"><div class="mt-hero-main">'
        f'<div class="mt-hero-label">Hospital status {tip_html}{status_pill(status)}</div>'
        f'<div class="mt-hero-status"><span class="word">{_e(headline)}</span></div>'
        f'<div class="mt-hero-sum">{_e(summary)}</div></div>'
        f'{rings_html}<div class="mt-hero-facts">{facts_html}</div></div>'
    )


def sparkline(values: list[float], color: str | None = None, width: int = 120, height: int = 34) -> str:
    """Tiny trend line with an end marker. Returns '' when there is too little data."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 3:
        return ""
    color = color or theme.TEAL
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = [
        (3 + i * (width - 6) / (len(vals) - 1), height - 4 - (v - lo) / span * (height - 8))
        for i, v in enumerate(vals)
    ]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"3,{height} " + path + f" {width - 3},{height}"
    return (
        f'<svg class="mt-spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<polygon points="{area}" fill="{color}" fill-opacity=".10"/>'
        f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.6" fill="{color}"/></svg>'
    )


def stat_strip(items: list[dict]) -> str:
    """One continuous KPI band: label, value, context, trend and source badge per column."""
    cells = []
    for it in items:
        unit = f'<span class="u">{_e(it["unit"])}</span>' if it.get("unit") else ""
        bar = occupancy_bar(it["bar_pct"], 6) if it.get("bar_pct") is not None else ""
        tip = info(it["tip"]) if it.get("tip") else ""
        cells.append(
            f'<div class="mt-stat"><div class="top"><span class="k">{_e(it["label"])}</span>{tip}</div>'
            f'<div class="v">{_e(it["value"])}{unit}</div>{bar}'
            f'<div class="s">{it.get("sub", "")}</div>'
            f'<div class="foot">{it.get("spark", "")}{it.get("badge", "")}</div></div>'
        )
    return f'<div class="mt-strip">{"".join(cells)}</div>'


def nav_label(text: str) -> str:
    """Group heading in the navigation rail (a hairline when the rail is compact)."""
    return f'<div class="mt-nav-label"><span class="t">{_e(text)}</span></div>'


def rail_brand_html() -> str:
    """Logo mark for the rail; the name appears when the rail expands."""
    return f'<div class="mt-rail-brand">{logo_svg(36)}<span class="nm">MediTwin</span></div>'


def page_footer(data_through: str | None) -> str:
    """One quiet line under every workspace page: synthetic data and the non-diagnostic notice."""
    when = f" through {_e(data_through)}" if data_through else ""
    return (f'<div class="mt-page-footer"><span class="mt-dot"></span><span>Synthetic dataset{when}, not a live feed. '
            'Educational prototype. Not a clinical decision system; no diagnosis, treatment or triage.</span></div>')


def rail_status(data_through: str | None) -> str:
    """Small synthetic-data indicator for the rail. The detail is in its tooltip."""
    tip = "Synthetic dataset, not a live feed" + (f" (through {data_through})" if data_through else "")
    return (f'<div class="mt-rail-status" title="{_e(tip)}" role="note" aria-label="{_e(tip)}">'
            '<span class="mt-dot"></span><span class="t">Synthetic data</span></div>')


def glossary_html(terms: list[tuple[str, str]]) -> str:
    """Plain-language definitions as a compact list."""
    return "".join(
        f'<div class="mt-term"><div class="t">{_e(t)}</div><div class="d">{_e(d)}</div></div>' for t, d in terms
    )


def severity_tile(label: str, count: int) -> str:
    ink, _bg = theme.SEVERITY[label]
    glyph = theme.SEVERITY_GLYPH[label]
    zero = " zero" if count == 0 else ""
    return (
        f'<div class="mt-sev{zero}" style="--sev:{ink}"><div><div class="l">{_e(label)}</div>'
        f'<div class="n">{count}</div></div><div class="g" aria-hidden="true">{glyph}</div></div>'
    )


def meter(label: str, value_text: str, fill_pct: float, mark_pct: float | None, caption: str = "") -> str:
    """Headroom meter: today's value against the level where an alert would start."""
    fill = max(0.0, min(100.0, fill_pct))
    mark = ""
    if mark_pct is not None:
        mark = f'<i class="mark" style="left:{max(0.0, min(100.0, mark_pct)):.1f}%"></i>'
    cap = f'<div class="cap">{_e(caption)}</div>' if caption else ""
    return (
        f'<div class="mt-meter"><div class="row"><span>{_e(label)}</span><b>{_e(value_text)}</b></div>'
        f'<div class="track"><div class="fill" style="width:{fill:.1f}%"></div>{mark}</div>{cap}</div>'
    )


def all_clear(title: str, text: str) -> str:
    check = (
        '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">'
        f'<circle cx="17" cy="17" r="16" stroke="{theme.TEAL}" stroke-opacity=".5"/>'
        f'<path d="M10.5 17.5l4.2 4.2 8.8-9.2" stroke="{theme.TEAL}" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )
    return f'<div class="mt-clear">{check}<div><div class="t">{_e(title)}</div><div class="s">{_e(text)}</div></div></div>'


def callout(text: str, kind: str = "") -> str:
    """kind: '' (neutral), 'warn' or 'ok'. ``text`` is escaped."""
    return f'<div class="mt-callout {kind}">{_e(text)}</div>'


def empty_state(title: str, text: str = "") -> str:
    body = f'<div>{_e(text)}</div>' if text else ""
    return f'<div class="mt-empty"><div class="t">{_e(title)}</div>{body}</div>'


def unavailable(what: str, detail: str | None = None) -> str:
    msg = f"{what} is unavailable."
    if detail:
        msg += f" ({detail})"
    return callout(msg, "warn")


def provenance_legend() -> str:
    items = [
        ("observed", "stored in the hospital data"),
        ("ml", "predicted by the system"),
        ("rule", "worked out with a simple rule"),
    ]
    parts = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.4rem;margin-right:1.1rem">{badge(k)}'
        f'<span class="mt-note">{_e(t)}</span></span>'
        for k, t in items
    )
    return f'<div style="margin:.2rem 0 .7rem">{parts}</div>'


def alert_card(
    severity: str,
    title: str,
    category: str,
    scope: str,
    date: str,
    basis_key: str | None,
    metric_text: str,
    scope_label: str | None = None,
) -> str:
    """Alert row. ``basis_key=None`` omits the provenance badge; empty ``date`` is skipped."""
    ink, _ = theme.SEVERITY[severity]
    scope_text = scope_label or ("Predicted" if scope == "predicted" else "Current")
    meta = " · ".join(_e(p) for p in (category, scope_text, date, metric_text) if p)
    basis = badge(basis_key) if basis_key else ""
    return (
        f'<div class="mt-alert" style="--sev:{ink}"><div class="mt-alert-top">'
        f'<div><div class="mt-alert-title">{_e(title)}</div>'
        f'<div class="mt-alert-meta">{meta}</div></div>'
        f'<div style="display:flex;gap:.4rem;align-items:center">{basis}{severity_pill(severity)}</div>'
        f"</div></div>"
    )


def table(headers: list[tuple[str, bool]], rows: Iterable[list[str]], group_col: bool = False) -> str:
    """
    Generic table. ``headers`` is [(text, is_numeric)]. Cells in ``rows`` are
    pre-built HTML (callers escape). A row given as ``["__group__", text]`` is a
    section divider.
    """
    head = "".join(
        f'<th class="{"num" if num else ""}">{_e(text)}</th>' for text, num in headers
    )
    body = []
    for row in rows:
        if row and row[0] == "__group__":
            body.append(f'<tr><td class="mt-group" colspan="{len(headers)}">{_e(row[1])}</td></tr>')
            continue
        cells = "".join(
            f'<td class="{"num" if headers[i][1] else ""}">{cell}</td>' for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="mt-table-wrap"><table class="mt-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def value_text(value: float | None, decimals: int, unit: str) -> str:
    if value is None:
        return "—"
    return _e(f"{fmt_num(value, decimals)}{unit}")


# ----------------------------------------------------------------------
# Streamlit rendering helpers (only place this module touches Streamlit)
# ----------------------------------------------------------------------
def md(html: str) -> None:
    import streamlit as st

    st.markdown(html, unsafe_allow_html=True)


def plot(fig, key: str | None = None) -> None:
    import streamlit as st

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def card(name: str):
    """A raised surface container (styled via its key)."""
    import streamlit as st

    return st.container(key=f"card-{name}")
