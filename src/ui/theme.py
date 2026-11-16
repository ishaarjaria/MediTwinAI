"""
MediTwin design system: "Medical Sage" (light).

One place for colour tokens, severity/status palettes, the global CSS and the Plotly
layout. Components and charts import from here so the product looks like one thing.

Principles: calm off-white surfaces, deep sage as the single brand colour, a teal
secondary, warm neutrals for quiet emphasis, and status colours that are never the only
signal (severity also uses a shape and a word). Text colours were chosen to meet
WCAG AA (4.5:1) on every surface they are used on; tests enforce this.
"""

from __future__ import annotations

# --- Surfaces ----------------------------------------------------------
BG = "#F2F7F5"  # app background (soft sage mist)
SIDEBAR = "#FBFDFC"  # navigation rail
SURFACE = "#FFFFFF"  # cards and panels
SURFACE_2 = "#F6FAF8"  # inputs, hover, raised rows
SURFACE_3 = "#E4EEEA"  # tracks, selected, tooltips
BORDER = "#D3E1DB"
BORDER_SOFT = "#E2ECE8"
GRID = "#E3ECE8"
WARM = "#E8E2D6"  # warm neutral (dividers on the landing page, quiet fills)
WARM_SOFT = "#F6F3EC"

# --- Text --------------------------------------------------------------
INK = "#142B2A"
MUTED = "#566B68"  # secondary text (the suggested #637775 is 4.4:1 on BG; this is 5.2:1)
FAINT = "#5C716E"

# --- Brand and accents ---------------------------------------------------
PRIMARY = "#145A55"  # deep sage: primary actions, active navigation, headings
PRIMARY_HOVER = "#0F4642"
SAGE = "#88BFA3"  # soft sage: decorative fills, chart accents (never text)
SAGE_SOFT = "#E3F1EA"
TEAL = "#2E7D73"  # secondary teal: links, chart line, positive emphasis
TEAL_DEEP = PRIMARY
TEAL_SOFT = "#E3F1ED"
BLUE = "#3F6DA8"
BLUE_SOFT = "#E7EFF8"
AMBER = "#8F5A00"  # text-safe amber
AMBER_SOFT = "#FBF1DB"
CORAL = "#D9705A"
ORANGE = "#B04A12"
ORANGE_SOFT = "#FCEBDD"
RED = "#B42B2B"
RED_SOFT = "#FBE5E5"

# Severity (alerts) and capacity status share one palette: (ink, soft background).
SEVERITY = {
    "Low": (BLUE, BLUE_SOFT),
    "Medium": (AMBER, AMBER_SOFT),
    "High": (ORANGE, ORANGE_SOFT),
    "Critical": (RED, RED_SOFT),
}
# A shape per severity so colour is never the only cue.
SEVERITY_GLYPH = {"Low": "○", "Medium": "●", "High": "■", "Critical": "◆"}
STATUS = {
    "Normal": (PRIMARY, TEAL_SOFT),
    "Moderate": SEVERITY["Medium"],
    "High": SEVERITY["High"],
    "Critical": SEVERITY["Critical"],
}

# Where a number comes from, in plain words. The internal keys (observed / ml / rule)
# are unchanged so every caller and test of the data layer keeps working.
PROVENANCE = {
    "observed": ("Recorded", MUTED, "#E9EFED"),
    "ml": ("Expected", PRIMARY, TEAL_SOFT),
    "rule": ("Estimated", AMBER, AMBER_SOFT),
    "unavailable": ("Unavailable", FAINT, "#EEF2F1"),
}
# Short labels for tight spaces such as KPI tiles.
PROVENANCE_SHORT = {"observed": "Recorded", "ml": "Expected", "rule": "Estimated", "unavailable": "n/a"}

# Chart colours (ordered): teal, blue, warm gold, coral, grey, violet. Charts only need
# 3:1 against the white surface, so the gold here is brighter than the text-safe AMBER.
SERIES = [TEAL, BLUE, "#C48A17", CORAL, "#7C8F8B", "#7B6FB0"]
FONT_STACK = '"Inter","Segoe UI","Helvetica Neue",Arial,sans-serif'


def plotly_layout(height: int = 300) -> dict:
    """Shared Plotly layout: light, quiet grid, readable labels."""
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=INK),
        colorway=SERIES,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER,
            font=dict(family=FONT_STACK, size=12, color=INK),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, linecolor=BORDER, tickfont=dict(size=11, color=MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID, zeroline=False, tickfont=dict(size=11, color=MUTED),
            linecolor="rgba(0,0,0,0)",
        ),
    )


# ----------------------------------------------------------------------
# Global CSS. Tokens are written __NAME__ and substituted below.
# ----------------------------------------------------------------------
_CSS = """
html, body, .stApp { font-family: __FONT__; -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }
.stApp { background: __BG__; color: __INK__; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding:var(--page-top) 2rem 3.5rem 1.25rem; max-width:1320px; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: __BORDER__; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
h1, h2, h3, h4 { color: __INK__; }
p, li { color: __INK__; }
[data-testid="stMain"] .stMarkdown p, [data-testid="stMain"] .stMarkdown li { font-size:.875rem; line-height:1.55; }
a { color: __TEAL__; }

/* Icon-font fallback: never show ligature names as raw text */
[data-testid="stIconMaterial"] { display:inline-block; width:1.25rem; max-width:1.25rem; overflow:hidden; white-space:nowrap; }

/* ---------- Navigation rail (the restyled Streamlit sidebar) ----------
   A slim floating rail. Compact it shows icons only; hovering it, or moving keyboard focus into
   it, reveals the labels as an overlay, so the page never shifts. "Keep menu open" (a server-side
   toggle) pins it open and the page makes room. Pure CSS: no scripts. --lab is 0 (icons) or 1 (labels). */
:root { --rail-w:84px; --rail-top:1.75rem; --page-top:calc(var(--rail-top) + 15px); } /* 15 = rail border 1 + padding 10 + (44 - 36) / 2 above the logo */
/* The invisible element that injects this stylesheet must not add a gap above the page header. */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) { display:none; }
section[data-testid="stSidebar"] { background:transparent !important; border:none !important; z-index:120; }
[data-testid="stSidebarContent"] { --lab:0; background:__SIDEBAR__; }
[data-testid="stSidebarUserContent"] { padding:0 !important; }
[data-testid="stSidebarUserContent"] > div { height:100%; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap:.1rem; }
.mt-rail-brand { display:flex; align-items:center; gap:.7rem; height:44px; margin:0 0 1.5rem; overflow:hidden; white-space:nowrap; }
.mt-rail-brand svg { flex:none; }
.mt-rail-brand .nm { color:__PRIMARY__; font-weight:680; font-size:1.02rem; letter-spacing:-.01em; opacity:var(--lab); transition:opacity .15s; }
.mt-nav-label { position:relative; margin-top:0; padding:14px 0 6px 7px; color:__FAINT__; font-size:.66rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; line-height:14px; overflow:hidden; white-space:nowrap; margin-bottom:1rem; }
.mt-nav-label .t { opacity:var(--lab); transition:opacity .15s; }
.mt-nav-label::after { content:""; position:absolute; left:.35rem; right:.35rem; top:calc(50% + 4px); border-top:1px solid __BORDER__; opacity:calc(1 - var(--lab)); transition:opacity .15s; }
section[data-testid="stSidebar"] [class*="st-key-nav-"],
section[data-testid="stSidebar"] [class*="st-key-rail-"] { margin:0; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button,
section[data-testid="stSidebar"] [class*="st-key-rail-"] button { position:relative; width:100%; height:40px; min-height:0; justify-content:flex-start; gap:calc(var(--lab) * .7rem); padding:0 .5rem 0 7px;
  background:transparent !important; border:none !important; box-shadow:none; border-radius:11px; color:__MUTED__; font-weight:500; font-size:.86rem; overflow:hidden; white-space:nowrap;
  transition:background .16s, color .16s; }
section[data-testid="stSidebar"] button > div { justify-content:flex-start; width:100%; gap:calc(var(--lab) * .7rem); }
section[data-testid="stSidebar"] button p { margin:0; max-width:calc(var(--lab) * 160px); opacity:var(--lab); overflow:hidden; white-space:nowrap; color:inherit; transition:opacity .16s, max-width .22s; }
[data-testid="stSidebar"] [data-testid="stIconMaterial"] { display:inline-flex; align-items:center; justify-content:center; width:1.5rem; max-width:1.5rem; min-width:1.5rem; height:1.5rem; font-variation-settings:'FILL' 0,'wght' 300,'GRAD' 0,'opsz' 24; font-size:1.3rem; line-height:1.5rem; flex:none; color:inherit; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button:hover,
section[data-testid="stSidebar"] [class*="st-key-rail-"] button:hover { background:__SURFACE_2__ !important; color:__PRIMARY__; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button:active, section[data-testid="stSidebar"] [class*="st-key-rail-"] button:active { background:__TEAL_SOFT__ !important; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button[kind="primary"],
section[data-testid="stSidebar"] [class*="st-key-nav-"] button[data-testid="stBaseButton-primary"] { background:__TEAL_SOFT__ !important; color:__PRIMARY__; font-weight:620; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button[data-testid="stBaseButton-primary"]::before { content:""; position:absolute; left:0; top:9px; bottom:9px; width:3px; border-radius:0 3px 3px 0; background:__PRIMARY__; }
section[data-testid="stSidebar"] [class*="st-key-nav-"] button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24; }
.st-key-rail-bottom [data-testid="stButton"], .st-key-rail-bottom button { width:100% !important; min-width:0 !important; }
.st-key-rail-bottom { position:absolute; left:8px; bottom:10px; width:calc(100% - 16px) !important; height:auto !important; min-height:0 !important; padding-top:.5rem; border-top:1px solid __BORDER_SOFT__; background:__SIDEBAR__; }
@media (max-height:720px) { .st-key-rail-bottom { position:static; margin-top:.8rem; } }
.mt-rail-status { display:flex; align-items:center; gap:.8rem; height:32px; margin-bottom:1rem; padding-left:15px; color:__MUTED__; font-size:.74rem; overflow:hidden; white-space:nowrap; }
.mt-rail-status .mt-dot { margin-left:.05rem; }
.mt-rail-status .t { opacity:var(--lab); transition:opacity .15s; }
.mt-user { display:flex; align-items:center; gap:.7rem; height:44px; margin:.2rem 0 1rem; padding-left:4px; overflow:hidden; white-space:nowrap; }
.mt-user .av { width:30px; height:30px; flex:none; }
.mt-user > div:last-child { opacity:var(--lab); transition:opacity .15s; }
.mt-user .nm { max-width:130px; overflow:hidden; text-overflow:ellipsis; }
.mt-page-footer { display:flex; align-items:center; gap:.6rem; margin-top:2.4rem; padding-top:1rem; border-top:1px solid __BORDER_SOFT__; color:__FAINT__; font-size:.72rem; line-height:1.5; }
.mt-dot { width:8px; height:8px; border-radius:50%; background:#C48A17; flex:none; }
.mt-dot.teal { background:__TEAL__; }
.mt-user .av { border-radius:50%; background:__PRIMARY__; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-size:.76rem; font-weight:600; }
.mt-user .nm { color:__INK__; font-size:.82rem; font-weight:600; line-height:1.2; }
.mt-user .rl { color:__MUTED__; font-size:.7rem; }
.mt-status-block { margin:.2rem 0 0; padding:.8rem .9rem; border:1px solid __BORDER_SOFT__; border-radius:12px; }
.mt-status-row { display:flex; align-items:center; gap:.5rem; color:__INK__; font-size:.8rem; font-weight:600; }
.mt-status-line { color:__MUTED__; font-size:.76rem; line-height:1.5; margin-top:.3rem; }
.mt-status-line b { color:__INK__; font-weight:600; }
.mt-side-disclaimer { color:__MUTED__; font-size:.74rem; line-height:1.55; padding:.8rem 0 0; }
.mt-term { padding:.55rem 0; border-bottom:1px solid __BORDER_SOFT__; }
.mt-term:last-child { border-bottom:none; }
.mt-term .t { color:__INK__; font-weight:600; font-size:.82rem; }
.mt-term .d { color:__MUTED__; font-size:.78rem; line-height:1.5; margin-top:.1rem; }
:focus-visible { outline:2px solid __PRIMARY__; outline-offset:2px; border-radius:6px; }
section[data-testid="stSidebar"] button:focus-visible { outline:2px solid __PRIMARY__ !important; outline-offset:-2px; box-shadow:none !important; }
[data-testid="stMain"] button:focus-visible, [data-testid="stMain"] [role="tab"]:focus-visible, [data-testid="stMain"] a:focus-visible,
[data-testid="stMain"] input:focus-visible, [data-testid="stMain"] [role="slider"]:focus-visible { outline:2px solid __PRIMARY__ !important; outline-offset:2px; }

@media (min-width:769px) {
  section[data-testid="stSidebar"] { width:var(--rail-w) !important; min-width:var(--rail-w) !important; max-width:var(--rail-w) !important; transform:none !important; overflow:visible !important; }
  [data-testid="stSidebarHeader"] { display:none !important; }
  [data-testid="stSidebarContent"] { position:fixed; left:16px; top:var(--rail-top); bottom:var(--rail-top); height:auto; width:56px; padding:10px 8px 10px; display:flex; flex-direction:column;
    border:1px solid __BORDER_SOFT__; border-radius:18px; box-shadow:0 1px 2px rgba(20,43,42,.04), 0 10px 28px rgba(20,43,42,.06); overflow-x:hidden; overflow-y:auto; z-index:200;
    transition:width .22s cubic-bezier(.2,.7,.2,1), box-shadow .22s; scrollbar-width:none; }
  [data-testid="stSidebarContent"]::-webkit-scrollbar { display:none; }
  [data-testid="stSidebarUserContent"] { flex:1; min-height:0; display:flex; flex-direction:column; }
  [data-testid="stSidebarUserContent"] > div { flex:1; }
  [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { min-height:100%; }
  section[data-testid="stSidebar"]:hover [data-testid="stSidebarContent"],
  section[data-testid="stSidebar"]:has(:focus-visible) [data-testid="stSidebarContent"] { width:228px; --lab:1; box-shadow:0 2px 4px rgba(20,43,42,.05), 0 18px 44px rgba(20,43,42,.14); }
}
@media (max-width:768px) {
  [data-testid="stSidebarContent"] { --lab:1; }
  .st-key-rail-pin { display:none; }
  .block-container { padding-top:3.6rem; }
}

/* ---------- Page header ---------- */
.mt-header { display:flex; justify-content:space-between; align-items:flex-start; gap:1.2rem; margin:0 0 1.35rem; }
.mt-eyebrow { display:flex; align-items:center; line-height:1.2; gap:.45rem; color:__TEAL__; font-size:.6875rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.35rem; }
.mt-eyebrow::before { content:""; width:14px; height:2px; background:__SAGE__; border-radius:2px; }
.mt-title { font-size:1.625rem; font-weight:650; letter-spacing:-.02em; color:__INK__; line-height:1.2; }
.mt-subtitle { color:__MUTED__; font-size:.875rem; margin-top:.3rem; max-width:64ch; line-height:1.55; }
.mt-meta { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; justify-content:flex-end; margin-top:1.15rem; min-height:31px; } /* centred on the title line: eyebrow (~19 px) + half the 31 px title */
.mt-chip-date { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:8px; background:__WARM_SOFT__; border:1px solid __WARM__; color:__MUTED__; font-size:.76rem; font-weight:500; }

/* ---------- Surfaces ---------- */
[class*="st-key-card-"] { background:__SURFACE__; border:1px solid __BORDER_SOFT__; border-radius:14px;
  padding:1.1rem 1.25rem 1.15rem; box-shadow:0 1px 2px rgba(20,43,42,.04), 0 6px 20px rgba(20,43,42,.05); }
.mt-section-row { display:flex; justify-content:space-between; align-items:baseline; gap:.6rem; }
.mt-section { display:flex; align-items:center; gap:.55rem; font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:__MUTED__; margin:0 0 .65rem; }
.mt-section::before { content:""; width:3px; height:12px; border-radius:2px; background:__SAGE__; }
.mt-note { color:__MUTED__; font-size:.78rem; line-height:1.55; }
.mt-how { color:__MUTED__; font-size:.775rem; line-height:1.55; margin-top:.6rem; padding:.1rem 0 .1rem .8rem; border-left:2px solid __SAGE__; }
.mt-how b { color:__INK__; font-weight:600; }

/* ---------- Hero (status band) ---------- */
.mt-hero { display:flex; gap:2rem; justify-content:space-between; align-items:center; background:linear-gradient(100deg, #F0F8F4 0%, __SURFACE__ 48%);
  border:1px solid __BORDER_SOFT__; border-radius:16px; padding:1.5rem 1.7rem; position:relative; overflow:hidden;
  box-shadow:0 1px 2px rgba(20,43,42,.04), 0 10px 30px rgba(20,43,42,.06); }
.mt-hero::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
.mt-hero, .mt-hero-main { min-width:0; }
.mt-hero-main { flex:1 1 auto; }
.mt-hero-label { display:flex; align-items:center; gap:.6rem; color:__MUTED__; font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; }
.mt-hero-status { margin:.5rem 0 .45rem; }
.mt-hero-status .word { font-size:1.7rem; font-weight:650; letter-spacing:-.028em; color:__PRIMARY__; line-height:1.2; }
.mt-hero-sum { color:__MUTED__; font-size:.875rem; line-height:1.6; max-width:60ch; }
.mt-hero-sum b { color:__INK__; font-weight:600; }
.mt-hero-rings { display:flex; gap:1.6rem; flex:none; align-items:center; padding:0 1.6rem; border-left:1px solid __BORDER_SOFT__; border-right:1px solid __BORDER_SOFT__; }
.mt-ring { display:flex; flex-direction:column; align-items:center; gap:.45rem; }
.mt-ring .ring-v { fill:__INK__; font-size:21px; font-weight:680; font-family:inherit; }
.mt-ring .ring-l { fill:__FAINT__; font-size:9px; letter-spacing:.12em; text-transform:uppercase; font-family:inherit; }
.mt-ring .ring-cap { display:flex; flex-direction:column; align-items:center; text-align:center; }
.mt-ring .ring-cap b { color:__INK__; font-size:.8rem; font-weight:600; }
.mt-ring .ring-cap span { color:__MUTED__; font-size:.72rem; }
.mt-hero-facts { display:flex; flex-direction:column; gap:.6rem; flex:none; }
.mt-fact { min-width:170px; padding:.1rem .9rem; border-left:2px solid __BORDER__; }
.mt-hero-facts .mt-fact { overflow-wrap:anywhere; }
.mt-fact .k { color:__MUTED__; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; }
.mt-fact .v { color:__INK__; font-size:1.05rem; font-weight:650; margin:.25rem 0 .1rem; letter-spacing:-.01em; }
.mt-fact .s { color:__MUTED__; font-size:.74rem; line-height:1.4; }

/* ---------- KPI tiles and strip ---------- */
.mt-tile { background:__SURFACE__; border:1px solid __BORDER_SOFT__; border-radius:14px; padding:1rem 1.1rem .95rem; min-height:182px; height:100%;
  box-shadow:0 1px 2px rgba(20,43,42,.04); transition:border-color .15s, box-shadow .15s; }
.mt-tile-top { display:flex; justify-content:space-between; align-items:center; gap:.4rem; }
.mt-tile-label { font-size:.7rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:__MUTED__; }
.mt-tile-value { font-size:1.7rem; font-weight:650; letter-spacing:-.025em; color:__INK__; margin:.4rem 0 .1rem; line-height:1.1; }
.mt-tile-unit { font-size:.85rem; font-weight:500; color:__FAINT__; margin-left:.35rem; letter-spacing:0; }
.mt-tile-sub { font-size:.78rem; color:__MUTED__; margin-top:.4rem; min-height:1.1rem; line-height:1.45; }
.mt-tile-foot { margin-top:.6rem; }
.mt-bar { height:7px; border-radius:7px; background:__SURFACE_3__; overflow:hidden; margin-top:.55rem; position:relative; }
.mt-bar > span { display:block; height:100%; border-radius:7px; background:__TEAL__; }
.mt-strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); background:linear-gradient(180deg, #F5FAF7 0%, __SURFACE__ 65%); border:1px solid __BORDER_SOFT__; border-radius:16px; overflow:hidden;
  box-shadow:0 1px 2px rgba(20,43,42,.04), 0 8px 24px rgba(20,43,42,.05); }
.mt-stat { padding:1.05rem 1.2rem 1rem; border-right:1px solid __BORDER_SOFT__; display:flex; flex-direction:column; min-width:0; transition:background .15s; }
.mt-stat:last-child { border-right:none; }
.mt-stat .top { display:flex; justify-content:space-between; align-items:center; }
.mt-stat .k { color:__MUTED__; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; }
.mt-stat .v { font-size:1.75rem; font-weight:650; letter-spacing:-.03em; line-height:1.1; margin:.45rem 0 .1rem; font-variant-numeric:tabular-nums; color:__PRIMARY__; }
.mt-stat .v .u { font-size:.85rem; font-weight:500; color:__FAINT__; margin-left:.35rem; letter-spacing:0; }
.mt-stat .s { color:__MUTED__; font-size:.76rem; line-height:1.5; margin-top:.35rem; flex:1 1 auto; }
.mt-stat .foot { display:flex; flex-direction:column; align-items:flex-start; gap:.45rem; margin-top:.7rem; }
.mt-stat .mt-bar { height:5px; }
.mt-stat .foot .mt-badge { flex:none; }
.mt-spark { flex:1 1 auto; min-width:0; max-width:150px; width:100%; height:34px; }

/* ---------- Tooltip ---------- */
.mt-info { position:relative; display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; border-radius:50%;
  border:1px solid __FAINT__; color:__MUTED__; font-size:.65rem; font-weight:700; cursor:help; flex:none; font-style:normal; }
.mt-info:hover, .mt-info:focus { color:__PRIMARY__; border-color:__PRIMARY__; outline:none; }
.mt-info::after { content:attr(data-tip); position:absolute; right:-6px; top:22px; width:230px; padding:.6rem .75rem; background:__INK__; color:#F4F8F6;
  border-radius:10px; font-size:.74rem; font-weight:400; line-height:1.5; letter-spacing:0; text-transform:none; text-align:left;
  white-space:normal; z-index:1000; box-shadow:0 12px 28px rgba(20,43,42,.25); opacity:0; visibility:hidden; transform:translateY(-3px); transition:opacity .12s, transform .12s; pointer-events:none; }
.mt-info:hover::after, .mt-info:focus::after { opacity:1; visibility:visible; transform:translateY(0); }

/* ---------- Pills / badges ---------- */
.mt-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.2rem .65rem; border-radius:8px; font-size:.74rem; font-weight:600; letter-spacing:.01em; white-space:nowrap;
  border:1px solid color-mix(in srgb, currentColor 22%, transparent); }
.mt-pill::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }
.mt-pill.sev::before { display:none; }
.mt-pill .g { font-size:.7rem; line-height:1; }
.mt-badge { display:inline-block; padding:.1rem .5rem; border-radius:6px; font-size:.68rem; font-weight:600; letter-spacing:.02em; white-space:nowrap; border:1px solid color-mix(in srgb, currentColor 18%, transparent); }
.mt-chip { display:inline-flex; align-items:baseline; gap:.4rem; padding:.3rem .7rem; border-radius:8px; font-size:.78rem; font-weight:600; border:1px solid color-mix(in srgb, currentColor 22%, transparent); }
.mt-chip b { font-size:1.05rem; font-weight:700; }

/* ---------- Severity summary tiles ---------- */
.mt-sev { background:__SURFACE__; border:1px solid __BORDER_SOFT__; border-top:3px solid var(--sev); border-radius:12px; padding:.8rem 1rem; display:flex; justify-content:space-between; align-items:center; }
.mt-sev .n { font-size:1.7rem; font-weight:680; color:__INK__; letter-spacing:-.02em; line-height:1; }
.mt-sev .l { color:var(--sev); font-size:.74rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.mt-sev .g { color:var(--sev); font-size:1.1rem; }
.mt-sev:not(.zero) { background:color-mix(in srgb, var(--sev) 8%, #FFFFFF); }
.mt-sev.zero { opacity:.75; }

/* ---------- Tables ---------- */
.mt-table-wrap { width:100%; overflow-x:auto; }
.mt-table { width:100%; border-collapse:collapse; font-size:.8125rem; border:none !important; }
.mt-table, .mt-table th, .mt-table td { border-left:none !important; border-right:none !important; }
.mt-table th { text-align:left; font-size:.66rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:__MUTED__; padding:.4rem .65rem; border-bottom:1px solid __BORDER__ !important; white-space:nowrap; background:transparent; }
.mt-table th.num, .mt-table td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.mt-table td { padding:.62rem .65rem; border-bottom:1px solid __BORDER_SOFT__ !important; color:__INK__; vertical-align:middle; background:transparent; }
.mt-table td:first-child { white-space:normal; }
.mt-table tr:last-child td { border-bottom:none !important; }
.mt-table tr:hover td { background:#F2F9F6; }
.mt-table td.num .mt-badge { display:inline-block; min-width:5.6rem; text-align:center; margin-left:.55rem; }
.mt-delta { font-weight:600; font-variant-numeric:tabular-nums; white-space:nowrap; }
.mt-delta.good { color:__PRIMARY__; } .mt-delta.bad { color:__ORANGE__; } .mt-delta.neutral { color:__MUTED__; }
.mt-group { font-size:.66rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:__FAINT__; padding-top:1rem !important; }

/* ---------- Alerts and callouts ---------- */
.mt-alert { background:color-mix(in srgb, var(--sev) 5%, #FFFFFF); border:1px solid __BORDER_SOFT__; border-left:4px solid var(--sev); border-radius:12px; padding:.85rem 1.05rem; margin-bottom:.4rem; }
.mt-alert-top { display:flex; justify-content:space-between; gap:.8rem; align-items:flex-start; }
.mt-alert-title { font-weight:600; font-size:.95rem; color:__INK__; }
.mt-alert-meta { color:__MUTED__; font-size:.78rem; margin-top:.28rem; }
.mt-callout { border-radius:12px; padding:.8rem 1rem; font-size:.85rem; line-height:1.55; border:1px solid __WARM__; background:__WARM_SOFT__; color:__MUTED__; }
.mt-callout.warn { background:__AMBER_SOFT__; border-color:#E9D5A3; color:#6B4300; }
.mt-callout.ok { background:__TEAL_SOFT__; border-color:#BFDCD3; color:__PRIMARY__; }
.mt-empty { text-align:center; padding:2.2rem 1rem; color:__MUTED__; border:1px dashed __BORDER__; border-radius:14px; }
.mt-empty .t { font-weight:600; color:__INK__; font-size:1rem; margin-bottom:.25rem; }
.mt-clear { display:flex; align-items:center; gap:1rem; padding:1.1rem 1.2rem; background:__TEAL_SOFT__; border:1px solid #BFDCD3; border-radius:14px; }
.mt-clear .t { color:__INK__; font-weight:650; font-size:1.02rem; }
.mt-clear .s { color:__MUTED__; font-size:.84rem; line-height:1.5; margin-top:.15rem; }
.mt-meter { margin:.1rem 0 .85rem; }
.mt-meter .row { display:flex; justify-content:space-between; align-items:baseline; font-size:.82rem; color:__MUTED__; margin-bottom:.35rem; }
.mt-meter .row b { color:__INK__; font-weight:600; font-variant-numeric:tabular-nums; }
.mt-meter .track { position:relative; height:8px; background:__SURFACE_3__; border-radius:8px; overflow:visible; }
.mt-meter .fill { height:100%; border-radius:8px; background:__TEAL__; }
.mt-meter .mark { position:absolute; top:-3px; bottom:-3px; width:2px; background:__AMBER__; border-radius:2px; }
.mt-meter .cap { color:__FAINT__; font-size:.72rem; margin-top:.3rem; }
.mt-list { margin:.35rem 0; color:__MUTED__; font-size:.88rem; line-height:1.55; }
.mt-list::before { content:"•"; color:__TEAL__; margin-right:.55rem; }
.mt-subhead { color:__INK__; font-size:.8rem; font-weight:600; margin:.7rem 0 .15rem; padding-bottom:.3rem; border-bottom:1px solid __BORDER_SOFT__; }
[class*="st-key-card-twin-current"], [class*="st-key-card-twin-predicted"] { min-height:520px; }
[class*="st-key-card-risk-result"], [class*="st-key-card-risk-probs"] { min-height:300px; }
[class*="st-key-card-"] div[data-testid="stExpander"] { margin-top:.8rem; }
.mt-chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:.7rem 0 .2rem; }
.mt-dchip { display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .75rem; border-radius:8px; background:__WARM_SOFT__; border:1px solid __BORDER_SOFT__; font-size:.8rem; color:__MUTED__; }
.mt-dchip b { color:__INK__; font-weight:600; }

/* ---------- Native widgets ---------- */
div[data-testid="stExpander"] { border:1px solid __BORDER_SOFT__; border-radius:12px; background:__SURFACE__; }
div[data-testid="stExpander"] summary { font-size:.86rem; font-weight:500; color:__INK__; }
div[data-testid="stExpander"] summary { transition:background .18s; }
div[data-testid="stExpander"] summary:hover { background:#F1F8F5; }
div[data-testid="stTabs"] button[role="tab"] { transition:color .18s; }
div[data-testid="stTabs"] button[role="tab"]:hover { color:__PRIMARY__; }
button[role="tab"] { font-weight:500; color:__MUTED__; padding:.55rem 1rem; font-size:.88rem; }
button[role="tab"][aria-selected="true"] { color:__PRIMARY__; font-weight:650; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background:__BORDER__; }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:__PRIMARY__; }
label, [data-testid="stWidgetLabel"] p { color:__MUTED__ !important; font-size:.8rem !important; font-weight:500; }
div[data-testid="stSegmentedControl"] label p, .stPills p { font-size:.82rem !important; }
.stSlider [data-baseweb="slider"] { padding-top:.1rem; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color:__FAINT__; }
.stSlider [role="slider"] { box-shadow:0 0 0 4px rgba(20,90,85,.14); }
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div, .stSelectbox [role="combobox"] { background:__SURFACE__ !important; border-color:__BORDER__ !important; border-radius:10px !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:__PRIMARY__ !important; box-shadow:0 0 0 1px __PRIMARY__; }
input { color:__INK__ !important; font-variant-numeric:tabular-nums; }
div[data-testid="stSegmentedControl"] { background:__SURFACE__; border-radius:12px; padding:3px; border:1px solid __BORDER__; }
div[data-testid="stSegmentedControl"] button { border-radius:9px !important; border:none !important; }
[data-testid="stMain"] div[data-testid="stButton"] button { border-radius:10px; border:1px solid __BORDER__; background:__SURFACE__; color:__INK__; min-height:2.5rem; transition:border-color .18s, background .18s, box-shadow .18s, transform .18s, color .18s; }
[data-testid="stMain"] div[data-testid="stButton"] button:hover { border-color:__PRIMARY__; color:__PRIMARY__; background:#F6FBF8; box-shadow:0 3px 10px rgba(20,43,42,.09); transform:translateY(-1px); }
[data-testid="stMain"] div[data-testid="stButton"] button:active { transform:translateY(0); box-shadow:none; background:__TEAL_SOFT__; }
[data-testid="stMain"] div[data-testid="stButton"] button[kind="primary"] { background:__PRIMARY__; color:#FFFFFF; border-color:__PRIMARY__; }
[data-testid="stMain"] div[data-testid="stButton"] button[kind="primary"]:hover { background:__PRIMARY_HOVER__; color:#FFFFFF; box-shadow:0 5px 14px rgba(20,90,85,.28); }
[data-testid="stMain"] div[data-testid="stButton"] button[kind="primary"]:active, [data-testid="stFormSubmitButton"] button:active { background:#0B3532; transform:translateY(0); box-shadow:none; }
[data-testid="stFormSubmitButton"] button { border-radius:10px; min-height:2.5rem; background:__PRIMARY__; color:#FFFFFF; font-weight:600; border:none; transition:background .18s, box-shadow .18s, transform .18s; }
[data-testid="stFormSubmitButton"] button:hover { background:__PRIMARY_HOVER__; color:#FFFFFF; box-shadow:0 5px 14px rgba(20,90,85,.28); transform:translateY(-1px); }
[data-testid="stMain"] button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p { color:#FFFFFF !important; }
.stAlert { border-radius:12px; }
hr { border-color:__BORDER_SOFT__; }

/* ---------- Responsive: laptops, tablets and a basic phone fallback ---------- */
@media (max-width:1280px) {
  .mt-hero { flex-wrap:wrap; }
  .mt-hero-main { flex:1 1 100%; }
  .mt-hero-rings { border-left:none; padding-left:0; border-right:none; padding-right:0; }
  .mt-hero-facts { flex-direction:row; flex-wrap:wrap; flex:1 1 auto; }
  .mt-fact { flex:1 1 150px; min-width:0; }
  .mt-strip { grid-template-columns:repeat(5,minmax(0,1fr)); }
  .mt-stat { padding:.9rem .8rem; }
  .mt-stat .v { font-size:1.7rem; }
  .mt-stat .v .u { display:block; margin:.1rem 0 0; }
}
@media (max-width:1000px) {
  .mt-strip { grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); }
  .mt-stat { border-bottom:1px solid __BORDER_SOFT__; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  [data-testid="stMain"] [data-testid="stColumn"] { min-width:100% !important; flex:1 1 100% !important; }
}
@media (max-width:820px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .mt-header { flex-direction:column; align-items:flex-start; }
  .mt-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .mt-stat:last-child:nth-child(odd) { grid-column:1 / -1; }
  .mt-hero-rings { border-right:none; padding-right:0; }
  .mt-title { font-size:1.45rem; }
}
@media (max-width:520px) {
  .mt-strip { grid-template-columns:1fr; }
  .mt-hero-rings { flex-wrap:wrap; justify-content:center; width:100%; }
  .mt-hero { padding:1.1rem; }
  .mt-hero-status .word { font-size:1.5rem; }
  .mt-fact { min-width:0; flex:1 1 100%; }
}
@media (prefers-reduced-motion:reduce) {
  *, *::before, *::after { transition:none !important; animation:none !important; }
}
"""

_TOKENS = {
    "FONT": FONT_STACK, "BG": BG, "SIDEBAR": SIDEBAR, "SURFACE": SURFACE, "SURFACE_2": SURFACE_2,
    "SURFACE_3": SURFACE_3, "BORDER": BORDER, "BORDER_SOFT": BORDER_SOFT, "WARM": WARM,
    "WARM_SOFT": WARM_SOFT, "INK": INK, "MUTED": MUTED, "FAINT": FAINT, "PRIMARY": PRIMARY,
    "PRIMARY_HOVER": PRIMARY_HOVER, "SAGE": SAGE, "SAGE_SOFT": SAGE_SOFT, "TEAL": TEAL, "TEAL_SOFT": TEAL_SOFT,
    "AMBER": AMBER, "AMBER_SOFT": AMBER_SOFT, "ORANGE": ORANGE,
}


def substitute(css: str) -> str:
    """Replace ``__NAME__`` placeholders with token values (longest names first)."""
    for name in sorted(_TOKENS, key=len, reverse=True):
        css = css.replace(f"__{name}__", _TOKENS[name])
    return css


CSS = substitute(_CSS)

# "Keep menu open": the rail stays expanded and the page makes room for it.
RAIL_PINNED_CSS = substitute("""
@media (min-width:769px) {
  :root { --rail-w:260px; }
  [data-testid="stSidebarContent"] { width:228px !important; --lab:1 !important; }
}
""")


# ----------------------------------------------------------------------
# Public pages (landing and sign-in). Injected only on those pages.
# ----------------------------------------------------------------------
_LANDING_CSS = """
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarNav"], header[data-testid="stHeader"] { display:none !important; }
.stApp { background:linear-gradient(180deg, #F8F9F4 0%, __BG__ 680px); }
.block-container { max-width:1180px; padding:1.1rem 2rem 2.5rem; position:relative; isolation:isolate; }
html { scroll-behavior:smooth; }
[id^="lp-"] { scroll-margin-top:1rem; }

/* Network background behind the hero */
.lp-bg { position:absolute; left:-2rem; right:-2rem; top:-1.2rem; height:760px; z-index:-1; overflow:hidden; pointer-events:none;
  -webkit-mask-image:linear-gradient(#000 30%, transparent); mask-image:linear-gradient(#000 30%, transparent); }
.lp-net { width:100%; height:100%; }
.lp-net-line { stroke:__SAGE__; stroke-opacity:.16; stroke-width:1; }
.lp-net-flow { stroke-dasharray:3 9; animation:lp-dash 14s linear infinite; }
.lp-net-dot { fill:__SAGE__; opacity:.3; animation:lp-pulse 7s ease-in-out infinite; }

/* Top navigation */
.st-key-pubnav { padding:.2rem 0 1rem; }
.lp-nav { display:flex; align-items:center; gap:2rem; height:2.5rem; margin-bottom:1rem; } /* the 1rem cancels Streamlit's -1rem markdown margin so this row centres with the buttons */
.lp-brand { display:flex; align-items:center; gap:.6rem; white-space:nowrap; color:__PRIMARY__; font-weight:700; font-size:1.15rem; letter-spacing:-.01em; text-decoration:none; }
.lp-brand svg { flex:none; }
.lp-links { display:flex; gap:1.7rem; margin-left:auto; }
.lp-links a { color:__MUTED__; font-size:.9rem; font-weight:500; text-decoration:none; transition:color .15s; }
.lp-links a:hover { color:__PRIMARY__; }
[data-testid="stMain"] .st-key-pubnav button { height:2.5rem; min-height:2.5rem; padding:0 1.2rem; font-weight:600; white-space:nowrap; border-radius:10px; }
.st-key-pubnav [data-testid="stHorizontalBlock"] { align-items:center !important; }
.st-key-pub_signin button { background:__SAGE_SOFT__ !important; border:1px solid #CDE5D8 !important; color:__PRIMARY__ !important; box-shadow:none !important; }
.st-key-pub_signin button:hover { background:#D6EBDF !important; border-color:#B9DCC9 !important; }

.st-key-pub_signin button p { color:__PRIMARY__ !important; }
.lp-btn-ghost, .lp-btn-ghost:hover, .lp-btn-ghost:visited { color:__PRIMARY__ !important; text-decoration:none !important; }
.lp-links a, .lp-footer nav a, .lp-brand { text-decoration:none !important; }

/* Hero */
.lp-eyebrow { display:inline-flex; align-items:center; gap:.5rem; padding:.3rem .7rem; border-radius:8px; background:__SAGE_SOFT__; color:__PRIMARY__; font-size:.78rem; font-weight:600; letter-spacing:.01em; }
.lp-eyebrow::before { content:""; width:6px; height:6px; border-radius:50%; background:__TEAL__; }
.lp-h1 { font-size:3rem; line-height:1.08; letter-spacing:-.032em; font-weight:650; color:__INK__; margin:1.2rem 0 1.1rem; max-width:30rem; }
.lp-h1 em { font-style:normal; color:__PRIMARY__; }
.lp-lead { color:__MUTED__; font-size:1.05rem; line-height:1.65; max-width:30rem; }
.lp-trust { display:flex; flex-wrap:wrap; gap:.5rem 1.1rem; color:__MUTED__; font-size:.8rem; margin-top:1.3rem; }
.lp-trust span::before { content:"✓"; color:__TEAL__; font-weight:700; margin-right:.45rem; }
.lp-hospital { width:100%; height:auto; display:block; overflow:visible; }
.st-key-cta-primary button, .st-key-cta-final button { min-height:3rem; padding:0 1.7rem; font-weight:600; }
.lp-btn-ghost { display:flex; align-items:center; justify-content:center; box-sizing:border-box; width:100%; height:3rem; padding:0 1.6rem; margin-bottom:1rem; border:1px solid __BORDER__; border-radius:10px;
  background:__SURFACE__; color:__PRIMARY__; font-weight:600; font-size:.9375rem; line-height:1; text-decoration:none; transition:border-color .15s, box-shadow .15s, transform .15s; }
[data-testid="stMain"] .st-key-cta-primary button, [data-testid="stMain"] .st-key-preview_demo button, [data-testid="stMain"] .st-key-lp-final button { height:3rem !important; min-height:3rem !important; padding:0 1.6rem; border-radius:10px; font-weight:600; }
[data-testid="stMain"] .st-key-cta-primary button p, [data-testid="stMain"] .st-key-preview_demo button p, [data-testid="stMain"] .st-key-lp-final button p { font-size:.9375rem !important; font-weight:600 !important; line-height:1; }
.lp-btn-ghost:hover { border-color:__PRIMARY__; background:#F6FBF8; box-shadow:0 4px 14px rgba(20,43,42,.1); transform:translateY(-1px); }
.lp-btn-ghost:active { transform:translateY(0); box-shadow:none; background:__TEAL_SOFT__; }

/* Value strip */
.lp-values { display:grid; grid-template-columns:repeat(3,1fr); margin-top:2.6rem; border-top:1px solid __BORDER__; border-bottom:1px solid __BORDER__; }
.lp-value { padding:1.4rem 1.6rem 1.4rem 0; display:flex; gap:1rem; align-items:flex-start; }
.lp-value + .lp-value { padding-left:1.6rem; border-left:1px solid __BORDER__; }
.lp-value .ic { color:__TEAL__; flex:none; margin-top:.1rem; }
.lp-value b { display:block; color:__INK__; font-size:.98rem; font-weight:650; margin-bottom:.2rem; }
.lp-value span { color:__MUTED__; font-size:.88rem; line-height:1.55; }

/* Sections */
.lp-section { padding:4.5rem 0 .5rem; }
.lp-kicker { color:__TEAL__; font-size:.76rem; font-weight:650; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.7rem; }
.lp-h2 { font-size:1.9rem; line-height:1.18; letter-spacing:-.026em; font-weight:650; color:__INK__; margin:0 0 .8rem; max-width:34rem; }
.lp-sub { color:__MUTED__; font-size:.98rem; line-height:1.65; max-width:36rem; }
.lp-features { display:grid; grid-template-columns:repeat(3,1fr); gap:2.6rem 2.8rem; margin-top:2.6rem; }
.lp-feature { border-top:1px solid __BORDER__; padding-top:1.3rem; }
.lp-feature .ic { width:46px; height:46px; border-radius:11px; background:__SAGE_SOFT__; color:__PRIMARY__; display:flex; align-items:center; justify-content:center; transition:background .2s, color .2s; }
.lp-feature:hover .ic { background:#D3E9DE; color:__PRIMARY_HOVER__; }
.lp-feature h3 { font-size:1.08rem; font-weight:650; margin:1rem 0 .4rem; color:__INK__; letter-spacing:-.01em; }
.lp-feature p { color:__MUTED__; font-size:.92rem; line-height:1.62; margin:0; }
.lp-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:2.6rem; margin-top:2.6rem; position:relative; }
.lp-step { position:relative; }
.lp-step .n { width:42px; height:42px; border-radius:11px; background:__PRIMARY__; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.05rem; }
.lp-step:not(:last-child)::after { content:""; position:absolute; top:21px; left:58px; right:-1.4rem; border-top:1px dashed __SAGE__; }
.lp-step h3 { font-size:1.12rem; font-weight:650; margin:1.1rem 0 .4rem; color:__INK__; }
.lp-step p { color:__MUTED__; font-size:.94rem; line-height:1.62; margin:0; max-width:19rem; }
.st-key-lp-howband { background:#EAF4EF; border-radius:20px; padding:0 2.6rem 3rem; margin-top:4.5rem; }
.st-key-lp-howband .lp-section { padding-top:3rem; }
.st-key-lp-band { background:__WARM_SOFT__; border-radius:20px; padding:3rem 2.6rem 2.4rem; margin-top:4.5rem; }
.lp-frame { position:relative; border:1px solid __BORDER__; border-radius:14px; overflow:hidden; background:#FFFFFF; box-shadow:0 18px 44px rgba(20,43,42,.12); margin-top:.6rem; }
.lp-frame-bar { display:flex; align-items:center; gap:6px; padding:.65rem .95rem; background:__SURFACE_2__; border-bottom:1px solid __BORDER_SOFT__; }
.lp-frame-bar i { width:9px; height:9px; border-radius:50%; background:__BORDER__; display:block; }
.lp-frame-bar span { margin-left:.9rem; color:__FAINT__; font-size:.74rem; }
.lp-shot { position:relative; width:100%; aspect-ratio:1200 / 750; background:#FFFFFF; }
.lp-shot img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:block; }.lp-frame::after { content:""; position:absolute; left:0; right:0; bottom:0; height:84px; pointer-events:none; background:linear-gradient(rgba(255,255,255,0), #FFFFFF); }
.lp-caption { color:__MUTED__; font-size:.82rem; margin:.9rem 0 1.4rem; }
.lp-about { display:grid; grid-template-columns:1.1fr .9fr; gap:3.5rem; align-items:start; }
.lp-fact { display:flex; gap:1rem; padding:1.05rem 0; border-top:1px solid __BORDER__; }
.lp-fact:last-child { border-bottom:1px solid __BORDER__; }
.lp-fact .ic { color:__TEAL__; flex:none; margin-top:.1rem; }
.lp-fact b { display:block; color:__INK__; font-weight:650; font-size:.96rem; }
.lp-fact span { color:__MUTED__; font-size:.88rem; line-height:1.55; }
.lp-disclaimer { display:flex; gap:1rem; background:__WARM_SOFT__; border:1px solid __WARM__; border-radius:14px; padding:1.15rem 1.35rem; margin-top:2.2rem; color:__INK__; font-size:.9rem; line-height:1.65; }
.lp-disclaimer .ic { color:__PRIMARY__; flex:none; margin-top:.1rem; }
.lp-disclaimer b { font-weight:650; }
.st-key-lp-final { background:__PRIMARY__; border-radius:20px; padding:3.2rem 2rem 3rem; margin-top:4.5rem; text-align:center; }
.st-key-lp-final .lp-h2 { color:#FFFFFF; margin:0 auto .7rem; text-align:center; max-width:none; }
.st-key-lp-final .lp-sub { color:#D4E8DF; margin:0 auto 1.6rem; text-align:center; }
[data-testid="stMain"] .st-key-lp-final button { min-height:3rem; padding:0 1.7rem; font-weight:600; border-radius:10px; }
[data-testid="stMain"] .st-key-lp-final button[kind="primary"] { background:#FFFFFF !important; border:1px solid #FFFFFF !important; }
[data-testid="stMain"] .st-key-lp-final button[kind="primary"]:hover { background:__SAGE_SOFT__ !important; }
[data-testid="stMain"] .st-key-lp-final button[kind="secondary"] { background:transparent !important; border:1px solid rgba(255,255,255,.6) !important; }
[data-testid="stMain"] .st-key-lp-final button[kind="secondary"]:hover { background:rgba(255,255,255,.12) !important; border-color:#FFFFFF !important; }
[data-testid="stMain"] .st-key-lp-final button[kind="primary"] p { color:__PRIMARY__ !important; }
[data-testid="stMain"] .st-key-lp-final button[kind="secondary"] p { color:#FFFFFF !important; }
.lp-footer { display:flex; flex-wrap:wrap; gap:1.2rem 3rem; justify-content:space-between; align-items:flex-start; margin-top:3.5rem; padding-top:1.8rem; border-top:1px solid __BORDER__; }
.lp-footer .tag { color:__MUTED__; font-size:.86rem; line-height:1.6; max-width:26rem; margin-top:.7rem; }
.lp-footer nav { display:flex; gap:1.5rem; flex-wrap:wrap; }
.lp-footer nav a { color:__MUTED__; font-size:.86rem; text-decoration:none; }
.lp-footer nav a:hover { color:__PRIMARY__; }
.lp-legal { color:__FAINT__; font-size:.78rem; line-height:1.6; margin-top:1.4rem; }

/* Sign-in */
.st-key-si-wrap { max-width:1000px; margin:0 auto; padding-top:4vh; }
.si-panel { background:__PRIMARY__; color:#FFFFFF; border-radius:18px; padding:2rem 2rem 1.8rem; min-height:610px; position:relative; overflow:hidden; display:flex; flex-direction:column; }
.st-key-card-signin { min-height:610px; padding:1.6rem 1.8rem 1.3rem; }
[data-testid="stMain"] div[data-testid="stButton"] button[kind="tertiary"] { border:none; background:transparent; min-height:0; padding:0; color:__MUTED__; font-size:.8rem; box-shadow:none; }
[data-testid="stMain"] div[data-testid="stButton"] button[kind="tertiary"]:hover { color:__PRIMARY__; background:transparent; box-shadow:none; }
.si-panel .foot { margin-top:auto; padding-top:2rem; color:#BFD9CE; font-size:.76rem; line-height:1.55; }
.si-panel .lp-brand { color:#FFFFFF; }
.si-panel h2 { color:#FFFFFF; font-size:1.65rem; line-height:1.2; letter-spacing:-.025em; margin:3rem 0 .9rem; font-weight:650; }
.si-panel p { color:#D4E8DF; font-size:.96rem; line-height:1.65; }
.si-panel ul { list-style:none; padding:0; margin:1.6rem 0 0; }
.si-panel li { color:#EAF4EF; font-size:.92rem; padding:.5rem 0; display:flex; gap:.7rem; }
.si-panel li::before { content:"✓"; color:__SAGE__; font-weight:700; }
.si-panel .lp-net { position:absolute; inset:0; opacity:.22; }
.si-panel > *:not(.lp-net) { position:relative; }
.si-notice { background:__AMBER_SOFT__; border:1px solid #E9D5A3; color:#6B4300; border-radius:10px; padding:.75rem .9rem; font-size:.82rem; line-height:1.55; margin:.3rem 0 1rem; }
.si-limits { color:__FAINT__; font-size:.76rem; line-height:1.55; margin-top:1rem; padding-bottom:.5rem; }
.si-or { display:flex; align-items:center; gap:.8rem; color:__FAINT__; font-size:.78rem; margin:1rem 0 .6rem; }
.si-or::before, .si-or::after { content:""; flex:1; border-top:1px solid __BORDER__; }

@keyframes lp-dash { to { stroke-dashoffset:-120; } }
@keyframes lp-pulse { 0%,100% { opacity:.15; } 50% { opacity:.4; } }
@keyframes lp-ring { 0% { transform:scale(.8); opacity:.7; } 100% { transform:scale(2); opacity:0; } }
@keyframes lp-float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-5px); } }
.lp-flow { stroke:__TEAL__; stroke-width:1.4; stroke-dasharray:4 6; opacity:.75; animation:lp-dash 7s linear infinite; }
.lp-drop { stroke:__PRIMARY__; stroke-width:1.2; stroke-dasharray:2 4; opacity:.55; }
.lp-ring { transform-box:fill-box; transform-origin:center; animation:lp-ring 3.4s ease-out infinite; }
.lp-chip { animation:lp-float 6s ease-in-out infinite; filter:drop-shadow(0 6px 10px rgba(20,43,42,.12)); }

@media (max-width:1000px) {
  .lp-values, .lp-features, .lp-steps { grid-template-columns:1fr; }
  .lp-features { grid-template-columns:repeat(2,1fr); }
  .lp-hospital { max-width:560px; margin:0 auto; }
  .lp-value + .lp-value { padding-left:0; border-left:none; border-top:1px solid __BORDER__; }
  .lp-step:not(:last-child)::after { display:none; }
  .lp-about { grid-template-columns:1fr; gap:2rem; }
  .lp-links { display:none; }
  .st-key-pubnav [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:.4rem !important; align-items:center; }
  .st-key-pubnav [data-testid="stColumn"] { min-width:0 !important; flex:0 0 auto !important; width:auto !important; }
  .st-key-pubnav [data-testid="stColumn"]:first-child { flex:1 1 auto !important; }
  .st-key-pubnav button { padding:0 .85rem; white-space:nowrap; }
  .lp-h1 { font-size:2.7rem; }
  .si-panel { min-height:0; }
}
@media (max-width:640px) {
  .lp-features { grid-template-columns:1fr; }
}
@media (max-width:520px) {
  h1.lp-h1 { font-size:1.85rem; }
  .lp-brand { font-size:1.02rem; gap:.45rem; }
  [data-testid="stMain"] .st-key-pubnav button { padding:0 .75rem; font-size:.85rem; }
  .st-key-pubnav [data-testid="stHorizontalBlock"] { gap:.3rem !important; }
}
@media (max-width:820px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .st-key-lp-howband { padding:0 1.1rem 2rem; border-radius:14px; }
  .lp-h1 { font-size:2.25rem; }
  .lp-h2 { font-size:1.65rem; }
  .st-key-lp-band { padding:2rem 1.1rem 1.6rem; border-radius:14px; }
  .st-key-lp-final { padding:2.2rem 1.1rem; }
  .lp-bg { left:-1rem; right:-1rem; }
}
@media (prefers-reduced-motion:reduce) {
  .lp-net-flow, .lp-net-dot, .lp-flow, .lp-ring, .lp-chip { animation:none !important; }
  html { scroll-behavior:auto; }
}
"""

LANDING_CSS = substitute(_LANDING_CSS)
