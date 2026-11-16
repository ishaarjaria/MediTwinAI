"""Shared pieces of the public pages: styles, top navigation, demo sign-in and footer."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.ui import auth, router, theme
from src.ui import components as ui

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "static"
NAV_LINKS = [("Platform", "lp-platform"), ("Features", "lp-features"), ("How It Works", "lp-how"), ("About", "lp-about")]


def inject_styles() -> None:
    st.markdown(f"<style>{theme.CSS}{theme.LANDING_CSS}</style>", unsafe_allow_html=True)


def demo_sign_in() -> None:
    """Start a demo session and open the workspace."""
    auth.sign_in(st.session_state, auth.DEMO_USER)
    router.go(router.WORKSPACE)


def nav() -> None:
    """Top bar: brand and in-page links on the left, Sign in and Get Started on the right."""
    links = "".join(f'<a href="#{target}">{label}</a>' for label, target in NAV_LINKS)
    signed_in = auth.is_authenticated(st.session_state)
    with st.container(key="pubnav"):
        brand, actions_a, actions_b = st.columns([7, 1.3, 1.6], vertical_alignment="center")
        with brand:
            ui.md(f'<div class="lp-nav"><span class="lp-brand">{ui.logo_svg(34)}MediTwin AI</span>'
                  f'<div class="lp-links">{links}</div></div>')
        with actions_a:
            if st.button("Open workspace" if signed_in else "Sign in", key="pub_signin", width="stretch"):
                router.go(router.WORKSPACE if signed_in else router.SIGNIN)
        with actions_b:
            if not signed_in and st.button("Get Started", key="pub_start", type="primary", width="stretch"):
                router.go(router.SIGNIN)


def footer() -> None:
    links = "".join(f'<a href="#{target}">{label}</a>' for label, target in NAV_LINKS)
    ui.md(
        '<div class="lp-footer"><div>'
        f'<span class="lp-brand">{ui.logo_svg(30)}MediTwin AI</span>'
        '<div class="tag">An educational prototype for hospital operations planning, built on synthetic data.</div></div>'
        f'<nav>{links}</nav></div>'
        '<div class="lp-legal">© 2026 MediTwin AI academic project. Not a medical device and not a clinical decision system. '
        'No real patient data is used and nothing here is connected to a real hospital.</div>'
    )


def preview_src(name: str) -> str | None:
    """Self-contained ``data:`` URI of a bundled preview image, or ``None`` if the file is missing.

    The image is embedded in the page instead of linked through ``/app/static/``, so it does not
    depend on static serving, the URL prefix or the host's file layout.
    """
    return _data_uri(name)


@lru_cache(maxsize=None)
def _data_uri(name: str) -> str | None:
    path = STATIC / name
    if not path.is_file():
        return None
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower())
    if mime is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
