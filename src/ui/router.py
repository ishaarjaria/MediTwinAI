"""
Page routing for the product journey: landing -> sign in -> workspace.

Uses Streamlit's own ``st.navigation`` / ``st.Page`` (hidden navigation, real URLs):

    /          landing page (public)
    /signin    sign in, create account, continue as demo user (public)
    /workspace the dashboard workspace (requires a session user)

Old deep links such as ``/?view=twin`` keep working: the landing route forwards them to the
workspace (or to sign-in first) and remembers the requested view.
"""

from __future__ import annotations

import streamlit as st

from src.ui import auth
from src.ui.views import normalize_view

LANDING, SIGNIN, WORKSPACE = "landing", "signin", "workspace"
PATHS = {LANDING: "", SIGNIN: "signin", WORKSPACE: "workspace"}  # URL paths; the landing page is the default route


def landing_page() -> None:
    requested = st.query_params.get("view")
    if requested is not None:  # legacy deep link: /?view=twin (unknown ids fall back to the overview)
        st.session_state.setdefault("view", normalize_view(requested))  # never overrides a later navigation
        workspace_page()  # rendered in place (no page switch), so widget state survives reruns
        return
    from src.ui.pages import landing

    landing.render()


def signin_page() -> None:
    from src.ui.pages import signin

    signin.render()


def workspace_page() -> None:
    if not auth.is_authenticated(st.session_state):
        go(SIGNIN)  # protected route: signed-out visitors are sent to sign-in
    from src.ui import workspace

    workspace.render()


def build_pages() -> dict[str, st.Page]:
    return {
        LANDING: st.Page(landing_page, title="MediTwin AI · Hospital operations, simplified", default=True),
        SIGNIN: st.Page(signin_page, title="Sign in · MediTwin AI", url_path=PATHS[SIGNIN]),
        WORKSPACE: st.Page(workspace_page, title="MediTwin AI · Workspace", url_path=PATHS[WORKSPACE]),
    }


def go(route: str) -> None:
    """Navigate to a route by name (raises Streamlit's stop-and-rerun signal)."""
    st.switch_page(build_pages()[route])


def run() -> None:
    pages = build_pages()
    st.navigation(list(pages.values()), position="hidden").run()
