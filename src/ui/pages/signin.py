"""Sign in, create a demo account, or continue as the demo user (educational demo access only)."""

from __future__ import annotations

import streamlit as st

from src.ui import auth, router
from src.ui import components as ui
from src.ui import landing_assets as art
from src.ui.pages import common


def _panel() -> str:
    return (
        '<div class="si-panel">' + art.network_background(520, 560, nodes=20, seed=5) +
        f'<span class="lp-brand">{ui.logo_svg(34)}MediTwin AI</span>'
        '<h2>Welcome to your operations workspace</h2>'
        '<p>See today, expect the days ahead, and plan before deciding.</p>'
        '<ul><li>Plain-language views of the hospital</li>'
        '<li>Alerts that say what needs attention</li>'
        '<li>Everything runs on synthetic data</li></ul>'
        '<div class="foot">Educational prototype. Not a clinical decision system.</div></div>'
    )


def _finish(user: auth.User) -> None:
    auth.sign_in(st.session_state, user)
    router.go(router.WORKSPACE)


def _sign_in_form() -> None:
    with st.form("signin_form", border=False):
        email = st.text_input("E-mail", key="si_email", placeholder="name@example.com")
        password = st.text_input("Password", key="si_password", type="password")
        submitted = st.form_submit_button("Sign in", key="si_submit", width="stretch")
    if submitted:
        try:
            _finish(auth.get_store().authenticate(email, password))
        except auth.AuthError as exc:
            st.error(str(exc))


def _create_account_form() -> None:
    with st.form("signup_form", border=False):
        name = st.text_input("Your name", key="su_name")
        email = st.text_input("E-mail", key="su_email", placeholder="name@example.com")
        password = st.text_input("Choose a demo password", key="su_password", type="password",
                                 help="Use any throwaway text. Do not reuse a real password.")
        confirm = st.text_input("Repeat the password", key="su_confirm", type="password")
        submitted = st.form_submit_button("Create account", key="su_submit", width="stretch")
    if submitted:
        try:
            if password != confirm:
                raise auth.AuthError("The two passwords do not match.")
            _finish(auth.get_store().create(name, email, password))
        except auth.AuthError as exc:
            st.error(str(exc))


def render() -> None:
    common.inject_styles()
    if auth.is_authenticated(st.session_state):
        router.go(router.WORKSPACE)

    with st.container(key="si-wrap"):
        if st.button("← Back to home", key="si_back", type="tertiary"):
            router.go(router.LANDING)
        left, right = st.columns(2, gap="medium")
        with left:
            ui.md(_panel())
        with right:
            with st.container(key="card-signin"):
                ui.md('<div class="mt-title" style="font-size:1.4rem;margin:.2rem 0 .5rem">Sign in to MediTwin AI</div>'
                      f'<div class="si-notice">{ui.esc(auth.DEMO_NOTICE)}</div>')
                tab_in, tab_new = st.tabs(["Sign in", "Create account"])
                with tab_in:
                    _sign_in_form()
                with tab_new:
                    _create_account_form()
                ui.md('<div class="si-or">or</div>')
                if st.button("Continue as demo user", key="demo_signin", width="stretch"):
                    _finish(auth.DEMO_USER)
                ui.md(f'<div class="si-limits">{ui.esc(auth.DEMO_LIMITS)}</div>')
