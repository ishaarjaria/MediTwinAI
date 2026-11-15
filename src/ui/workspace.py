"""
The authenticated dashboard shell: the navigation rail, the user block and the active view.

The rail is Streamlit's sidebar restyled by ``theme.CSS`` into a slim floating navigation rail
(icons when compact, labels on hover or keyboard focus, or pinned open with "Keep menu open").
All calculation stays in the backend packages; views live in ``src/ui/views``.
"""

from __future__ import annotations

import streamlit as st

from src.ui import auth, backend, components as ui, theme
from src.ui.data import GLOSSARY_TERMS, format_date
from src.ui.views import DEFAULT_VIEW, NAV_GROUPS, VIEWS, get_renderer, go_to, normalize_view


def active_view() -> str:
    """Session view, initialised from ``?view=`` so views can be deep-linked."""
    if "view" not in st.session_state:
        st.session_state["view"] = normalize_view(st.query_params.get("view", DEFAULT_VIEW))
    return st.session_state["view"]


def _do_sign_out() -> None:
    auth.sign_out(st.session_state)


def _toggle_pin() -> None:
    st.session_state["rail_pinned"] = not st.session_state.get("rail_pinned", False)


@st.dialog("Plain-language guide")
def _guide_dialog(data_through: str | None) -> None:
    ui.md(ui.glossary_html(GLOSSARY_TERMS))
    ui.md(ui.sidebar_footer(data_through))


def _rail(active: str, data_through: str | None) -> None:
    labels = {view_id: (label, icon) for view_id, label, icon in VIEWS}
    user = auth.current_user(st.session_state)
    pinned = bool(st.session_state.get("rail_pinned", False))
    with st.sidebar:
        ui.md(ui.rail_brand_html())
        for group, view_ids in NAV_GROUPS:
            ui.md(ui.nav_label(group))
            for view_id in view_ids:
                label, icon = labels[view_id]
                st.button(
                    label, icon=icon, key=f"nav-{view_id}", on_click=go_to, args=(view_id,),
                    type="primary" if view_id == active else "secondary", width="stretch",
                )
        with st.container(key="rail-bottom"):
            ui.md(ui.rail_status(data_through))
            if st.button("Plain-language guide", icon=":material/help:", key="rail-guide", width="stretch"):
                _guide_dialog(data_through)
            st.button(
                "Auto-hide menu" if pinned else "Keep menu open",
                icon=":material/left_panel_close:" if pinned else ":material/left_panel_open:",
                key="rail-pin", on_click=_toggle_pin, width="stretch",
            )
            if user:
                ui.md(ui.user_block(user.name, user.role, user.initials))
                st.button("Sign out", icon=":material/logout:", key="signout", on_click=_do_sign_out, width="stretch")


def render() -> None:
    """Render the workspace. The caller has already checked that someone is signed in."""
    pinned = bool(st.session_state.get("rail_pinned", False))
    st.markdown(f"<style>{theme.CSS}{theme.RAIL_PINNED_CSS if pinned else ''}</style>", unsafe_allow_html=True)
    if backend.is_ready():
        warm = backend.warm_up()  # already loaded: no spinner on page changes
    else:
        with st.spinner("Loading models…"):
            warm = backend.warm_up()  # loads the prediction service and twin once per process
            backend.prefetch()  # fills the data caches once per process

    ops = backend.operations()
    through = format_date(ops.value["date"].max()) if ops.ok else None
    active = active_view()
    _rail(active, through)
    st.query_params["view"] = active

    if not warm.ok:
        ui.md(ui.unavailable("Models", warm.error))
        ui.md(ui.callout("Train the models first: python -m src.models.train_all"))
    try:
        get_renderer(active)()
    except Exception as exc:  # a broken view must not take the whole app down
        ui.md(ui.unavailable("This view", f"{type(exc).__name__}: {exc}"))
    ui.md(ui.page_footer(through))
