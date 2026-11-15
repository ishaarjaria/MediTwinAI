"""Routing, route protection, the sign-in page and sign-out, with Streamlit's AppTest."""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import auth, router
from src.ui.views import NAV_GROUPS, VIEW_IDS, VIEWS

APP = str(ROOT / "app.py")

# The sign-in page calls router.go(...), which really navigates. The harness records the
# destination instead, and always restores the real function so other tests are unaffected.
HARNESS = """
import streamlit as st
from src.ui import router
_real = router.go
router.go = lambda route: st.session_state.__setitem__("went_to", route)
try:
    from src.ui.pages import signin
    signin.render()
finally:
    router.go = _real
"""


def page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def nav_buttons(at) -> list:
    return [b for b in at.sidebar.button if str(b.key).startswith("nav-")]


def app(view: str | None = None, session: dict | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=240)
    if view is not None:
        at.query_params["view"] = view
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def signin_page(session: dict | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(HARNESS, default_timeout=60)
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def unique_email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


class TestRoutes(unittest.TestCase):
    def test_three_routes_with_the_planned_urls(self):
        pages = router.build_pages()
        self.assertEqual(list(pages), ["landing", "signin", "workspace"])
        self.assertEqual(router.PATHS, {"landing": "", "signin": "signin", "workspace": "workspace"})

    def test_static_serving_does_not_collide_with_a_route(self):
        # Streamlit serves static files under /app/static; no page may use the "app" path.
        for path in router.PATHS.values():
            self.assertNotEqual(path, "app")


class TestRouteProtection(unittest.TestCase):
    def test_signed_out_visitor_sees_the_public_landing_page(self):
        at = app()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Hospital Operations", page_text(at))
        self.assertEqual(nav_buttons(at), [])  # no dashboard navigation

    def test_deep_link_while_signed_out_goes_to_sign_in_not_the_dashboard(self):
        for view in VIEW_IDS:
            with self.subTest(view=view):
                at = app(view)
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(nav_buttons(at), [])
                self.assertIn("Continue as demo user", [b.label for b in at.button])
                self.assertNotIn('class="mt-header"', page_text(at))  # no dashboard page header

    def test_deep_link_while_signed_in_opens_that_view(self):
        for view in ("twin", "alerts", "analytics"):
            with self.subTest(view=view):
                at = app(view, {auth.SESSION_KEY: auth.DEMO_USER})
                self.assertEqual(at.session_state["view"], view)
                self.assertEqual(len(nav_buttons(at)), 7)

    def test_sign_in_from_a_deep_link_lands_on_the_requested_view(self):
        at = app("twin")
        at.button(key="demo_signin").click().run()
        self.assertEqual(len(at.exception), 0)
        # AppTest does not follow st.switch_page the way a browser does (it reruns the default page
        # and the post-click session is not a faithful model), so only the remembered view is
        # asserted here. The real sign-in -> workspace page change is exercised in the browser journey.
        self.assertEqual(at.session_state["view"], "twin")

    def test_unknown_view_falls_back_to_overview_when_signed_in(self):
        at = app("nonsense", {auth.SESSION_KEY: auth.DEMO_USER})
        self.assertEqual(at.session_state["view"], "overview")


class TestSignOut(unittest.TestCase):
    def test_sign_out_ends_the_session_and_blocks_the_dashboard(self):
        at = app("twin", {auth.SESSION_KEY: auth.DEMO_USER})
        self.assertIn("signout", [b.key for b in at.sidebar.button])
        at.sidebar.button(key="signout").click().run()
        self.assertNotIn(auth.SESSION_KEY, at.session_state)
        self.assertEqual(nav_buttons(at), [])
        self.assertIn("Continue as demo user", [b.label for b in at.button])

    def test_sidebar_shows_who_is_signed_in(self):
        at = app("overview", {auth.SESSION_KEY: auth.DEMO_USER})
        text = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("Demo Administrator", text)
        self.assertIn("Demo access", text)


class TestSignInPage(unittest.TestCase):
    def test_page_states_that_it_is_a_demo(self):
        at = signin_page()
        text = page_text(at)
        self.assertIn("Educational demo access", text)
        self.assertIn("not production authentication", text)
        self.assertEqual(len(at.exception), 0)

    def test_continue_as_demo_user(self):
        at = signin_page()
        at.button(key="demo_signin").click().run()
        self.assertEqual(at.session_state[auth.SESSION_KEY], auth.DEMO_USER)
        self.assertEqual(at.session_state["went_to"], "workspace")

    def test_create_account_then_sign_in_again(self):
        email = unique_email()
        at = signin_page()
        at.text_input(key="su_name").input("Ada Lovelace")
        at.text_input(key="su_email").input(email)
        at.text_input(key="su_password").input("throwaway1")
        at.text_input(key="su_confirm").input("throwaway1")
        at.button(key="su_submit").click().run()
        self.assertEqual(at.session_state[auth.SESSION_KEY].email, email)
        self.assertEqual(at.session_state["went_to"], "workspace")

        again = signin_page()  # a fresh session, same server-side store
        again.text_input(key="si_email").input(email.upper())
        again.text_input(key="si_password").input("throwaway1")
        again.button(key="si_submit").click().run()
        self.assertEqual(again.session_state[auth.SESSION_KEY].name, "Ada Lovelace")

    def test_wrong_password_shows_an_error_and_does_not_sign_in(self):
        email = unique_email()
        auth.get_store  # the store is reachable without a running page
        at = signin_page()
        at.text_input(key="su_name").input("Ada")
        at.text_input(key="su_email").input(email)
        at.text_input(key="su_password").input("throwaway1")
        at.text_input(key="su_confirm").input("throwaway1")
        at.button(key="su_submit").click().run()
        bad = signin_page()
        bad.text_input(key="si_email").input(email)
        bad.text_input(key="si_password").input("not-it-at-all")
        bad.button(key="si_submit").click().run()
        self.assertNotIn(auth.SESSION_KEY, bad.session_state)
        self.assertTrue(any("do not match" in e.value for e in bad.error))

    def test_mismatched_passwords_are_rejected(self):
        at = signin_page()
        at.text_input(key="su_name").input("Ada")
        at.text_input(key="su_email").input(unique_email())
        at.text_input(key="su_password").input("throwaway1")
        at.text_input(key="su_confirm").input("different1")
        at.button(key="su_submit").click().run()
        self.assertNotIn(auth.SESSION_KEY, at.session_state)
        self.assertTrue(any("do not match" in e.value for e in at.error))

    def test_invalid_email_and_short_password_are_explained(self):
        at = signin_page()
        at.text_input(key="su_name").input("Ada")
        at.text_input(key="su_email").input("nope")
        at.text_input(key="su_password").input("abc")
        at.text_input(key="su_confirm").input("abc")
        at.button(key="su_submit").click().run()
        self.assertNotIn(auth.SESSION_KEY, at.session_state)
        self.assertTrue(at.error)

    def test_already_signed_in_visitors_are_forwarded(self):
        at = signin_page({auth.SESSION_KEY: auth.DEMO_USER})
        self.assertEqual(at.session_state["went_to"], "workspace")

    def test_password_fields_are_masked(self):
        at = signin_page()
        for key in ("si_password", "su_password", "su_confirm"):
            self.assertEqual(at.text_input(key=key).proto.type, 1, key)  # PASSWORD


class TestNavigationStructure(unittest.TestCase):
    def test_seven_views_in_three_groups_with_plain_labels(self):
        self.assertEqual([g for g, _ in NAV_GROUPS], ["Today", "Look ahead", "Plan"])
        self.assertEqual(dict(NAV_GROUPS)["Today"], ["overview", "alerts", "analytics"])
        self.assertEqual(dict(NAV_GROUPS)["Look ahead"], ["predictions", "patient_risk"])
        self.assertEqual(dict(NAV_GROUPS)["Plan"], ["twin", "whatif"])
        labels = [label for _id, label, _icon in VIEWS]
        self.assertEqual(sorted(labels), sorted(["Overview", "Alerts", "Trends", "Forecasts", "Patient Risk", "Hospital Twin", "What-If"]))


if __name__ == "__main__":
    unittest.main()
