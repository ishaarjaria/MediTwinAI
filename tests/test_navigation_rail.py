"""The custom navigation rail, the typography scale and the page-change performance design."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import auth, charts, components as ui, theme
from src.ui.views import NAV_GROUPS, VIEW_IDS, VIEWS

APP = str(ROOT / "app.py")


def workspace(view: str = "overview", session: dict | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["view"] = view
    at.session_state[auth.SESSION_KEY] = auth.DEMO_USER
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def rem_px(css: str, selector: str, prop: str = "font-size") -> float:
    """Value of ``prop`` (in rem, returned in px at the 16px root) from the first rule for ``selector``."""
    for block in re.finditer(re.escape(selector) + r"\s*\{([^}]*)\}", css):
        m = re.search(prop + r":\s*([\d.]+)rem", block.group(1))
        if m:
            return float(m.group(1)) * 16
    raise AssertionError(f"{selector} {{ {prop} }} not found")


class TestRailStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = workspace("twin")

    def test_seven_navigation_buttons_in_the_three_groups(self):
        keys = [b.key for b in self.at.sidebar.button if str(b.key).startswith("nav-")]
        expected = [f"nav-{v}" for _g, ids in NAV_GROUPS for v in ids]
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), 7)

    def test_group_headings_and_brand_are_in_the_rail(self):
        html = " ".join(m.value for m in self.at.sidebar.markdown)
        for group in ("Today", "Look ahead", "Plan"):
            self.assertIn(f'<span class="t">{group}</span>', html)
        self.assertIn("mt-rail-brand", html)
        self.assertIn("Synthetic data", html)  # the small status indicator

    def test_bottom_controls_exist(self):
        keys = {b.key for b in self.at.sidebar.button}
        self.assertTrue({"rail-guide", "rail-pin", "signout"} <= keys)

    def test_only_the_active_page_is_marked_primary(self):
        kinds = {b.key: b.proto.type for b in self.at.sidebar.button if str(b.key).startswith("nav-")}
        self.assertEqual(kinds["nav-twin"], "primary")
        self.assertTrue(all(v == "secondary" for k, v in kinds.items() if k != "nav-twin"))

    def test_navigation_items_are_real_buttons_so_the_keyboard_works(self):
        # Every nav item is a Streamlit button (focusable, Enter/Space activate). No custom JavaScript.
        source = (ROOT / "src/ui/workspace.py").read_text(encoding="utf8")
        for banned in ("components.html", "st.html", "<script", "javascript:"):
            self.assertNotIn(banned, source)
        self.assertNotIn("<script", theme.CSS)

    def test_no_native_tooltips_clutter_the_expanded_labels(self):
        for b in self.at.sidebar.button:
            if str(b.key).startswith("nav-"):
                self.assertFalse(b.help, b.key)

    def test_icons_are_a_consistent_set(self):
        icons = [icon for _i, _l, icon in VIEWS]
        self.assertEqual(len(set(icons)), 7)
        self.assertTrue(all(i.startswith(":material/") for i in icons))

    def test_user_block_and_sign_out(self):
        html = " ".join(m.value for m in self.at.sidebar.markdown)
        self.assertIn("Demo Administrator", html)
        self.assertIn("Demo access", html)


class TestRailPinning(unittest.TestCase):
    def test_keep_menu_open_toggles_and_changes_the_layout_css(self):
        at = workspace("overview")
        css = " ".join(m.value for m in at.markdown)
        self.assertNotIn("--rail-w:260px", css)
        self.assertEqual(at.sidebar.button(key="rail-pin").label, "Keep menu open")
        at.sidebar.button(key="rail-pin").click().run()
        self.assertTrue(at.session_state["rail_pinned"])
        self.assertEqual(at.sidebar.button(key="rail-pin").label, "Auto-hide menu")
        self.assertIn("--rail-w:260px", " ".join(m.value for m in at.markdown))
        at.sidebar.button(key="rail-pin").click().run()
        self.assertFalse(at.session_state["rail_pinned"])
        self.assertNotIn("--rail-w:260px", " ".join(m.value for m in at.markdown))

    def test_pin_state_survives_page_changes(self):
        at = workspace("overview", {"rail_pinned": True})
        at.sidebar.button(key="nav-alerts").click().run()
        self.assertTrue(at.session_state["rail_pinned"])
        self.assertEqual(at.session_state["view"], "alerts")

    def test_sign_out_clears_the_pin_with_the_session(self):
        at = workspace("overview", {"rail_pinned": True})
        at.sidebar.button(key="signout").click().run()
        self.assertNotIn("rail_pinned", at.session_state)

    def test_every_page_has_the_same_rail(self):
        for view in VIEW_IDS:
            with self.subTest(view=view):
                at = workspace(view)
                self.assertEqual(len([b for b in at.sidebar.button if str(b.key).startswith("nav-")]), 7)
                self.assertEqual(len(at.exception), 0)


class TestRailCss(unittest.TestCase):
    css = theme.CSS

    def test_compact_and_expanded_widths_are_controlled(self):
        self.assertIn("width:56px", self.css)
        self.assertIn("width:228px", self.css)
        self.assertIn("--rail-w:84px", self.css)
        self.assertIn("--rail-w:260px", theme.RAIL_PINNED_CSS)

    def test_hover_overlay_does_not_resize_the_layout_column(self):
        # The layout column (the section) keeps var(--rail-w); only the fixed-position content grows.
        self.assertRegex(self.css, r'\[data-testid="stSidebarContent"\]\s*\{[^}]*position:fixed')
        self.assertRegex(self.css, r'section\[data-testid="stSidebar"\]\s*\{[^}]*width:var\(--rail-w\)')
        hover = re.search(r'stSidebar"\]:hover \[data-testid="stSidebarContent"\][^{]*\{([^}]*)\}', self.css, re.S)
        self.assertIsNotNone(hover)
        self.assertNotIn("margin", hover.group(1))

    def test_keyboard_focus_reveals_labels_and_is_visible(self):
        self.assertIn(":has(:focus-visible)", self.css)
        self.assertRegex(self.css, r":focus-visible\s*\{[^}]*outline:2px solid")

    def test_labels_hidden_when_compact_and_shown_when_expanded(self):
        self.assertIn("--lab:0", self.css)
        self.assertIn("--lab:1", self.css)
        self.assertIn("opacity:var(--lab)", self.css)

    def test_motion_is_subtle_and_respects_reduced_motion(self):
        self.assertIn("prefers-reduced-motion:reduce", self.css)
        self.assertRegex(self.css, r"prefers-reduced-motion:reduce\)\s*\{[^}]*transition:none")
        self.assertNotRegex(self.css, r"transition:[^;}]*[3-9]\d\ds")  # nothing slower than ~300 ms

    def test_mobile_uses_the_native_drawer_with_labels(self):
        mobile = self.css[self.css.index("@media (max-width:768px) {\n  [data-testid=\"stSidebarContent\"]"):]
        self.assertIn("--lab:1", mobile[:200])
        self.assertIn(".st-key-rail-pin { display:none; }", mobile)

    def test_rail_top_is_aligned_with_the_page_header(self):
        self.assertIn("--rail-top:1.75rem", self.css)
        self.assertIn("--page-top:calc(var(--rail-top) + 15px)", self.css)  # one token: header top = brand mark top
        self.assertRegex(self.css, r"\.block-container\s*\{\s*padding:var\(--page-top\)")
        self.assertIn("top:var(--rail-top)", self.css)

    def test_pinned_css_tokens_resolve(self):
        self.assertNotRegex(theme.RAIL_PINNED_CSS, r"__[A-Z_0-9]+__")


class TestRailComponents(unittest.TestCase):
    def test_group_label_markup_and_escaping(self):
        self.assertEqual(ui.nav_label("Today"), '<div class="mt-nav-label"><span class="t">Today</span></div>')
        self.assertNotIn("<script>", ui.nav_label('<script>x</script>'))

    def test_status_indicator_says_synthetic_and_not_live(self):
        html = ui.rail_status("31 Dec 2024")
        self.assertIn("Synthetic data", html)
        self.assertIn("not a live feed", html)
        self.assertIn("31 Dec 2024", html)

    def test_page_footer_keeps_the_disclaimers(self):
        html = ui.page_footer("31 Dec 2024")
        for phrase in ("Synthetic dataset", "not a live feed", "Educational prototype", "Not a clinical decision system"):
            self.assertIn(phrase, html)

    def test_rail_brand_contains_the_logo(self):
        self.assertIn("<svg", ui.rail_brand_html())


class TestTypography(unittest.TestCase):
    css = theme.CSS

    def test_scale_matches_the_design_brief(self):
        self.assertTrue(24 <= rem_px(self.css, ".mt-title") <= 28)                 # page titles 24-28 px
        self.assertTrue(25 <= rem_px(self.css, ".mt-stat .v") <= 30)               # KPI values 25-30 px
        self.assertTrue(25 <= rem_px(self.css, ".mt-tile-value") <= 30)
        self.assertTrue(13 <= rem_px(self.css, ".mt-subtitle") <= 14.5)            # body 13-14 px
        self.assertTrue(11 <= rem_px(self.css, ".mt-eyebrow") <= 12)               # secondary labels 11-12 px
        self.assertTrue(11 <= rem_px(self.css, ".mt-section") <= 12.5)

    def test_nothing_is_smaller_than_ten_pixels(self):
        sizes = [float(x) * 16 for x in re.findall(r"font-size:\s*([\d.]+)rem", self.css)]
        self.assertGreaterEqual(min(sizes), 10.0)

    def test_inter_is_self_hosted_and_there_is_no_external_font_request(self):
        for css in (theme.CSS, theme.LANDING_CSS):
            self.assertNotIn("fonts.googleapis", css)
            self.assertNotIn("@import", css)
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf8")
        self.assertIn('font = "Inter, sans-serif"', config)
        self.assertIn("[[theme.fontFaces]]", config)
        self.assertIn("app/static/inter-latin.woff2", config)
        font = ROOT / "static" / "inter-latin.woff2"
        self.assertTrue(font.exists())
        self.assertEqual(font.read_bytes()[:4], b"wOF2")
        self.assertTrue((ROOT / "static" / "INTER-LICENSE.txt").exists())


class TestPageChangePerformanceDesign(unittest.TestCase):
    def test_pure_chart_builders_are_cached_and_what_if_charts_are_not(self):
        for name in ("opd_forecast_chart", "demand_trend", "occupancy_trend", "wait_trend", "anomaly_chart",
                     "risk_distribution", "model_comparison_chart"):
            self.assertTrue(hasattr(getattr(charts, name), "clear"), f"{name} should be cached")
        for name in ("paired_occupancy", "risk_probability_chart"):
            self.assertFalse(hasattr(getattr(charts, name), "clear"), f"{name} must not be cached")

    def test_cached_chart_is_equal_for_equal_input_and_new_for_new_input(self):
        import pandas as pd

        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "opd_patients": [1, 2, 3, 4, 5],
                           "emergency_patients": [1, 1, 1, 1, 1]})
        import json

        def content(fig):  # data and layout; plotly's default template is replaced by Streamlit's theme anyway
            d = json.loads(fig.to_json())
            d["layout"].pop("template", None)
            return d

        a, b = charts.demand_trend(df), charts.demand_trend(df)
        self.assertEqual(content(a), content(b))
        c = charts.demand_trend(df.assign(opd_patients=[9, 9, 9, 9, 9]))
        self.assertNotEqual(content(a), content(c))

    def test_trends_builds_only_the_open_tab_and_every_tab_still_works(self):
        default = workspace("analytics")
        self.assertLessEqual(len(default.get("plotly_chart")), 4)  # the Demand tab only, not all 13 charts
        for tab, expected in (("Capacity & flow", "Admissions and discharges"), ("Staffing & supplies", "Doctors and nurses"),
                              ("Unusual activity", "Days flagged as unusual"), ("Patient risk", "Risk class distribution")):
            at = workspace("analytics", {"trends_tab": tab})
            self.assertEqual(len(at.exception), 0, tab)
            self.assertIn(expected, " ".join(m.value for m in at.markdown), tab)

    def test_forecasts_tabs_are_lazy_but_complete(self):
        default = workspace("predictions")
        self.assertNotIn("Waiting-time prediction", " ".join(m.value for m in default.markdown))
        for tab, expected in (("Waiting time", "Waiting-time prediction"), ("Resources", "Resource projections"),
                              ("How this is calculated", "All models at a glance")):
            text = " ".join(m.value for m in workspace("predictions", {"forecasts_tab": tab}).markdown)
            self.assertIn(expected, text, tab)

    def test_prefetch_is_idempotent_and_does_not_change_results(self):
        from src.ui import backend

        backend.warm_up()
        backend.prefetch()
        self.assertTrue(backend.is_ready())
        before = backend.predicted_state(3).value.to_dict()
        backend.prefetch()
        self.assertEqual(backend.predicted_state(3).value.to_dict(), before)

    def test_no_spinner_after_the_first_load(self):
        source = (ROOT / "src/ui/workspace.py").read_text(encoding="utf8")
        self.assertIn("if backend.is_ready():", source)


class TestFocusedPolish(unittest.TestCase):
    """Targeted UI polish: icon boxes, header alignment, navbar, CTAs, accents and hover rules."""

    css = theme.CSS
    landing = theme.LANDING_CSS

    def test_rail_icon_box_is_larger_than_the_glyph(self):
        # Measured in the browser: the glyph is 21 x 22 px; the old box was 20.8 x 16 px and clipped it.
        rule = re.search(r'\[data-testid="stSidebar"\] \[data-testid="stIconMaterial"\] \{([^}]*)\}', self.css).group(1)
        for prop in ("width", "min-width", "height"):
            self.assertGreaterEqual(float(re.search(prop + r":([\d.]+)rem", rule).group(1)) * 16, 22.4, prop)
        self.assertIn("display:inline-flex", rule)
        self.assertIn("line-height:1.5rem", rule)

    def test_compact_buttons_have_no_gap_to_hidden_labels_and_bottom_group_fits_the_rail(self):
        self.assertIn("gap:calc(var(--lab) * .7rem)", self.css)
        self.assertIn("width:calc(100% - 16px) !important", self.css)

    def test_header_top_uses_one_token_and_matches_the_brand_mark(self):
        self.assertIn("--page-top:calc(var(--rail-top) + 15px)", self.css)
        self.assertRegex(self.css, r"\.mt-rail-brand \{[^}]*height:44px;[^}]*margin:0 0 1\.5rem")  # logo 36 px centred in 44 px
        self.assertEqual(self.css.count("var(--page-top)"), 1)  # used once: the page container (the rail uses --rail-top)

    def test_navbar_is_one_centred_row(self):
        self.assertRegex(self.landing, r"\.lp-nav \{[^}]*height:2\.5rem;[^}]*margin-bottom:1rem")
        self.assertRegex(self.landing, r'\.st-key-pubnav button \{[^}]*height:2\.5rem[^}]*white-space:nowrap')
        self.assertIn('.st-key-pubnav [data-testid="stHorizontalBlock"] { align-items:center !important; }', self.landing)

    def test_sign_in_is_a_subtle_sage_pill_and_get_started_stays_primary(self):
        rule = re.search(r"\.st-key-pub_signin button \{([^}]*)\}", self.landing).group(1)
        self.assertIn(theme.SAGE_SOFT, rule)
        self.assertNotIn("transparent", rule.split("background:")[1].split(";")[0])
        self.assertIn("border-radius:10px", self.landing)

    def test_large_cta_buttons_share_size_radius_and_type(self):
        for sel in (".st-key-cta-primary", ".st-key-preview_demo", ".st-key-lp-final"):
            self.assertIn(f'[data-testid="stMain"] {sel} button', self.landing)
        self.assertIn("height:3rem !important; min-height:3rem !important", self.landing)
        ghost = re.search(r"\.lp-btn-ghost \{([^}]*)\}", self.landing).group(1)
        for needle in ("height:3rem", "border-radius:10px", "font-size:.9375rem", "font-weight:600"):
            self.assertIn(needle, ghost)
        self.assertIn("font-size:.9375rem !important; font-weight:600 !important", self.landing)

    def test_static_cards_do_not_react_to_hover(self):
        for static in ('[class*="st-key-card-"]:hover', ".mt-alert:hover", ".mt-stat:hover", ".mt-tile:hover", ".mt-sev:hover"):
            self.assertNotIn(static, self.css, static)

    def test_interactive_elements_have_hover_active_and_focus_states(self):
        for needle in ('div[data-testid="stButton"] button:hover', 'div[data-testid="stButton"] button:active',
                       'button[kind="primary"]:hover', '[data-testid="stFormSubmitButton"] button:active',
                       ".lp-btn-ghost:active", 'button[role="tab"]:hover', "summary:hover"):
            self.assertIn(needle, self.css + self.landing, needle)
        self.assertIn("transform:translateY(-1px)", self.css)  # a very small lift
        hover_rules = re.findall(r"[^{}]*:(?:hover|active)[^{}]*\{([^}]*)\}", self.css + self.landing)
        self.assertTrue(hover_rules)
        for body in hover_rules:  # hover/active may lift 1 px, never grow, spin or glow
            self.assertNotIn("scale(", body)
            self.assertNotIn("rotate(", body)
            self.assertNotRegex(body, r"box-shadow:\s*0 0 ")
        self.assertRegex(self.css, r"\[data-testid=\"stMain\"\] button:focus-visible")

    def test_transitions_stay_within_the_calm_range(self):
        durations = [float(x) for x in re.findall(r"transition:[^;}]*?\s\.?(\d?\.\d+)s", self.css + self.landing)]
        self.assertTrue(durations)
        self.assertLessEqual(max(durations), 0.25)

    def test_accents_are_restrained(self):
        self.assertIn("color-mix(in srgb, var(--sev) 5%, #FFFFFF)", self.css)  # alert cards tinted by severity, faintly
        self.assertIn(".mt-sev:not(.zero)", self.css)
        self.assertIn(".st-key-lp-howband", self.landing)
        self.assertIn("linear-gradient(100deg, #F0F8F4", self.css)  # one soft mint wash on the status band
        self.assertNotRegex(self.css + self.landing, r"#(?:00FF|FF00|0FF)[0-9A-Fa-f]{0,3}")  # no neon
        for prop in ("box-shadow:0 0 12px", "text-shadow"):
            self.assertNotIn(prop, self.css + self.landing)

    def test_text_accent_colours_keep_wcag_contrast(self):
        from tests.test_ui_components import contrast

        for fg, bg in ((theme.PRIMARY, "#F0F8F4"), (theme.PRIMARY, theme.SURFACE), (theme.PRIMARY, "#F5FAF7"),
                       (theme.PRIMARY, theme.SAGE_SOFT), (theme.PRIMARY, "#EAF4EF"), (theme.MUTED, theme.WARM_SOFT),
                       (theme.PRIMARY, "#F6FBF8")):
            self.assertGreaterEqual(contrast(fg, bg), 4.5, f"{fg} on {bg}")


if __name__ == "__main__":
    unittest.main()
