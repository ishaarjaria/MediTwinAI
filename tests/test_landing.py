"""Landing page content, visual assets, plain-language labels and accessibility rules."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import auth, landing_assets as art, theme
from src.ui.pages import common, landing

TECHNICAL = ("XGBoost", "Isolation Forest", "Huber", "ML-predicted", "Rule-derived", "Prophet", "empirical")


def landing_text() -> str:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    return " ".join(m.value for m in at.markdown) + " " + " ".join(b.label for b in at.button)


class TestLandingContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = landing_text()

    def test_hero_headline_and_actions(self):
        self.assertIn("Hospital Operations,", self.text)
        self.assertIn("Simplified", self.text)
        self.assertIn("with AI", self.text)
        for label in ("Get Started", "View Demo", "Sign in"):
            self.assertIn(label, self.text)

    def test_top_navigation_links(self):
        for label in ("Platform", "Features", "How It Works", "About"):
            self.assertIn(f">{label}<", self.text)
        for anchor in ("lp-platform", "lp-features", "lp-how", "lp-about"):
            self.assertIn(f'href="#{anchor}"', self.text)
            self.assertIn(f'id="{anchor}"', self.text)

    def test_all_six_features_and_three_steps(self):
        plain = self.text.replace("&amp;", "&")
        for title in ("Hospital Operations Overview", "Patient Risk Assessment", "OPD & Waiting Time Forecasts",
                      "Hospital Digital Twin", "What-If Scenario Planning", "Alerts & Recommendations"):
            self.assertIn(title, plain)
        self.assertEqual(len(landing.FEATURES), 6)
        for step in ("Understand hospital data", "Predict operational trends", "Plan better decisions"):
            self.assertIn(step, plain)

    def test_disclaimers_are_present_and_honest(self):
        lowered = self.text.lower()
        for phrase in ("educational prototype", "synthetic", "not a clinical decision system",
                       "not connected to any real hospital", "not production authentication"):
            self.assertIn(phrase, lowered)

    def test_no_technical_model_names_or_old_pills_on_the_public_page(self):
        for word in TECHNICAL:
            self.assertNotIn(word, self.text, word)

    def test_no_invented_statistics_or_performance_claims(self):
        copy = "\n".join(
            [d for _i, _t, d in landing.VALUES] + [d for _i, _t, d in landing.FEATURES] + [d for _t, d in landing.STEPS]
        )
        self.assertNotRegex(copy, r"\d+\s*%")  # no percentages
        for banned in ("accuracy", "saves", "reduces", "guarantee", "clinically proven", "trusted by", "customers"):
            self.assertNotIn(banned, copy.lower())

    def test_footer_states_no_real_data(self):
        self.assertIn("No real patient data", self.text)


class TestPreviewScreenshots(unittest.TestCase):
    def test_every_preview_tab_has_a_real_image_file(self):
        for _label, filename, alt, _slug in landing.PREVIEWS:
            path = common.STATIC / filename
            self.assertTrue(path.exists(), filename)
            self.assertGreater(path.stat().st_size, 10_000, filename)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", filename)
            self.assertTrue(alt)

    def test_preview_url_uses_streamlit_static_serving(self):
        self.assertEqual(common.preview_src("preview-overview.png"), "/app/static/preview-overview.png")
        self.assertIsNone(common.preview_src("does-not-exist.png"))
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf8")
        self.assertIn("enableStaticServing = true", config)


class TestAssets(unittest.TestCase):
    def test_illustration_is_accessible_and_carries_no_numbers(self):
        svg = art.hospital_illustration()
        self.assertIn('role="img"', svg)
        self.assertIn("aria-label=", svg)
        self.assertNotIn("<text", svg)  # a picture of data points, never a statistic

    def test_network_background_is_deterministic_and_decorative(self):
        self.assertEqual(art.network_background(), art.network_background())
        self.assertIn('aria-hidden="true"', art.network_background())
        self.assertNotEqual(art.network_background(seed=1), art.network_background(seed=2))

    def test_every_feature_icon_exists(self):
        for icon_name, *_ in landing.FEATURES + landing.VALUES:
            self.assertIn(icon_name, art._ICONS)
            self.assertIn("<svg", art.icon(icon_name))

    def test_browser_frame_escapes_its_inputs(self):
        html = art.browser_frame('x"><script>alert(1)</script>', '"><img onerror=1>', "<b>")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img onerror", html)


class TestPublicStyles(unittest.TestCase):
    def test_tokens_resolved_and_reduced_motion_respected(self):
        self.assertNotRegex(theme.LANDING_CSS, r"__[A-Z_0-9]+__")
        block = theme.LANDING_CSS[theme.LANDING_CSS.index("prefers-reduced-motion"):]
        for cls in ("lp-net-flow", "lp-flow", "lp-ring", "lp-chip"):
            self.assertIn(cls, block)
        self.assertIn("animation:none", block)

    def test_responsive_breakpoints(self):
        for query in ("max-width:1000px", "max-width:820px"):
            self.assertIn(query, theme.LANDING_CSS)

    def test_no_neon_or_glow_effects(self):
        for css in (theme.CSS, theme.LANDING_CSS):
            self.assertNotIn("backdrop-filter", css)  # no glassmorphism
            self.assertNotRegex(css, r"box-shadow:\s*0 0 [1-9]")  # no glow
            self.assertNotIn("linear-gradient(135deg", css.replace("#000 45%", ""))

    def test_public_pages_hide_the_dashboard_sidebar(self):
        self.assertIn('section[data-testid="stSidebar"]', theme.LANDING_CSS)

    def test_no_external_images_or_scripts(self):
        for path in (ROOT / "src/ui/pages").glob("*.py"):
            text = path.read_text(encoding="utf8")
            self.assertNotRegex(text, r"https?://", path.name)
            self.assertNotIn("<script", text)


class TestPlainLanguageLabels(unittest.TestCase):
    def test_provenance_labels(self):
        self.assertEqual({k: v[0] for k, v in theme.PROVENANCE.items() if k != "unavailable"},
                         {"observed": "Recorded", "ml": "Expected", "rule": "Estimated"})

    def test_normal_screens_avoid_model_names(self):
        from streamlit.testing.v1 import AppTest

        for view in ("overview", "alerts", "analytics", "twin", "whatif"):
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
            at.query_params["view"] = view
            at.session_state[auth.SESSION_KEY] = auth.DEMO_USER
            at.run()
            inside_expanders = {m.value for e in at.expander for m in e.markdown}  # optional detail is allowed there
            text = " ".join(m.value for m in at.markdown if m.value not in inside_expanders)
            for word in ("XGBoost", "Isolation Forest", "Huber", "ML-predicted", "Rule-derived"):
                self.assertNotIn(word, text, f"{view}: {word}")

    def test_technical_detail_lives_in_how_this_is_calculated(self):
        from streamlit.testing.v1 import AppTest

        for view in ("predictions", "twin", "whatif", "patient_risk"):
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
            at.query_params["view"] = view
            at.session_state[auth.SESSION_KEY] = auth.DEMO_USER
            at.run()
            labels = [e.label for e in at.expander]
            self.assertTrue(any(label.startswith("How this is calculated") for label in labels), (view, labels))

    def test_forecasts_keep_the_model_names_and_metrics_for_those_who_want_them(self):
        from streamlit.testing.v1 import AppTest

        text = ""
        for tab in ("Outpatients", "How this is calculated"):  # tabs run lazily: open each in turn
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
            at.query_params["view"] = "predictions"
            at.session_state[auth.SESSION_KEY] = auth.DEMO_USER
            at.session_state["forecasts_tab"] = tab
            at.run()
            text += " " + " ".join(m.value for m in at.markdown)
        for needed in ("XGBoost", "Day-of-week mean", "Isolation Forest", "Test MAE", "Prophet"):
            self.assertIn(needed, text)


if __name__ == "__main__":
    unittest.main()
