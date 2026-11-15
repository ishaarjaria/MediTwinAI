"""Tests for dashboard components, theme and chart builders (Phase 7)."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import charts, theme
from src.ui import components as ui
from src.ui import data as d

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
EVIL = '<script>alert("x")</script>'


class TestTheme(unittest.TestCase):
    def test_palette_values_are_valid_hex(self):
        tokens = [theme.SIDEBAR, theme.PRIMARY, theme.INK, theme.BG, theme.SURFACE, theme.BORDER, theme.TEAL, theme.TEAL_SOFT]
        for pair in list(theme.SEVERITY.values()) + list(theme.STATUS.values()):
            tokens.extend(pair)
        for _label, ink, bg in theme.PROVENANCE.values():
            tokens.extend([ink, bg])
        tokens.extend(theme.SERIES)
        for t in tokens:
            self.assertRegex(t, HEX, t)

    def test_every_severity_and_status_is_styled(self):
        self.assertEqual(set(theme.SEVERITY), {"Low", "Medium", "High", "Critical"})
        self.assertEqual(set(theme.STATUS), {"Normal", "Moderate", "High", "Critical"})
        self.assertEqual(set(theme.PROVENANCE), {"observed", "ml", "rule", "unavailable"})
        self.assertEqual(set(theme.PROVENANCE_SHORT), set(theme.PROVENANCE))

    def test_css_defines_the_classes_components_use(self):
        for cls in ("mt-tile", "mt-pill", "mt-badge", "mt-table", "mt-alert", "mt-callout",
                    "mt-header", "mt-bar", "mt-empty", "st-key-card-", "st-key-nav-"):
            self.assertIn(cls, theme.CSS, cls)

    def test_css_does_not_override_the_icon_font(self):
        # A wildcard font-family rule once turned Material icons into raw text.
        self.assertNotIn('[class*="st-"]', theme.CSS.split("/* ---------- Sidebar")[0])

    def test_plotly_layout_is_transparent_and_sized(self):
        layout = theme.plotly_layout(333)
        self.assertEqual(layout["height"], 333)
        self.assertEqual(layout["paper_bgcolor"], "rgba(0,0,0,0)")


class TestAtoms(unittest.TestCase):
    def test_badges_and_pills(self):
        self.assertIn("Recorded", ui.badge("observed"))
        self.assertIn("Expected", ui.badge("ml"))
        self.assertIn(">Expected<", ui.badge("ml", compact=True))
        self.assertIn("Estimated", ui.badge("rule"))
        self.assertIn("Unavailable", ui.badge("something-else"))
        self.assertIn("Critical", ui.status_pill("Critical"))
        unknown = ui.status_pill("not-a-status")  # unknown label: shown as given, neutral styling
        self.assertIn("not-a-status", unknown)
        self.assertIn(theme.STATUS["Normal"][0], unknown)

    def test_text_is_escaped_everywhere(self):
        builders = [
            ui.page_header(EVIL, EVIL, EVIL.replace("script", "b")),
            ui.section(EVIL),
            ui.tile(EVIL, EVIL, EVIL, "sub", foot_html=""),
            ui.callout(EVIL),
            ui.empty_state(EVIL, EVIL),
            ui.unavailable(EVIL, EVIL),
            ui.alert_card("High", EVIL, EVIL, "current", EVIL, "ml", EVIL),
            ui.table([(EVIL, False)], [[ui.esc(EVIL)]]),
            ui.sidebar_footer(EVIL),
        ]
        for html in builders:
            self.assertNotIn("<script>", html)

    def test_builders_emit_single_line_html(self):
        # Indented or multi-line HTML would be rendered as a Markdown code block.
        samples = [
            ui.tile("Beds", "61", "/ 250", "189 free", bar_pct=24.4, foot_html=ui.badge("observed")),
            ui.alert_card("Low", "t", "c", "current", "31 Dec", "rule", "m"),
            ui.page_header("T", "S"),
            ui.table([("A", False), ("B", True)], [["__group__", "G"], ["x", "1"]]),
            ui.provenance_legend(),
            ui.brand_html(),
            ui.occupancy_bar(50),
        ]
        for html in samples:
            self.assertNotIn("\n", html)

    def test_occupancy_bar_clamps_and_survives_bad_values(self):
        self.assertIn("width:100.0%", ui.occupancy_bar(250))
        self.assertIn("width:0.0%", ui.occupancy_bar(-5))
        for bad in (None, float("nan"), float("inf")):
            self.assertIn("mt-bar", ui.occupancy_bar(bad))
        self.assertIn("left:70.0%", ui.occupancy_bar(10))  # tick at the 70% mark

    def test_occupancy_colour_follows_the_capacity_ladder(self):
        c = [ui.occupancy_color(v) for v in (10, 70, 85, 95)]
        self.assertEqual(len(set(c)), 4)
        self.assertEqual(ui.occupancy_color(69.99), ui.occupancy_color(0))
        self.assertEqual(ui.occupancy_color(95), theme.SEVERITY["Critical"][0])

    def test_delta_html(self):
        self.assertIn("bad", ui.delta_html(3, "bad", 1, "%"))
        self.assertIn("▲", ui.delta_html(3, "bad", 1, "%"))
        self.assertIn("▼", ui.delta_html(-3, "good", 1, "%"))
        self.assertIn("—", ui.delta_html(None, "neutral"))

    def test_value_text_handles_missing_values(self):
        self.assertEqual(ui.value_text(None, 1, " min"), "—")
        self.assertEqual(ui.value_text(12.34, 1, " min"), "12.3 min")

    def test_table_group_rows_span_all_columns(self):
        html = ui.table([("A", False), ("B", True), ("C", True)], [["__group__", "Capacity"], ["x", "1", "2"]])
        self.assertIn('colspan="3"', html)
        self.assertIn("Capacity", html)
        self.assertEqual(html.count("<tr>"), 3)  # header + group + data row

    def test_alert_card_optional_badge_and_date(self):
        with_badge = ui.alert_card("High", "t", "Beds", "current", "31 Dec", "ml", "m")
        without = ui.alert_card("High", "t", "Beds", "predicted", "", None, "m", scope_label="Scenario")
        self.assertIn("Expected", with_badge)
        self.assertNotIn("mt-badge", without)
        self.assertIn("Scenario", without)
        self.assertNotIn(" ·  · ", without)  # empty date does not leave a gap

    def test_sidebar_footer_keeps_the_disclaimer(self):
        html = ui.sidebar_footer("31 Dec 2024")
        self.assertIn("31 Dec 2024", html)
        self.assertIn("not a live feed", html.lower())
        self.assertIn("Not a clinical decision system", html)
        self.assertIn("Synthetic", ui.sidebar_footer(None))


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class TestLightThemeAccessibility(unittest.TestCase):
    AA = 4.5

    def test_text_colours_meet_wcag_aa_on_every_surface(self):
        for fg in (theme.INK, theme.MUTED, theme.FAINT):
            for bg in (theme.BG, theme.SURFACE, theme.SURFACE_2):
                if fg == theme.FAINT and bg == theme.SURFACE_2:
                    continue  # FAINT is only used on page and card backgrounds
                self.assertGreaterEqual(contrast(fg, bg), self.AA, f"{fg} on {bg}")

    def test_severity_status_and_provenance_text_are_readable(self):
        for name, (ink, bg) in {**theme.SEVERITY, **theme.STATUS}.items():
            self.assertGreaterEqual(contrast(ink, bg), self.AA, name)
            self.assertGreaterEqual(contrast(ink, theme.SURFACE), self.AA, name)
        for key, (_label, ink, bg) in theme.PROVENANCE.items():
            self.assertGreaterEqual(contrast(ink, bg), self.AA, key)

    def test_accent_colours_are_readable_on_the_dark_background(self):
        for colour in (theme.TEAL, theme.BLUE, theme.AMBER, theme.ORANGE, theme.RED):
            self.assertGreaterEqual(contrast(colour, theme.BG), self.AA, colour)

    def test_theme_is_light(self):
        self.assertGreater(_luminance(theme.BG), 0.85)
        self.assertGreater(_luminance(theme.SURFACE), 0.95)
        self.assertGreater(_luminance(theme.SIDEBAR), 0.9)
        self.assertLess(_luminance(theme.INK), 0.05)  # near-black text

    def test_brand_colours_follow_the_medical_sage_system(self):
        self.assertEqual(theme.PRIMARY.lower(), "#145a55")
        self.assertEqual(theme.TEAL.lower(), "#2e7d73")
        self.assertEqual(theme.SAGE.lower(), "#88bfa3")
        self.assertEqual(theme.BG.lower(), "#f2f7f5")
        self.assertEqual(theme.SURFACE.lower(), "#ffffff")
        self.assertEqual(theme.WARM.lower(), "#e8e2d6")
        self.assertEqual(theme.INK.lower(), "#142b2a")
        self.assertGreaterEqual(contrast(theme.PRIMARY, theme.SURFACE), 7.0)  # primary buttons and headings

    def test_primary_button_text_is_readable(self):
        self.assertGreaterEqual(contrast("#FFFFFF", theme.PRIMARY), self.AA)
        self.assertGreaterEqual(contrast("#FFFFFF", theme.PRIMARY_HOVER), self.AA)

    def test_streamlit_native_theme_matches_the_design_tokens(self):
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf8").lower()
        self.assertIn('base = "light"', config)
        for token in (theme.BG, theme.PRIMARY, theme.INK, theme.SURFACE_2, theme.SIDEBAR):
            self.assertIn(token.lower(), config)

    def test_views_do_not_hardcode_colours(self):
        hex_colour = re.compile(r"#[0-9A-Fa-f]{6}\b")
        for path in sorted((ROOT / "src" / "ui" / "views").glob("*.py")):
            self.assertEqual(hex_colour.findall(path.read_text(encoding="utf8")), [], path.name)

    def test_no_leftover_dark_theme_tokens_in_css(self):
        for dark in ("#0B1220", "#111C2D", "#080E19", "#172438", "#F1F5F9", "#36B5A0"):
            self.assertNotIn(dark.lower(), theme.CSS.lower(), dark)

    def test_reduced_motion_is_respected(self):
        self.assertIn("prefers-reduced-motion:reduce", theme.CSS)

    def test_every_css_token_placeholder_is_resolved(self):
        self.assertNotRegex(theme.CSS, r"__[A-Z_0-9]+__")


class TestRedesignComponents(unittest.TestCase):
    def test_info_tooltip_is_escaped_and_accessible(self):
        html = ui.info(EVIL)
        self.assertNotIn("<script>", html)
        self.assertIn('data-tip="', html)
        self.assertIn('tabindex="0"', html)
        self.assertNotIn("\n", html)

    def test_tile_with_tip_and_source_label(self):
        html = ui.tile("Beds in use", "61", "/ 250", "189 free", bar_pct=24.4,
                       foot_html=ui.badge("observed", compact=True), tip="Plain explanation")
        self.assertIn("Plain explanation", html)
        self.assertIn("Recorded", html)
        self.assertIn("/ 250", html)
        self.assertNotIn("\n", html)
        self.assertNotIn("mt-info", ui.tile("A", "1"))  # no tooltip unless asked

    def test_hero_shows_status_summary_and_facts(self):
        html = ui.hero("High", "Under high pressure", "Beds are filling up.",
                       [("Active alerts", "3", "Highest: High"), ("Data as of", "31 Dec", "Synthetic")], tip="Explain")
        for text in ("Under high pressure", "Beds are filling up.", "Active alerts", "Highest: High", "Explain"):
            self.assertIn(text, html)
        self.assertIn(theme.STATUS["High"][0], html)
        self.assertNotIn("\n", html)
        self.assertNotIn("<script>", ui.hero("Normal", EVIL, EVIL, [(EVIL, EVIL, EVIL)]))

    def test_severity_tiles_use_distinct_shapes_not_just_colour(self):
        glyphs = list(theme.SEVERITY_GLYPH.values())
        self.assertEqual(len(set(glyphs)), 4)
        for forbidden in ("▲", "▼"):  # reserved for change arrows
            self.assertNotIn(forbidden, glyphs)
        self.assertIn(theme.SEVERITY_GLYPH["Critical"], ui.severity_pill("Critical"))
        zero, nonzero = ui.severity_tile("High", 0), ui.severity_tile("High", 4)
        self.assertIn("zero", zero)
        self.assertNotIn("zero", nonzero)
        self.assertIn(">4<", nonzero)

    def test_meter_clamps_values(self):
        html = ui.meter("Bed occupancy", "24.4%", 250, -5, "Alert at 70%")
        self.assertIn("width:100.0%", html)
        self.assertIn("left:0.0%", html)
        self.assertNotIn("mark", ui.meter("x", "1", 50, None))
        self.assertNotIn("\n", html)

    def test_all_clear_how_to_read_and_logo(self):
        self.assertIn("No active alerts", ui.all_clear("No active alerts", "All good"))
        self.assertNotIn("<script>", ui.all_clear(EVIL, EVIL))
        self.assertIn("How to read this.", ui.how_to_read("A note"))
        self.assertNotIn("<script>", ui.how_to_read(EVIL))
        logo = ui.logo_svg()
        self.assertTrue(logo.startswith("<svg"))
        self.assertIn("aria-label", logo)
        self.assertNotIn("\n", logo)

    def test_page_header_eyebrow_and_date_chip(self):
        html = ui.page_header("Title", "Sub", ui.date_chip("As of 31 Dec 2024"), eyebrow="Command centre")
        for text in ("Command centre", "As of 31 Dec 2024", "Title", "Sub"):
            self.assertIn(text, html)
        self.assertNotIn("mt-eyebrow", ui.page_header("T"))

    def test_sidebar_status_block_has_dataset_date_and_synthetic_indicator(self):
        html = ui.sidebar_footer("31 Dec 2024")
        self.assertIn("Dataset through", html)
        self.assertIn("Synthetic data", html)
        self.assertNotIn("\n", html)


class TestCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = d.load_operations()
        cls.empty = cls.df.iloc[0:0]

    def test_every_chart_returns_a_figure_with_data(self):
        forecast = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3), "opd_forecast": [300, 310, 320],
            "yhat_lower": [250, 260, 270], "yhat_upper": [350, 360, 370],
        })
        flow, _ = d.to_weekly(self.df, ["admissions", "discharges"])
        scores = d.load_anomaly_scores()
        patients = d.load_patients()
        figs = {
            "opd_forecast": charts.opd_forecast_chart(self.df.tail(30), forecast),
            "opd_no_forecast": charts.opd_forecast_chart(self.df.tail(30), None),
            "occupancy": charts.occupancy_trend(self.df),
            "demand": charts.demand_trend(self.df),
            "weekday": charts.weekday_profile_chart(d.weekday_profile(self.df)),
            "monthly": charts.monthly_chart(d.monthly_profile(self.df)),
            "scatter": charts.wait_scatter(self.df),
            "flow": charts.flow_chart(flow, "weekly mean"),
            "staffing": charts.staffing_chart(self.df),
            "stock": charts.stock_chart(self.df),
            "consumption": charts.consumption_chart(self.df),
            "wait": charts.wait_trend(self.df),
            "anomaly": charts.anomaly_chart(self.df, scores),
            "anomaly_no_scores": charts.anomaly_chart(self.df, None),
            "risk": charts.risk_distribution(patients),
            "risk_age": charts.risk_age_box(patients),
            "paired": charts.paired_occupancy(["Beds", "ICU"], [20, 40], [25, 45], "A", "B"),
        }
        for name, fig in figs.items():
            self.assertIsInstance(fig, go.Figure, name)
            self.assertGreaterEqual(len(fig.data), 1, name)

    def test_forecast_chart_includes_interval_and_forecast_traces(self):
        forecast = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3), "opd_forecast": [1, 2, 3],
            "yhat_lower": [0, 1, 2], "yhat_upper": [2, 3, 4],
        })
        names = [t.name for t in charts.opd_forecast_chart(self.df.tail(10), forecast).data]
        self.assertIn("Likely range", names)
        self.assertIn("Expected outpatients", names)
        plain = [t.name for t in charts.opd_forecast_chart(self.df.tail(10), None).data]
        self.assertNotIn("Expected outpatients", plain)

    def test_empty_inputs_give_a_message_not_an_exception(self):
        for fig in (
            charts.opd_forecast_chart(self.empty, None),
            charts.occupancy_trend(self.empty),
            charts.demand_trend(self.empty),
            charts.weekday_profile_chart(d.weekday_profile(self.empty)),
            charts.monthly_chart(d.monthly_profile(self.empty)),
            charts.wait_scatter(self.empty),
            charts.flow_chart(self.empty, "daily"),
            charts.staffing_chart(self.empty),
            charts.stock_chart(self.empty),
            charts.consumption_chart(self.empty),
            charts.wait_trend(self.empty),
            charts.anomaly_chart(self.empty, None),
        ):
            self.assertTrue(fig.layout.annotations, "empty chart should explain itself")

    def test_anomaly_markers_only_use_flagged_days_in_range(self):
        scores = d.load_anomaly_scores()
        window = self.df.iloc[100:200]
        fig = charts.anomaly_chart(window, scores)
        flagged = [t for t in fig.data if t.name == "Flagged day"]
        expected = scores[(scores["anomaly_flag"] == 1) & scores["date"].between(window["date"].min(), window["date"].max())]
        self.assertEqual(len(flagged[0].x) if flagged else 0, len(expected))

    def test_forecast_chart_marks_a_distinct_forecast_zone(self):
        forecast = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3), "opd_forecast": [1, 2, 3],
            "yhat_lower": [0, 1, 2], "yhat_upper": [2, 3, 4],
        })
        fig = charts.opd_forecast_chart(self.df.tail(10), forecast)
        self.assertTrue(any(s.type == "rect" for s in fig.layout.shapes), "forecast zone")
        self.assertTrue(any(s.type == "line" for s in fig.layout.shapes), "latest-data marker")
        self.assertIn("Forecast", [a.text for a in fig.layout.annotations])
        plain = charts.opd_forecast_chart(self.df.tail(10), None)
        self.assertEqual(len(plain.layout.shapes), 0)

    def test_chart_text_is_light_on_dark(self):
        fig = charts.demand_trend(self.df)
        self.assertEqual(fig.layout.font.color, theme.INK)
        self.assertEqual(fig.layout.yaxis.gridcolor, theme.GRID)

    def test_charts_use_the_shared_style(self):
        fig = charts.occupancy_trend(self.df)
        self.assertEqual(fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(fig.layout.font.family, theme.FONT_STACK)


if __name__ == "__main__":
    unittest.main()
