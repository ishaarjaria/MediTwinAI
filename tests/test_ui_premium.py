"""Tests for the premium-UI components, navigation structure and responsive CSS."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui
from src.ui import data as d
from src.ui import theme
from src.ui.views import NAV_GROUPS, VIEW_IDS, VIEWS

EVIL = '<script>alert("x")</script>'


class TestRing(unittest.TestCase):
    def test_ring_encodes_percentage_and_ticks(self):
        html = ui.occupancy_ring(62, "Beds", "154 of 250")
        self.assertIn("62%", html)
        self.assertIn('aria-label="Beds 62 percent"', html)
        self.assertEqual(html.count("<line"), 3)  # 70 / 85 / 95 status marks
        arc = re.search(r'stroke-dasharray="([\d.]+) ([\d.]+)"', html)
        self.assertAlmostEqual(float(arc.group(1)) / float(arc.group(2)), 0.62, places=2)

    def test_ring_clamps_out_of_range_values_and_escapes_text(self):
        self.assertIn("100%", ui.occupancy_ring(140, "x"))
        self.assertIn("0%", ui.occupancy_ring(-5, "x"))
        self.assertNotIn("<script>", ui.occupancy_ring(50, EVIL, EVIL))

    def test_hero_renders_rings_only_when_given(self):
        base = ui.hero("Normal", "Fine", "All good", [("a", "b", "c")])
        self.assertNotIn("mt-ring", base)
        withr = ui.hero("Normal", "Fine", "All good", [("a", "b", "c")], rings=[("Beds", 60, "x"), ("ICU", 50, "y")])
        self.assertEqual(withr.count('class="mt-ring"'), 2)


class TestSparklineAndStrip(unittest.TestCase):
    def test_sparkline_needs_enough_points(self):
        self.assertEqual(ui.sparkline([]), "")
        self.assertEqual(ui.sparkline([1, 2]), "")
        self.assertIn("<polyline", ui.sparkline([1, 3, 2, 5]))

    def test_sparkline_flat_series_does_not_divide_by_zero(self):
        self.assertIn("<polyline", ui.sparkline([4, 4, 4, 4]))

    def test_strip_has_one_cell_per_item_and_escapes_text(self):
        items = [dict(label=EVIL, value=EVIL, unit=EVIL, sub="ok", badge=ui.badge("ml"), bar_pct=40) for _ in range(3)]
        html = ui.stat_strip(items)
        self.assertEqual(html.count('class="mt-stat"'), 3)
        self.assertNotIn("<script>", html)

    def test_spark_series_from_dataset(self):
        df = d.load_operations()
        self.assertEqual(len(d.spark_series(df, "opd_patients", 30)), 30)
        pressure = d.spark_series(df, "staff_pressure", 10)
        self.assertEqual(len(pressure), 10)
        self.assertTrue(all(p > 0 for p in pressure))
        self.assertEqual(d.spark_series(df, "missing"), [])


class TestTableWrapper(unittest.TestCase):
    def test_table_is_wrapped_so_wide_tables_scroll_instead_of_overflowing(self):
        html = ui.table([("A", False), ("B", True)], [["x", "1"]])
        self.assertTrue(html.startswith('<div class="mt-table-wrap"><table class="mt-table">'))
        self.assertIn(".mt-table-wrap { width:100%; overflow-x:auto; }", theme.CSS)


class TestNavigation(unittest.TestCase):
    def test_every_view_is_in_exactly_one_group(self):
        grouped = [v for _g, ids in NAV_GROUPS for v in ids]
        self.assertEqual(sorted(grouped), sorted(VIEW_IDS))
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(len(VIEWS), 7)

    def test_group_labels(self):
        self.assertEqual([g for g, _ in NAV_GROUPS], ["Today", "Look ahead", "Plan"])


class TestGlossary(unittest.TestCase):
    def test_required_terms_are_explained_in_plain_language(self):
        names = {t for t, _ in d.GLOSSARY_TERMS}
        for needed in ("OPD (outpatients)", "ICU pressure", "Patient risk", "Waiting-time prediction",
                       "Unusual activity", "Forecast interval", "Recorded", "Expected", "Estimated"):
            self.assertIn(needed, names)
        for _t, text in d.GLOSSARY_TERMS:
            self.assertGreater(len(text), 25)
        interval = dict(d.GLOSSARY_TERMS)["Forecast interval"]
        self.assertNotIn("validated 80%", interval.replace("nominal 80%", ""))

    def test_glossary_html_escapes(self):
        self.assertNotIn("<script>", ui.glossary_html([(EVIL, EVIL)]))


class TestResponsiveCss(unittest.TestCase):
    def test_breakpoints_present_and_no_unresolved_tokens(self):
        for query in ("max-width:1280px", "max-width:1000px", "max-width:820px", "max-width:520px"):
            self.assertIn(query, theme.CSS)
        self.assertNotRegex(theme.CSS, r"__[A-Z_0-9]+__")

    def test_icon_font_fallback_rule_exists(self):
        self.assertIn('[data-testid="stIconMaterial"]', theme.CSS)
        self.assertIn("overflow:hidden", theme.CSS)


if __name__ == "__main__":
    unittest.main()
