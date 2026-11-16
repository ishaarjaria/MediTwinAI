"""
Visual assets for the public pages, drawn as inline SVG (no image files, no dependencies).

* ``hospital_illustration`` - an isometric hospital with a floating "digital twin" layer.
  The twin layer shows only abstract data nodes; it deliberately carries no numbers, so
  the picture cannot be mistaken for a statistic.
* ``network_background`` - a faint, slowly drifting network behind the hero.
* ``icon`` - small line icons for the feature list.

Animation is CSS only (see ``theme.LANDING_CSS``) and switches off under
``prefers-reduced-motion``.
"""

from __future__ import annotations

import math
import random
from html import escape

from src.ui import theme

_C30 = math.cos(math.radians(30))


class _Scene:
    """Tiny isometric renderer: boxes, flat shapes and windows, with a bounding box."""

    def __init__(self, unit: float = 30.0) -> None:
        self.u = unit
        self.parts: list[str] = []
        self.xs: list[float] = []
        self.ys: list[float] = []

    def p(self, x: float, y: float, z: float) -> tuple[float, float]:
        px = (x - y) * _C30 * self.u
        py = (x + y) * 0.5 * self.u - z * self.u
        self.xs.append(px)
        self.ys.append(py)
        return px, py

    def poly(self, pts, fill: str, stroke: str = "none", extra: str = "") -> str:
        coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        return f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width=".8" stroke-linejoin="round" {extra}/>'

    def box(self, x, y, z, w, d, h, top, left, right, stroke: str = "none") -> None:
        p = self.p
        self.parts.append(self.poly([p(x, y + d, z), p(x + w, y + d, z), p(x + w, y + d, z + h), p(x, y + d, z + h)], left, stroke))
        self.parts.append(self.poly([p(x + w, y, z), p(x + w, y + d, z), p(x + w, y + d, z + h), p(x + w, y, z + h)], right, stroke))
        self.parts.append(self.poly([p(x, y, z + h), p(x + w, y, z + h), p(x + w, y + d, z + h), p(x, y + d, z + h)], top, stroke))

    def flat(self, x, y, z, w, d, fill: str, extra: str = "") -> None:
        p = self.p
        self.parts.append(self.poly([p(x, y, z), p(x + w, y, z), p(x + w, y + d, z), p(x, y + d, z)], fill, extra=extra))

    def windows_left(self, x, y, z, w, d, h, cols, rows, fill: str, inset=(.18, .22)) -> None:
        """Windows on the face that looks toward +y (the left-front face)."""
        for i in range(cols):
            for j in range(rows):
                u0, u1 = (i + inset[0]) / cols, (i + 1 - inset[0]) / cols
                v0, v1 = (j + inset[1]) / rows, (j + 1 - inset[1]) / rows
                p = self.p
                self.parts.append(self.poly([
                    p(x + u0 * w, y + d, z + v0 * h), p(x + u1 * w, y + d, z + v0 * h),
                    p(x + u1 * w, y + d, z + v1 * h), p(x + u0 * w, y + d, z + v1 * h)], fill))

    def windows_right(self, x, y, z, w, d, h, cols, rows, fill: str, inset=(.18, .22)) -> None:
        """Windows on the face that looks toward +x (the right-front face)."""
        for i in range(cols):
            for j in range(rows):
                u0, u1 = (i + inset[0]) / cols, (i + 1 - inset[0]) / cols
                v0, v1 = (j + inset[1]) / rows, (j + 1 - inset[1]) / rows
                p = self.p
                self.parts.append(self.poly([
                    p(x + w, y + u0 * d, z + v0 * h), p(x + w, y + u1 * d, z + v0 * h),
                    p(x + w, y + u1 * d, z + v1 * h), p(x + w, y + u0 * d, z + v1 * h)], fill))

    def tree(self, x, y, size: float = 1.0) -> None:
        bx, by = self.p(x, y, 0)
        r = 0.55 * self.u * size
        self.parts.append(f'<rect x="{bx - 1.6:.1f}" y="{by - r * 1.1:.1f}" width="3.2" height="{r * 1.1:.1f}" fill="{theme.WARM}"/>')
        self.parts.append(f'<circle cx="{bx:.1f}" cy="{by - r * 1.7:.1f}" r="{r:.1f}" fill="{theme.SAGE}"/>')
        self.parts.append(f'<circle cx="{bx - r * .3:.1f}" cy="{by - r * 1.95:.1f}" r="{r * .55:.1f}" fill="#A9D3BC"/>')
        self.xs += [bx - r, bx + r]
        self.ys += [by - r * 2.7]


def hospital_illustration() -> str:
    s = _Scene(30)
    white = ("#FFFFFF", "#E6F0EC", "#D2E3DC")
    warm = ("#FBF9F4", "#EAE4D8", "#DDD5C5")
    teal = ("#4E9F93", "#2E7D73", "#1F6A61")
    deep = ("#2E7D73", "#145A55", "#0F4642")

    # Ground and paths
    s.box(0, 0, -0.55, 13, 11, 0.55, "#E1F0E8", "#C7DED3", "#B4CFC3")
    s.flat(4.6, 7.9, 0.02, 2.2, 3.1, "#F6FAF8")
    s.flat(8.2, 6.5, 0.02, 3.6, 1.9, "#EEF5F2")
    for k in range(4):  # ambulance-bay markings
        s.flat(8.7 + k * .8, 6.8, 0.03, .3, 1.3, "#FFFFFF")

    # Buildings, ordered back to front
    s.box(0.8, 4.6, 0, 2.6, 3.2, 2.3, *warm)                                  # outpatient wing
    s.windows_left(0.8, 4.6, 0, 2.6, 3.2, 2.3, 3, 2, "#9FCBC0")
    s.windows_right(0.8, 4.6, 0, 2.6, 3.2, 2.3, 3, 2, "#8FBFB3")

    s.box(3.4, 2.4, 0, 5.0, 3.8, 4.8, *white)                                  # main tower
    s.windows_left(3.4, 2.4, 0.0, 5.0, 3.8, 4.8, 6, 5, "#9FCBC0")
    s.windows_right(3.4, 2.4, 0.0, 5.0, 3.8, 4.8, 4, 5, "#8FBFB3")
    s.box(3.9, 3.0, 4.8, 1.7, 1.5, 0.55, *warm)                                # rooftop plant
    s.box(6.0, 3.2, 4.8, 1.9, 2.4, 0.2, "#F6FAF8", "#E6F0EC", "#D2E3DC")        # roof deck
    # Cross emblem on the roof deck
    s.flat(6.5, 4.15, 5.02, 1.2, .42, theme.PRIMARY)
    s.flat(6.89, 3.76, 5.02, .42, 1.2, theme.PRIMARY)

    s.box(5.0, 6.2, 0, 2.6, 1.3, 1.0, *teal)                                   # entrance canopy
    s.windows_left(5.0, 6.2, 0, 2.6, 1.3, 1.0, 4, 1, "#E3F1ED", inset=(.2, .28))

    s.box(8.4, 3.0, 0, 3.2, 3.0, 2.9, *warm)                                   # intensive-care wing
    s.windows_left(8.4, 3.0, 0, 3.2, 3.0, 2.9, 4, 3, "#9FCBC0")
    s.windows_right(8.4, 3.0, 0, 3.2, 3.0, 2.9, 3, 3, "#8FBFB3")
    s.box(8.4, 3.0, 2.9, 3.2, 3.0, 0.22, *deep)                                # accent band

    s.box(9.0, 7.0, 0, 1.2, 0.62, 0.55, "#FFFFFF", "#EAF2EF", "#D6E5DF")       # ambulance
    s.box(9.0, 7.0, 0.55, 0.55, 0.62, 0.22, "#FFFFFF", "#EAF2EF", "#D6E5DF")
    s.flat(9.0, 7.0, 0.56, 1.2, 0.62, theme.TEAL)

    for x, y, sz in ((0.9, 9.3, 1.0), (2.4, 9.9, .85), (11.3, 8.8, .95), (11.9, 4.6, .8), (0.5, 2.3, .9)):
        s.tree(x, y, sz)

    # Floating digital-twin layer above the hospital
    z = 8.4
    p = s.p
    plane = [p(1.2, 1.6, z), p(11.4, 1.6, z), p(11.4, 8.6, z), p(1.2, 8.6, z)]
    twin = [f'<g class="lp-twin">{s.poly(plane, "rgba(46,125,115,.07)", theme.TEAL, "stroke-dasharray=\"5 6\"")}']
    nodes = [(2.2, 2.6), (6.0, 2.2), (10.2, 3.0), (3.0, 6.8), (6.6, 5.4), (10.0, 7.4)]
    npts = [p(x, y, z) for x, y in nodes]
    links = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
    for a, b in links:
        twin.append(f'<line class="lp-flow" x1="{npts[a][0]:.1f}" y1="{npts[a][1]:.1f}" x2="{npts[b][0]:.1f}" y2="{npts[b][1]:.1f}"/>')
    roofs = [(5.5, 4.3, 5.35), (9.9, 4.5, 3.2), (2.1, 6.2, 2.3)]  # where the twin "reads" the building
    for i, (rx, ry, rz) in enumerate(roofs):
        top = p(rx, ry, rz)
        src = npts[[1, 2, 3][i]]
        twin.append(f'<line class="lp-drop" x1="{src[0]:.1f}" y1="{src[1]:.1f}" x2="{top[0]:.1f}" y2="{top[1]:.1f}"/>')
        twin.append(f'<circle cx="{top[0]:.1f}" cy="{top[1]:.1f}" r="3.2" fill="{theme.PRIMARY}"/>')
    for i, (nx, ny) in enumerate(npts):
        twin.append(f'<circle class="lp-ring" style="animation-delay:{i * .55:.2f}s" cx="{nx:.1f}" cy="{ny:.1f}" r="7" fill="none" stroke="{theme.TEAL}" stroke-width="1.2"/>')
        twin.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="4.2" fill="#FFFFFF" stroke="{theme.TEAL}" stroke-width="1.8"/>')
    twin.append("</g>")

    # Abstract data chips (glyphs only, no figures)
    def chip(cx: float, cy: float, glyph: str, delay: float) -> str:
        return (
            f'<g class="lp-chip" style="animation-delay:{delay}s" transform="translate({cx:.1f},{cy:.1f})">'
            f'<rect x="-26" y="-17" width="52" height="34" rx="9" fill="#FFFFFF" stroke="{theme.BORDER}"/>{glyph}</g>'
        )

    pulse = f'<path d="M-15 2h8l4-9 6 17 4-12 3 4h9" fill="none" stroke="{theme.PRIMARY}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    bed = (f'<path d="M-14 7v-12M-14 1h28v6M-14 1v-3h12a4 4 0 0 1 4 4" fill="none" stroke="{theme.PRIMARY}" '
           'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
    bars = "".join(
        f'<rect x="{-14 + i * 9}" y="{9 - h}" width="6" height="{h}" rx="1.6" fill="{c}"/>'
        for i, (h, c) in enumerate(((8, theme.SAGE), (14, theme.TEAL), (20, theme.PRIMARY)))
    ) + ""
    left, top_, right = min(s.xs), min(s.ys), max(s.xs)
    chips = (
        chip(right - 8, top_ + 105, pulse, 0.0)
        + chip(left + 40, top_ + 150, bed, 1.3)
        + chip(right - 30, top_ + 290, bars, 2.4)
    )
    pad = 40
    minx, maxx = min(s.xs) - pad, max(s.xs) + pad
    miny, maxy = min(s.ys) - pad, max(s.ys) + pad
    return (
        f'<svg class="lp-hospital" viewBox="{minx:.0f} {miny:.0f} {maxx - minx:.0f} {maxy - miny:.0f}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Illustration of a hospital with a floating digital twin layer of connected data points">'
        f'<ellipse cx="{(minx + maxx) / 2:.0f}" cy="{maxy - 90:.0f}" rx="{(maxx - minx) * .42:.0f}" ry="{(maxy - miny) * .12:.0f}" fill="rgba(20,90,85,.07)"/>'
        + "".join(s.parts) + "".join(twin) + chips + "</svg>"
    )


def network_background(width: int = 1200, height: int = 700, nodes: int = 34, seed: int = 11) -> str:
    """A faint network of points and links. Layout is seeded, so it is identical on every run."""
    rnd = random.Random(seed)
    pts = []
    for _ in range(nodes * 40):  # bounded attempts keep the layout deterministic and fast
        if len(pts) >= nodes:
            break
        x, y = rnd.uniform(20, width - 20), rnd.uniform(20, height - 20)
        if all((x - a) ** 2 + (y - b) ** 2 > 120 ** 2 for a, b in pts):
            pts.append((x, y))
    lines, dots = [], []
    for i, (x1, y1) in enumerate(pts):
        near = sorted(range(len(pts)), key=lambda j: (pts[j][0] - x1) ** 2 + (pts[j][1] - y1) ** 2)[1:4]
        for j in near:
            if j > i:
                x2, y2 = pts[j]
                cls = "lp-net-line lp-net-flow" if (i + j) % 4 == 0 else "lp-net-line"
                lines.append(f'<line class="{cls}" x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}"/>')
        dots.append(f'<circle class="lp-net-dot" style="animation-delay:{(i % 7) * .7:.1f}s" cx="{x1:.0f}" cy="{y1:.0f}" r="{2.2 + (i % 3) * .6:.1f}"/>')
    return (
        f'<svg class="lp-net" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid slice" aria-hidden="true" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(lines)}{"".join(dots)}</svg>'
    )


_ICONS = {
    "overview": '<rect x="3" y="3" width="7.5" height="9" rx="1.6"/><rect x="13.5" y="3" width="7.5" height="5.5" rx="1.6"/><rect x="13.5" y="11.5" width="7.5" height="9.5" rx="1.6"/><rect x="3" y="15" width="7.5" height="6" rx="1.6"/>',
    "risk": '<path d="M12 21s-7.5-4.6-7.5-10.2A4.3 4.3 0 0 1 12 8.2a4.3 4.3 0 0 1 7.5 2.6C19.5 16.4 12 21 12 21z"/><path d="M7.5 12.5h2.6l1.4-2.6 2 4.2 1.2-1.6h2"/>',
    "forecast": '<path d="M3 20h18"/><path d="M4 15.5l4.5-4.5 3.5 3L19 6.5"/><path d="M15 6.5h4v4"/>',
    "twin": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12L4 7.5M12 12v9"/>',
    "whatif": '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2.2"/><circle cx="9" cy="17" r="2.2"/>',
    "alerts": '<path d="M6 16.5V11a6 6 0 0 1 12 0v5.5l1.6 2H4.4z"/><path d="M10 21h4"/>',
    "data": '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>',
    "trend": '<path d="M3 17l5-5 4 3 8-8"/><circle cx="8" cy="12" r="1.3"/><circle cx="12" cy="15" r="1.3"/><circle cx="20" cy="7" r="1.3"/>',
    "plan": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 12.5l2.6 2.6L16 9.5"/>',
    "shield": '<path d="M12 3l7.5 3v5.5c0 4.6-3.2 8-7.5 9.5-4.3-1.5-7.5-4.9-7.5-9.5V6z"/><path d="M12 8.5v5M9.5 11h5"/>',
}


def icon(name: str, size: int = 24) -> str:
    body = _ICONS.get(name, _ICONS["overview"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def browser_frame(src: str, alt: str, url_label: str = "meditwin · workspace") -> str:
    """A quiet browser window around a real screenshot of the product."""
    return (
        '<div class="lp-frame"><div class="lp-frame-bar"><i></i><i></i><i></i>'
        f'<span>{escape(url_label)}</span></div>'
        '<div class="lp-shot">'
        f'<img src="{escape(src, quote=True)}" alt="{escape(alt, quote=True)}" width="1200" height="750"/></div></div>'
    )
