# Release notes and repository hygiene

## Launch

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt                        # exact tested versions (or requirements.txt)
streamlit run app.py                                        # http://localhost:8501  (landing page)
python -m unittest discover -s tests -t .                   # 419 tests
```

Open http://localhost:8501, choose **Get Started** (create a demo account) or **Continue as demo user**. Old deep links such as `http://localhost:8501/?view=twin` still work
(they ask you to sign in first and then open that page).
The data and models are already in the repository. To rebuild them, follow the run order in the README. Use `--skip patient_risk` with `train_all` to keep the existing patient-risk model.

## Final UI polish (focused fixes)

* **Rail icons were clipped.** Measured in the browser: every icon glyph was 21 x 22 px inside a 20.8 x 16 px box with `overflow:hidden`, so the bottom of every icon was cut. The box is now 24 x 24 px (flex-centred, no hidden overflow), the icon-to-label gap is zero when compact, and the bottom group was 16 px wider than the rail (a Streamlit `width:100%` overrode my insets). All 10 icons are now measured as fully inside their box and the rail at 1440, 1280 and 820 px (compact and expanded) and in the 390 px drawer.
* **Header aligned with the brand.** One token, `--page-top`, puts the top of every page header at the same y as the MediTwin mark (43 px, measured on all seven pages in compact, expanded and pinned states); content x is identical across pages and unchanged by hovering (97 px compact and hovered, 260 px pinned). Date and status chips are centred on the title line.
* **Landing navbar.** Logo, four links, Sign in and Get Started share one centre line (spread 0.0 px at 1440, 1280, 820 and 390; it was 8 px). Sign in is a soft sage button; Get Started stays primary. The phone navbar fits on one row.
* **CTAs.** Get Started and View Demo are both 48 px high, 10 px radius, 15 px / 600 (they were 40 and 48 px, 16 and 14.4 px). The same size now applies to the preview, final-call-to-action and sign-in buttons.
* **Accents (restrained).** A faint mint wash on the status band and KPI strip, deep-sage status headline and KPI numbers, warm-neutral date chips, notes and callouts, severity-tinted alert cards and summary tiles, a sage band behind "How it works". Contrast of every text/background pair is tested (WCAG AA).
* **Hover.** Buttons lift 1 px with a soft shadow, have pressed and keyboard-focus states, and transitions are 180 ms (rail width 220 ms), off under reduced motion. Static information cards, KPI cells and alert cards no longer react to hover (checked).

Verification: 430 tests pass (419 before, +11), 0 skipped; browser journey 56 / 56; hover checks 14 / 14 (includes "static cards do not react", 150-220 ms durations, reduced motion); layout checks 18 / 18 (icons, header alignment on all seven pages in three rail states, no horizontal shift, drawer at 390 px); checksums of the 39 data, model and backend files identical. The same two Streamlit `404` console messages on a direct load of `/signin` or `/workspace` remain (framework behaviour). Before/after images: `docs/screenshots/11*.png`. Not individually viewed: the 820 and 390 sign-in pages and the Trends, Forecasts, Twin and Patient Risk pages after the final accent change (they use the same rules as the pages that were viewed).

## Latest stage: custom navigation rail, typography, calmer pages and faster page changes

Everything in the earlier stages below still applies; this stage changes presentation, navigation and speed only. No data, model, Digital Twin, What-If, alert or recommendation logic changed (the 39 data, model and backend-source files have checksums identical to the baseline, and `tests/test_backend_snapshot.py` passes).

**Navigation rail.** Streamlit's sidebar is restyled (CSS only, no JavaScript) into a slim floating rail. Compact it is 56 px wide and shows the MediTwin mark and one line icon per page, with a hairline between the groups Today, Look ahead and Plan. Hovering the rail, or moving keyboard focus into it, reveals the labels as an overlay (228 px) **without moving the page**: the layout column keeps its width, so the content x position is the same compact, hovered and focused (measured in the browser). **Keep menu open** pins it expanded and the page then makes room for it; that is an explicit choice, remembered for the session, and **Auto-hide menu** reverses it. The active page has a sage tint and a left indicator. The bottom of the rail holds a small synthetic-data indicator, the Plain-language guide (now a dialog, closed with Escape), the pin toggle, the signed-in user and Sign out. The top of the rail is aligned with the top of the page header (both 28 px). Every item is a real button, so Tab, Enter and Space work, with a visible focus ring. On phones Streamlit's own drawer is used, with labels. Native tooltips were deliberately not used on the icons, because the labels already appear the moment the rail is hovered and the tooltip bubble covered them.
The explanatory text that used to fill the sidebar moved to one quiet line under every page (synthetic data, not a live feed, not a clinical decision system), so the disclaimers are still visible on every page.

**Typography and spacing.** Inter is now self-hosted (`static/inter-latin.woff2`, 48 KB, SIL Open Font License, declared in `.streamlit/config.toml`), so it really applies everywhere (before, the headings fell back to Streamlit's default font while a Google Fonts `@import` ran on every page change). Scale: page titles 26 px, section labels 11 px, body 13.5 to 14 px, tables 13 px, KPI values 27 to 28 px, nothing below 10.4 px (tested). Page padding, rail gutter and card spacing use one set of tokens; the "how to read" notes lost their boxed background (a left rule instead), the status band, KPI strip and headings were resized, and subtitles were shortened. Information was moved, not removed: model names, metrics, limitations and every disclaimer are still there (under "How this is calculated" and in the page footer).

**Landing page and sign-in.** Warmer, calmer hero (warm-white to sage-mist background, a much fainter and sparser network, 48 px heading with controlled width, consistent button sizes and spacing), tablet layout with a capped illustration and two-column features, and phone heading size. Sign-in is a centred, symmetrical pair of equal-height panels (about 610 px), a compact amber demo notice, one small note instead of a long limitations paragraph, and "Back to home" as a quiet text link above the panels. Continue as demo user stays directly under the form. All authentication behaviour is unchanged.

**Page-change speed (measured, same method before and after).** Cause found by profiling, not assumed: Streamlit's `st.tabs` runs the code of every tab on every rerun, so Trends built 13 Plotly figures each time (about 0.1 s of property validation per figure, 1.4 to 1.9 s of a 2.2 s render); cold first visits also paid for loading data and importing each view; and a "Loading models..." spinner ran on every page change. Not the cause: Google Fonts (no font request occurs during a page change) and static files (under 0.15 s each). Changes: pure chart builders are cached by their inputs (`st.cache_data`; What-If and form-result charts are deliberately not cached), tabs on Trends and Forecasts run lazily (`on_change="rerun"`, only the open tab computes), the data caches and view imports are filled once per server process (`backend.prefetch`), and the spinner shows only on that first load. A cached chart renders pixel-identically to a fresh one (0 differing pixels across the three Overview charts once the render is stable).

| Page | First visit after a server start: before | after | Repeat visit (median): before | after |
|---|---|---|---|---|
| Overview | 1,883 ms | 971 ms | 1,148 ms | 629 ms |
| Alerts | 595 ms | 407 ms | 636 ms | 363 ms |
| Trends | 3,182 ms | 740 ms | 2,704 ms | 941 ms |
| Forecasts | 2,372 ms | 2,352 ms | 1,403 ms | 600 ms |
| Patient Risk | 5,176 ms | 616 ms | 630 ms | 410 ms |
| Hospital Twin | 346 ms | 630 ms | 790 ms | 515 ms |
| What-If | 3,915 ms | 796 ms | 948 ms | 595 ms |
| **All pages, median** | | | **948 ms** | **578 ms** |

Times are from the click until the page stops changing (headless Edge, real server, three rounds; a second run on a warm server gave 730 ms before and 613 ms after, so differences of a few hundred milliseconds on a single page are within run-to-run noise). Server compute per page change (warm): Trends 1.47 s to 0.21 s, Forecasts 0.41 s to 0.11 s, Overview 0.44 s to 0.20 s. Honest caveats: the first visit to Forecasts is still about 2.4 s (first use of its widgets plus its figures; 0.6 s afterwards); the very first workspace load after sign-in is about 0.4 s longer (3.0 s against 2.6 s) because the prefetch now runs there, once per server process; Hospital Twin's first visit was not faster (346 ms against 630 ms, within noise); and an earlier set of "before" timings with 5 to 10 s spikes was discarded because stray test browsers were still running (the measurements were repeated from a clean state).

### Verification of this stage

| Check | Result |
|---|---|
| Automated tests | **419 pass, 0 skipped** (386 before this stage, +33 in `tests/test_navigation_rail.py`) |
| Browser journey (headless Edge, real server): landing, deep links and direct `/workspace` blocked, create account, demo user, sign out, all 7 pages, Patient Risk, What-If with alerts and recommendations, **rail compact 56 px, hover 228 px without moving the page, keyboard focus expands with a visible ring and no page shift, Keep menu open / Auto-hide, pin survives a page change, guide dialog opens and Escape closes it, every Trends and Forecasts tab and filter** | **56 / 56 pass** |
| Hover states (real mouse, computed styles) | 11 / 11 change as designed |
| Console / network | 2 console messages, 0 failed requests: the same Streamlit `404` pair for `/<route>/_stcore/health` and `/host-config` on a **direct** load of `/signin` or `/workspace`, reproduced by a minimal two-page Streamlit app with none of this project's code (the client probes the route as a base path, then retries at `/`); no functional effect, not suppressed |
| Test assertions changed | Only the four tests that read content of non-default tabs now open that tab through its state key (`trends_tab`, `forecasts_tab`), because tabs run lazily; the assertions are unchanged. The test that checks the disclaimer on every page is unchanged and passes through the new page footer |
| Responsive checks | landing and sign-in at 1440 / 1280 / 820 / 390; overview at 1440 / 1280 / 820 / 390; no horizontal page scroll. Phone uses Streamlit's drawer |

Test additions: rail structure and groups, pinning and state, keyboard-reachable buttons and no scripts, CSS contracts (widths, no layout shift, focus-visible, reduced motion, mobile), page footer and status components, the typography scale, self-hosted font, cached builders, lazy tabs completeness, prefetch idempotence.

### Screenshots (27 images in `docs/screenshots/`) and what was inspected

Captured from the final build (full-page, after scrolling so lazy images are loaded). Viewed individually: landing at 1440 (two halves), 1280, 820 (top) and 390 (top); sign-in at 1440, 1280, 820 and 390; the rail compact, expanded by hover, with keyboard focus, and pinned; the mobile page and drawer at 390; overview at 1440, 1280 and 820; alerts, trends, forecasts, patient risk (form and result), hospital twin and what-if (with the emergency-surge scenario) at 1440; and the landing preview image. Not individually viewed in the final copies: the lower parts of the 820 and 390 landing pages and sign-in 390 after its last wording tweak, hospital twin at 1280, the Forecasts "How this is calculated" tab, and the 390 landing heading after its last size change (re-captured, checked only for overflow). The critical-state screenshots from an earlier stage were not recreated.

### Defects found in the visual inspection of this stage and fixed

* Rail: the bottom group (status, guide, pin, user, sign out) floated mid-rail and then covered the navigation (a block with `height:100%`); labels were clipped by the buttons under them (Streamlit's markdown wrapper has a `-1rem` bottom margin); the page header sat 24 px lower than the rail (an invisible stylesheet-injection element added a gap, and an inline-flex line box added 8 px); the native tooltip bubble covered the hover labels; **keyboard focus showed no ring** (the rail styles had removed Streamlit's); the menu button overlapped the heading on phones.
* The landing preview images still showed the old sidebar (they are static files); regenerated from the new build.
* Forecasts had three controls named "How this is calculated"; two were renamed.

### Remaining issues

* First visit to Forecasts is about 2.4 s; first workspace load is about 0.4 s longer than before (prefetch); Trends repeat visits are about 0.9 s (four charts render in the browser).
* The rail is compact on tablets (no hover on touch): tap Keep menu open for labels; phones use Streamlit's drawer.
* The Alerts page and What-If button still say "What-If Simulator" while the navigation says "What-If".
* Only Edge was tested; Inter covers Latin text only (other scripts fall back to the system font); the Streamlit direct-load `404` pair remains.
* Demo access is still not production authentication.

---

## Earlier stage: redesign verification and notes

## What changed in the product redesign

* **Journey:** public landing page (`/`), sign-in / create account / demo user (`/signin`), protected workspace (`/workspace`), sign out. Routing uses Streamlit's own `st.navigation` with hidden navigation and real URLs.
* **Medical Sage light theme** across landing page and dashboard: deep sage `#145A55`, teal `#2E7D73`, soft sage `#88BFA3`, background `#F2F7F5`, white surfaces, warm neutral `#E8E2D6`, text `#142B2A`.
  Secondary text is `#566B68` rather than the suggested `#637775` (which is 4.4:1 on the page background, below the 4.5:1 WCAG AA target; the replacement is 5.2:1). Tests check contrast on every surface.
* **Landing page:** top navigation, hero with an isometric hospital and a floating "digital twin" layer, three benefits, six features, three steps, a tabbed preview made from real screenshots, about and disclaimer, final call to action, footer.
  The animated network and the illustration are inline SVG/CSS (no images, no script, no new dependency); animation stops under `prefers-reduced-motion`. The illustration contains no numbers, and the page makes no performance or customer claims.
* **Plain-language labels** (the internal keys `observed`, `ml`, `rule` are unchanged):

  | Before | Now |
  |---|---|
  | Observed | Recorded |
  | ML-predicted | Expected |
  | Rule-derived | Estimated |
  | Anomaly / Anomalies | Unusual activity |
  | Alerts & recommendations | Alerts |
  | Operations & analytics | Trends |
  | Predictions | Forecasts |
  | Digital twin | Hospital twin |

  Navigation: **Today** (Overview, Alerts, Trends), **Look ahead** (Forecasts, Patient Risk), **Plan** (Hospital Twin, What-If).
* **"How this is calculated"** holds model names, metrics, baselines, limitations and notes: an expander on Forecasts (chart, comparison tables), Patient Risk (result and reliability), Hospital Twin, What-If and Trends > Patient risk,
  and the last tab of Forecasts. Nothing technical was removed, it was moved. The Forecasts tab "How this is calculated" still lists XGBoost, Huber regression, Isolation Forest, MAE/RMSE/R² and the Prophet note.
  The honest statements stay on the main screens in plain words: the trained forecast is about as good as a weekday average, the likely range held on 70% of test days, and the risk model misses nearly half of High-risk records.
* **Backend untouched.** 39 data, model and backend-source files (`data/**`, `models/**`, `src/twin`, `src/simulation`, `src/alerts`, `src/recommendations`, `src/models`) have checksums identical to those recorded before the work,
  and `tests/test_backend_snapshot.py` compares twin, What-If, alert and patient-risk outputs with values recorded before the redesign.

## Demo access: what it is and is not

Implemented in `src/ui/auth.py`. **It is demonstration access for an educational prototype, not production authentication.**

* Accounts exist only in the memory of the running server process; they are not written anywhere and disappear on restart.
* Access lasts for one browser session. Refreshing the page or opening another tab asks you to sign in again.
* Passwords are hashed with PBKDF2-HMAC-SHA256 and a random salt (standard library), so memory never holds a plain-text password. That is hygiene, not security: there is no rate limiting, e-mail verification, password reset, TLS handling or audit log.
* The gate is a convenience in the user interface. Anyone who can read the source can bypass it. Do not enter a real password; "Continue as demo user" needs none.
* A production system would use an identity provider (Streamlit `st.login` with OIDC) over HTTPS. It was not implemented: it needs `Authlib`, an external provider and secrets.

## Verification of the earlier redesign stage (superseded by the table above for test counts)

| Check | Result |
|---|---|
| Automated tests | **386 pass, 0 skipped** (327 before this stage) |
| New tests (net +59) | `test_auth` 15, `test_navigation_gate` 18, `test_landing` 22, `test_backend_snapshot` 1, 3 more theme tests in `test_ui_components` |
| Browser journey G (headless Edge, real server): landing, in-page links, preview images, signed-out deep link and direct `/workspace` blocked, Get Started, mismatched and valid account creation, all 7 pages, Forecasts "How this is calculated", Patient Risk assessment, What-If scenario with its alert and recommendation, the Alerts page scenario section, the Recommendations tab, Clear scenario, sign out, protection again, demo user from `?view=twin` | **32 / 32 pass** (final build) |
| Hover and focus states, measured with real mouse hover (computed styles before and after) on landing buttons, link, nav and icon tile, sign-in buttons, sidebar item, KPI cell, secondary button, card | **11 / 11 change as designed** |
| Browser console / network | 2 console messages, 0 failed requests. Both messages are `404` for `/<route>/_stcore/health` and `/host-config` when the page is loaded **directly** at `/signin` or `/workspace`. **Investigated:** a minimal two-page Streamlit app with none of this project's code logs the identical pair for `/other`; the client first probes the route as a base path (404), retries at the root (200) and connects its websocket there. No effect on functionality (pages render, websocket connects, every journey passes). Not suppressed; none occur during in-app navigation or on `/` |
| Backend regression | checksums of 39 files identical; snapshot test passes |
| Responsive checks (no horizontal page scroll) | landing, sign-in and overview at 1440 / 1280 / 820 / 390, twin at 1280 and 820 (at 820 the wide comparison table scrolls inside its card) |

### Test assertions that were changed, and why

No assertion about a number, a calculation or a behaviour was weakened. These expectations were outdated by the redesign and were updated:

* `tests/test_ui_components.py`: palette token test uses the new `SIDEBAR` / `PRIMARY` tokens; badge text (`Observed`, `ML-predicted`, `ML`, `Rule-derived` became `Recorded`, `Expected`, `Estimated`) in three tests; chart trace names (`Likely range (empirical)` became `Likely range`, `OPD forecast (ML)` became `Expected outpatients`);
  the dark-theme accessibility class became a light-theme class: the WCAG AA checks stayed, "theme is dark" became "theme is light" with the new palette values, the Streamlit config check expects `base = "light"`, and "no leftover light tokens" became "no leftover dark tokens".
* `tests/test_ui_app.py`: view titles (`Trends`, `Forecasts`, `Hospital twin`, `What-if scenarios`, `Alerts`); sidebar button count now counts the seven navigation buttons (a Sign out button was added); twin explanation label and source labels; KPI source labels.
* `tests/test_ui_premium.py`: navigation group labels and glossary term names.
* `tests/test_ui_prediction_views.py`: `Rule-derived` became `Estimated`.
* `run_app` helpers in `test_ui_app.py`, `test_ui_prediction_views.py` and `test_alert_reachability.py`: one line seeds the demo user, because the workspace is now a protected route.

### Screenshots (21 images in `docs/screenshots/`) and what was actually inspected

Captured from the final build by a script that scrolls the page first, so lazy-loaded images are present and the full height is captured. (An earlier capture script measured the page height before the preview image loaded and cut the landing page short, missing the call to action and footer; this was found during the final audit and the images were regenerated.)

Viewed individually in the final audit: landing at 1440 (whole page in two halves, the preview area after its fix and the final bottom section), 1280, 820 (three sections plus the area changed afterwards) and 390 (three sections); sign-in at 1440, 1280, 820 and 390; overview at 1440 and 820 (and the KPI area after the 820 fix); alerts, trends, forecasts, patient risk (form, result and the area after its fix), hospital twin and what-if at 1440.
Viewed on the build immediately before a final small CSS change and **not re-viewed** in the regenerated copies: overview 1280, hospital twin 1280, the Forecasts "How this is calculated" tab and the what-if emergency-surge scenario.
Not viewed: the 1280 sign-in and landing regenerated copies are identical in layout to the 1440 ones (overflow-tested only in the regenerated set).
The four landing preview images in `static/` are crops of the finished workspace, viewed in place on the landing page. The critical-state screenshots from the earlier stage (a stressed state injected in memory) were **not** recreated.

### What the visual inspection found and fixed

* Button labels were dark on dark-green buttons (a global paragraph colour rule overrode them); fixed, including the white-on-white and green-on-green buttons in the final call to action.
* "View Demo" showed default blue underlined link styling.
* The first background network was too sparse and long-lined; it is now denser and fainter.
* Narrow screens stacked "Sign in" above a full-width "Get Started" in the top bar; the bar now stays on one row.
* A visible patient-risk sentence named the model (XGBoost) and the Trends page had a "Model: XGBoost" tile; both moved under "How this is calculated".
* Final audit: the landing preview screenshot was cut off abruptly mid-card (now fades out), the caption sat tight against the demo button, the final call-to-action heading left "prototype" alone on a line, the nav "Sign in" sat a few pixels off the link baseline, the tablet hero illustration was oversized and the six features formed a very long single column (now capped, two columns at tablet), the fifth KPI tile sat alone in a half-empty row at 820 px (now spans the row), the two Patient Risk result cards had different heights and expanders touched the text above them, and the sign-in legal note was tight to the card edge.

## Limitations of this stage

* Demo access only (above). No persistence, no password reset, no e-mail, no roles.
* `st.switch_page` is not followed by Streamlit's `AppTest`, so unit tests cover pages and gates in isolation and the real page change is covered by the browser journey only.
* Loading `/signin` or `/workspace` directly logs two `404` messages from Streamlit's own client (see the verification table: reproduced in a minimal Streamlit app); the app still loads and works. They were not suppressed.
* Phone layout is a basic fallback: the landing page and sign-in stack correctly; the dashboard drawer overlays content. Only Edge was tested; not tested: Safari, Firefox, high-DPI scaling, screen readers, print.
* The earlier browser journeys A to F were not re-run against the new interface (they predate the sign-in gate); journey G and the AppTest suite cover the same pages. Journey G does not exercise the Analytics date filters, the Alerts severity/timing filters on a stressed state, or invalid-input handling (covered by AppTest only).
* On a phone the landing preview screenshot is too small to read; it is an illustration there.
* The app loads the Inter font from Google Fonts (as before); offline it falls back to the system font.
* Everything about the data and models is unchanged: synthetic data, OPD forecast about equal to a weekday average with a 70% (not 80%) likely range, patient-risk High recall 54%, anomaly evidence from four spikes, rule-based resource projections.

## Git and artifact hygiene findings

Nothing was committed, pushed, staged, reset or discarded by this work; `stash@{0}` is intact; no backups were removed. The index already held 16 staged changes from earlier work and is unchanged.

| Finding | Detail | Action taken / recommendation |
|---|---|---|
| New required files are untracked | `src/ui/` (now including `pages/`, `auth.py`, `router.py`, `workspace.py`, `landing_assets.py`), `static/` (4 preview PNGs, required by the landing page), `tests/` (new files), `docs/`, `.streamlit/config.toml` (now light theme and static serving), `.gitignore`, `requirements-lock.txt`, `models/opd_forecast/forecaster.joblib` | They must be added when committing or a fresh clone will not run |
| Tracked cache files | about 37 `__pycache__/*.pyc` files are tracked (a `.gitignore` does not untrack them) | Recommend `git rm --cached` for them (not done) |
| `models/opd_forecast/prophet_model.joblib` | Tracked in `HEAD`, deleted in the working tree, modified in the index. No code, test or document loads it; it was trained on the previous data and cannot be rebuilt here | Recommended: keep it deleted and stage the deletion with `git rm --cached`. Recover the committed copy with `git checkout HEAD -- <path>` (not `git checkout --`, which restores the stale staged copy). It was not restored |
| Four old log files in the repository root | `train_all_log.txt`, `train_all_out.txt`, `patient_risk_train_out.txt`, `prophet_install_log.txt` are tracked, dated 1 Oct, UTF-16, contain a personal Windows profile path and old metrics that contradict the model cards; nothing references them | Safe to remove; not removed (your files) |
| Old screenshots | The 25 dark-theme screenshots from the previous stage were replaced; the set was regenerated again in the final audit (21 images) | The old images were copied to the session scratchpad first (not durable) |
| Backups | Session backups of the old `data/`, `models/`, `notebooks/` exist only in a temporary scratchpad | Not durable; the previous committed versions are in Git history |

## Files changed in this stage

* **New:** `src/ui/auth.py`, `src/ui/router.py`, `src/ui/workspace.py`, `src/ui/landing_assets.py`, `src/ui/pages/{__init__,common,landing,signin}.py`, `static/preview-*.png` (4), `tests/test_auth.py`, `tests/test_navigation_gate.py`, `tests/test_landing.py`, `tests/test_backend_snapshot.py`, `tests/fixtures/backend_snapshot.json`.
* **Modified:** `app.py` (now only the router), `.streamlit/config.toml`, `src/ui/theme.py` (rewritten: light tokens, CSS, landing CSS), `src/ui/components.py` (logo, user block, label text), `src/ui/charts.py` (trace names and colours), `src/ui/data.py` (glossary and label text), `src/ui/views/{__init__,alerts,analytics,overview,patient_risk,predictions,twin,whatif}.py` (labels, titles, "How this is calculated"), the three test helpers and the test files listed above, `README.md`, this file, `DEMO_WALKTHROUGH.md`, `PROJECT_STATUS.md`, `docs/screenshots/`.
* **Not changed:** data, models, ML code, Digital Twin, What-If, alert and recommendation logic, `backend.py`, `requirements*.txt`. No new dependency was added.
