# Demo walkthrough (about 8 minutes)

MediTwin AI is an **educational prototype on synthetic data**. Say this at the start: nothing here is connected to a real hospital, sensor or patient record,
and the sign-in is demonstration access, not real account security.

Start the app (`streamlit run app.py`, then open http://localhost:8501). Screenshots referenced below are in [`screenshots/`](screenshots/).

| Step | Page | What to show | What to say |
|---|---|---|---|
| 1 | **Landing page** ([01](screenshots/01-landing-1440.png)) | Scroll from the hero through the six features and the three steps. Open the preview tabs (Overview, Forecasts, Hospital Twin, What-If). Point at the disclaimer band. | "A hospital administrator's workspace: see today, expect the coming days, test decisions. Everything is on synthetic data." The preview images are screenshots of the real prototype. |
| 2 | **Sign in** ([02](screenshots/02-sign-in-1440.png)) | Press **Get Started**. Create an account with any made-up name and a throwaway password (or press **Continue as demo user**). Mention the amber notice. | "This is demo access: accounts live in server memory for one browser session, there is no e-mail or password reset. A real system would use an identity provider." |
| 3 | **The navigation rail** ([compact](screenshots/03-navigation-rail-compact.png), [expanded](screenshots/03b-navigation-rail-expanded-on-hover.png), [pinned](screenshots/03d-navigation-rail-pinned-open.png)) | Hover the slim rail on the left: the labels appear without moving the page. Press Tab to show keyboard use ([focus ring](screenshots/03c-navigation-rail-keyboard-focus.png)). Press **Keep menu open**, then **Auto-hide menu**. Open the **Plain-language guide**. | "Icons when compact, labels on hover or keyboard focus. Pages are grouped Today, Look ahead and Plan." |
| 4 | **Overview** ([04](screenshots/04-overview-1440.png)) | The status band (bed and ICU rings), the KPI strip with 30-day trend lines, the "Needs attention" panel. Hover an "i" icon. | "Every number is marked Recorded, Expected or Estimated, so you always know where it comes from." The waiting time shows today's *expected* value beside the recorded one so the comparison is fair. |
| 5 | **Trends** ([06](screenshots/06-trends-1440.png)) | Switch the period to *Last 90 days*, open each tab, including *Unusual activity*. | "Weekends are half as busy for outpatients, and the August outbreak is visible in the bed and ICU curves. 'Unusual' means different from normal, not harmful." |
| 6 | **Forecasts** ([07](screenshots/07-forecasts-1440.png), [07b](screenshots/07b-forecasts-how-this-is-calculated-1440.png)) | The headline strip, the forecast with its likely range. Open **How this is calculated** (the expander, then the last tab) for model names, baselines and metrics. | Be candid: "The forecast is about as good as a weekday average on this synthetic data, and the likely range held on 70% of test days, not 80%." Mention Prophet could not run on this machine. |
| 7 | **Patient Risk** ([08b](screenshots/08b-patient-risk-result-1440.png)) | Pick *Several warning signs*, press *Assess risk*; show probabilities and the warning-sign table. Enter systolic 80 and diastolic 90 to show the friendly error. | "A demonstration, not a diagnosis. Its labels come from fixed rules, and it misses nearly half of high-risk records." |
| 8 | **Hospital Twin** ([09](screenshots/09-hospital-twin-1440.png)) | Change the look-ahead from 1 to 7 days. Open "What do Recorded, Expected and Estimated mean?". | "Only outpatients and waiting time are expected values from trained models. Beds, ICU and stock are simple estimates, so they barely move." |
| 9 | **What-If** ([10](screenshots/10-what-if-1440.png), [10b](screenshots/10b-what-if-emergency-surge-1440.png)) | Click *Emergency surge +50%*, read the change chips and the table; then *Delivery delay (stock -60%)*. Show the alert and recommendation under the scenario. | "The same model scores both sides, so *no change* shows exactly zero difference." |
| 10 | **Alerts** ([05](screenshots/05-alerts-1440.png)) | The real state is all clear; scroll to *What-if scenario alerts* (clearly labelled hypothetical) and press *Clear scenario*. | "Real alerts and hypothetical ones are never mixed." |
| 11 | **Sign out** | Press **Sign out** at the bottom of the rail, then open `/?view=alerts` again. | "The dashboard is protected: a signed-out visitor goes to sign-in first." Say again that this is demo access only. |

Other widths: [overview 1280](screenshots/04b-overview-1280.png), [overview 820](screenshots/04c-overview-820.png), [twin 1280](screenshots/09b-hospital-twin-1280.png),
[landing 1280](screenshots/01b-landing-1280.png), [landing 820](screenshots/01c-landing-820.png), [landing 390](screenshots/01d-landing-390.png),
[sign-in 1280](screenshots/02b-sign-in-1280.png), [820](screenshots/02c-sign-in-820.png), [390](screenshots/02d-sign-in-390.png),
[phone page](screenshots/03e-mobile-390-page.png) and [phone navigation drawer](screenshots/03f-mobile-390-navigation-drawer.png).
The dashboard is desktop-first; on a phone the landing page and sign-in stack and the dashboard uses Streamlit's navigation drawer.

## Questions to expect

* *Is this real hospital data?* No. All data is generated; see `DATA_DICTIONARY.md`.
* *Is the sign-in secure?* No. It is a classroom demo (in-memory accounts, one browser session). See "Demo access" in `RELEASE_NOTES.md`.
* *Why not use the four CSVs from the first report?* Different granularity (weekly, per service); the app is built on the daily dataset.
* *How good are the models?* See `MODEL_CARDS.md` and the "How this is calculated" sections: honest numbers, including where a model does not beat a baseline.
* *Can it run live?* No: static synthetic data, no database or API.
