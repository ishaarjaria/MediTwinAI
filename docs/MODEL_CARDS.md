# Model cards

Every figure below is copied from the saved `models/*/metrics.json` files produced by the training
scripts, and the numbers are re-checked by `tests/test_model_training.py`. All models are trained on the
**synthetic** data described in `DATA_DICTIONARY.md`; scores say how well a model learns the generator,
not how it would perform in a real hospital.

Retrain everything (data must be generated first, see the README run order):

```bash
python -m src.models.train_all                 # all four models
python -m src.models.train_all --skip patient_risk   # keep the patient-risk model as it is
```

## What is a trained model and what is a rule

| Output | Source | Label in the app |
|---|---|---|
| Patient risk class and probabilities | trained classifier | ML-predicted |
| OPD (outpatient) forecast | trained forecaster | ML-predicted |
| Waiting time | trained regressor | ML-predicted |
| Unusual-day flag | trained Isolation Forest | ML-predicted |
| Beds, ICU, emergency cases, medicine stock and consumption, staff workload in the twin and What-If | fixed formulas copied from the data generator or simple heuristics | Rule-derived |
| Alerts | threshold ladders on observed, ML or rule values | basis shown per alert |
| Recommendations | fixed rules, advisory only | Rule-derived |

**No trained model exists for bed, ICU, medicine or staffing demand.** These are rule-derived and labelled so.
Training such models was left out deliberately: the generator makes them near-deterministic functions of
admissions and emergency volume, so a model would only re-learn the generator's formulas.

## 1. OPD forecast (`models/opd_forecast/`)

* **Target:** `opd_patients`. **Protocol:** chronological train (first 278 usable days, 28 days are only lag history),
  validation (2 Nov to 1 Dec 2024), test (2 to 31 Dec 2024). Each evaluated day is scored as a forecast of at most 7 days ahead
  (all features use lags of at least 7 days). **Selection:** lowest validation MAE; the test block is reported once.
* **Deployed model:** XGBoost on weekday, weekend, holiday, month and lagged-OPD features (lags 7, 14, 21, 28 days), refit on all data.
  Forecasts beyond 7 days are recursive (predictions feed later lags), so they are less reliable.

| Model | Kind | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Training mean | baseline | 63.28 | 76.48 | 84.27 | -0.027 |
| Day-of-week mean | baseline | 27.61 | 34.00 | 52.32 | 0.604 |
| Previous week, same day | baseline | 37.27 | 41.57 | 65.94 | 0.371 |
| Same weekday, last 4 weeks average | baseline | 31.98 | 37.51 | 57.42 | 0.523 |
| Ridge (calendar + seasonal lags) | model | 27.58 | **30.72** | 37.93 | 0.792 |
| **XGBoost (calendar + seasonal lags)** (selected) | model | **26.30** | 34.04 | 41.64 | 0.749 |
| Prophet (weekly + calendar) | model | not evaluated on this machine | | | |

**Honest reading.** XGBoost won on validation (by 1.3 patients per day) and was therefore selected, but on the later test block it
is no better than the day-of-week mean (34.04 vs 34.00 MAE) and Ridge is better. The two trained models and the simple weekday average
are within noise on a 30-day test block. The trained models do clearly beat the training mean and the naive same-day-last-week baseline.

* **Prophet.** An earlier run of Prophet on the previous version of the data scored worse than the training mean (R² -0.32).
  The Prophet candidate is kept in the training script but could not run when the models were retrained, because the Windows
  Application Control policy on this machine blocked the Stan executable (`WinError 4551`). It is recorded as *skipped* with that
  reason rather than reported. If Prophet runs on your machine it is evaluated and can be selected by the same rule.
* **Interval.** The band is `forecast +/- 45.7 patients`, the 80th percentile of validation absolute errors. It contained the
  actual value on 80% of validation days (by construction) but only **70% of test days**, so it is called an *empirical likely range*
  and is **not** described as a validated 80% interval (`validated_80pct: false`).
* **Old claim withdrawn:** the previous model card reported an "80% interval" that covered 17% of holdout days.

## 2. Waiting time (`models/waiting_time/`)

* **Target:** `average_waiting_time_minutes`. **Features:** OPD, emergency cases, doctors, nurses, weekend, holiday, day of week, month,
  bed occupancy, admissions. **Protocol:** chronological 220 train / 73 validation / 73 test days (test from 20 Oct 2024). Selection by lowest
  validation MAE (then RMSE); test reported once. The final model is refit on all data.
* **Selection rule changed during this work, and why:** it was previously lowest RMSE. The validation block contains one injected
  OPD-rush spike (19 Sep) whose squared error dominated RMSE and favoured ordinary least squares, which then did worse than a
  one-variable baseline on the test block. MAE is less dominated by a single day. This change was made after seeing that result,
  so treat the ranking as a reasoned choice, not an untouched blind selection.

| Model | Kind | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Training mean | baseline | | 8.28 | 10.59 | -0.121 |
| OPD-only linear | baseline | | 4.60 | 5.61 | 0.685 |
| Day-of-week mean | baseline | | 6.34 | 8.06 | 0.351 |
| Linear regression | model | 4.64 | 4.74 | 5.80 | 0.664 |
| **Huber regression** (selected) | model | **3.96** | 3.91 | **4.71** | 0.778 |
| Random forest | model | 5.13 | 3.96 | 4.70 | 0.779 |
| XGBoost | model | 5.02 | 3.83 | 4.80 | 0.770 |

The selected model beats the best simple baseline (OPD-only linear, RMSE 5.61). Ordinary linear regression is hurt by the five injected spike days
(multiplied waiting times act as outliers), which is why a robust linear model was added. On this synthetic data the waiting time is built
from OPD, emergency volume, occupancy and doctors, so a good score means the model learned that formula. The model has no ICU-capacity input,
so changing ICU capacity does not change waiting time in the What-If view.

## 3. Patient risk (`models/patient_risk/`) (unchanged)

* **Target:** `risk_level` (Low / Medium / High), a **deterministic rule** of the inputs, not a clinical label. Random stratified 80/20 split (200 test records).
* Logistic regression, random forest and XGBoost compared; XGBoost selected by macro-F1: accuracy 84.5%, macro-F1 0.807, **High-class recall 54.2%**
  (it misses nearly half of High-risk records). A majority-class guess scores about 51.5% accuracy.
* The service validates ranges, gender, flags and `systolic > diastolic`, and returns class probabilities (sum to 1).
* The Patient Risk page in the dashboard is an educational demonstration, **not a medical diagnosis**.

## 4. Anomaly detection (`models/anomaly_detection/`)

* **Model:** Isolation Forest on OPD, emergency cases, bed %, ICU % and waiting time. Unsupervised.
* **Evaluation (new):** forward chaining. Four blocks from day 60 onwards are each scored by a detector fitted only on earlier days, against the five
  injected spikes. The 23 Jan spike has no earlier history and is excluded, leaving 4 scored spikes: a **very small sample**.
* **Contamination** (expected share of unusual days) was chosen by pooled F1 on that evaluation: 0.02.

| Detector | Contamination | Precision | Recall | F1 | Days flagged (of 306) |
|---|---|---|---|---|---|
| Isolation Forest | 0.01 | 1.00 | 0.25 | 0.40 | 1 |
| **Isolation Forest** (deployed) | **0.02** | 0.67 | 0.50 | **0.57** | 3 |
| Isolation Forest | 0.03 | 0.14 | 0.75 | 0.24 | 21 |
| Isolation Forest | 0.05 | 0.07 | 1.00 | 0.13 | 59 |
| Local Outlier Factor | 0.01 | 1.00 | 0.75 | 0.86 | 3 |
| Local Outlier Factor | 0.02 | 0.11 | 0.75 | 0.19 | 27 |

LOF scored higher at its best setting, but both are based on 3 to 4 hits; Isolation Forest is the integrated detector. Real pressure periods (not injected
spikes) can also be flagged and count as false positives in this evaluation even though they are genuinely unusual. The January warm-up artifact of the old
generator no longer exists (at most 1 flag in the first week). The saved `scored_operations.csv` is an in-sample display table; use the metrics above for accuracy.

## Reproducibility notes

* Seeds are fixed (42). Package versions used are listed in `requirements-lock.txt`; results can differ slightly with other versions.
* The waiting-time, OPD and anomaly `metrics.json` files store file names, not machine paths. The patient-risk file (not retrained) still contains the absolute path from its earlier training run, and so does the combined summary that copies it.
* The first 28 days only supply lag history for the forecaster.
