# Data dictionary

All data in this project is **synthetic**: produced by `src/data/generate_synthetic_data.py`
with a fixed seed (42). Nothing comes from a real hospital or real patients.

## Dataset decision

The application, the model training, the digital twin, the What-If simulator and the
dashboard use **one** primary source: the synthetic *daily hospital-operations* dataset below
(plus the synthetic *patient* dataset for risk classification).

The four CSVs used in the project's first report (`patients.csv`, `services_weekly.csv`,
`staff.csv`, `staff_schedule.csv`: weekly, per-service data) have a different time
granularity and structure. They are **not** merged into this pipeline and are **not used by the
application**. They may be kept as separate, earlier exploratory material. Any statement that the
app "uses" those files would be wrong.

Because the dataset is daily and hospital-wide, the app has **no per-service breakdown and no
"refused requests" or per-service bed fields**.

## Files

| File | Rows | Produced by | Used for |
|---|---|---|---|
| `data/synthetic/hospital_operations.csv` | 366 (2024-01-01 to 2024-12-31) | `generate_synthetic_data` | raw daily operations |
| `data/synthetic/patient_data.csv` | 1,000 | `generate_synthetic_data` | raw patient records |
| `data/synthetic/injected_anomalies.csv` | 5 | `generate_synthetic_data` | ground truth for anomaly *evaluation* only |
| `data/processed/hospital_operations_clean.csv` | 366 | `src.data.preprocess` | cleaned operations |
| `data/processed/hospital_operations_features.csv` | 366 | `src.features.feature_engineering` | models, twin, dashboard |
| `data/processed/patient_data_clean.csv` / `_features.csv` | 1,000 | preprocess / features | risk model |
| `data/raw/` | empty | | reserved; no public dataset is used |

## Hospital operations columns

| Column | Meaning | How it is produced |
|---|---|---|
| `date` | calendar day | daily range |
| `day_of_week` | 0 = Monday ... 6 = Sunday | calendar |
| `is_weekend`, `is_holiday` | 0/1 flags | calendar; 9 synthetic holidays in `HOLIDAY_DATES` |
| `season` | Winter / Spring / Monsoon / Autumn | month |
| `opd_patients` | outpatient visits that day | seasonal base x weekend x holiday x weekly wave x pressure period x persistent demand shock, plus noise |
| `emergency_patients` | emergency arrivals | same structure, own base |
| `admissions` | patients admitted | `0.35*emergency + 0.04*OPD + noise`, scaled in pressure periods |
| `discharges` | patients discharged | about 27% of occupied beds on weekdays, 19% at weekends (length of stay), plus noise |
| `total_beds` | 250 (constant) | |
| `occupied_beds` | beds in use | stock-and-flow: `previous + admissions - discharges`, capped at capacity |
| `available_beds`, `bed_occupancy_percentage` | derived | `total - occupied`, `occupied / total` |
| `total_icu_beds` | 30 (constant) | |
| `occupied_icu_beds`, `icu_occupancy_percentage` | ICU use | `0.17*emergency + 0.045*occupied_beds + noise`, capped |
| `doctors_available`, `nurses_available` | staff on duty | normal around 45 / 120, lower on weekends/holidays and during the staff-sickness period |
| `average_waiting_time_minutes` | mean OPD wait | `0.12*OPD + 0.12*emergency + 0.05*bed_occupancy% - 0.35*doctors + 2 + noise` (spike days multiplied) |
| `medicine_stock_units` | stock at day end | opening 16,000; consumed daily; reorder at or below 10,500 for 10,000 units with 3 to 5 day lead time (+9 days in the supplier-delay window) |
| `medicine_consumption_units` | units used that day | `0.8*OPD + 2.5*emergency + 1.2*admissions + noise` (spike days multiplied) |
| `equipment_availability_percentage` | | falls with occupancy; **not used by any model, twin, alert or view** |

### Documented high-demand and resource-pressure periods

| Period | Dates | Effect |
|---|---|---|
| Winter respiratory wave | 15 Jan to 12 Feb | OPD x1.10, emergency x1.30, admissions x1.08 |
| Monsoon dengue outbreak | 12 Aug to 8 Sep | OPD x1.15, emergency x1.35, admissions x1.10 |
| Festival accident / burn surge | 30 Oct to 4 Nov | emergency x1.40, admissions x1.10 |
| Outbreak staff sickness | 19 to 30 Aug | doctors and nurses x0.80 |
| Supplier delay | 25 Aug to 10 Sep | orders placed arrive 9 days late |
| Five injected spike days | chosen by seed (23 Jan, 4 Mar, 15 May, 20 May, 19 Sep) | `emergency_surge`, `opd_rush` or `icu_pressure` |

Result on the saved data: bed occupancy ranges from 46.8% to 97.6% (70% or more on 68 days, 90% or more on 15),
ICU reaches 70% or more on 66 days, and medicine days of cover (stock divided by that day's use) range from 8.5 to 91 with a median of 32.5 (below 21 days on 38 days).
Most days are normal; pressure is concentrated in the documented periods. Occupancy has **no floor drift and no
warm-up artifact** (the earlier version pinned 16% of days at a lower clip and started with a collapse).

## Engineered features (`hospital_operations_features.csv`)

`month`, `quarter`, `is_weekday`, `daily_patient_count`, `total_patient_load`, `admission_rate`,
`discharge_rate`, `medicine_consumption_rate`, `staff_to_patient_ratio`, `emergency_ratio`,
`lag_1`, `lag_7`, `rolling_7`, `rolling_14` (OPD lags/rolling means use past values only), `peak_load_indicator`.
The first rows have empty lag values by construction. Leakage checks are in
`validate_no_obvious_leakage` and tested.

## Patient columns

`patient_id`, `age`, `gender` (Male/Female/Other), `heart_rate`, `systolic_bp`, `diastolic_bp`, `temperature`,
`oxygen_saturation`, `chronic_disease` (Yes/No), `emergency_case` (Yes/No), `previous_admissions`, `risk_level`.
Engineered: `pulse_pressure`, `age_group`, `chronic_disease_flag`, `emergency_case_flag`.

**`risk_level` is a deterministic function of the other columns** (`assign_risk_level`; points for age, vitals,
chronic disease, emergency flag and previous admissions; 7 or more is High, 4 to 6 Medium). It is not a clinical label.
The proposal also mentioned symptoms, diagnosis, medical history and admission/discharge dates; these fields are **not**
present in this synthetic dataset.

## Not present

Weather, per-service data, refused patients, real patient or hospital records, live sensor feeds.
