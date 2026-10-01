# MediTwin AI

**Intelligent Hospital Digital Twin & Predictive Healthcare Platform**

University Data Science + AI/ML project (educational prototype).

## Disclaimer

All datasets and risk labels in this repository are **synthetic** and created
for academic learning only. MediTwin AI is **not** a real clinical decision
system and must **not** be used for diagnosis, treatment, triage, or live
hospital operations.

## Project structure (Phase 1)

```
meditwin-ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── twin/
│   ├── simulation/
│   ├── recommendations/
│   └── utils/
├── models/
└── tests/
```

## Setup

Use Python 3.11+.

```bash
cd meditwin-ai
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate Phase 1 synthetic data

From the `meditwin-ai` project root:

```bash
python -m src.data.generate_synthetic_data
```

This creates:

- `data/synthetic/hospital_operations.csv` — daily hospital operations (≥ 1 year)
- `data/synthetic/patient_data.csv` — patient-level records with Low/Medium/High risk labels

Optional arguments:

```bash
python -m src.data.generate_synthetic_data --n-patients 1000 --seed 42
```

## Run the Streamlit landing page

```bash
streamlit run app.py
```

## Synthetic risk labelling (educational)

Patient `risk_level` values are assigned with transparent rule-based scoring
(age, vitals, chronic disease, emergency flag, previous admissions). These
rules are simplified for teaching and reproducibility — **not** medical guidelines.

## Current phase

**Phase 4:** prediction service + hospital digital twin.

```bash
python -m unittest tests.test_hospital_twin tests.test_prediction_service
```

```python
from src.models.prediction_service import get_prediction_service
from src.twin.hospital_twin import build_twin_from_latest

service = get_prediction_service()  # loads Joblib models once
twin = build_twin_from_latest()
print(twin.get_current_state())
print(twin.get_capacity_status())
print(twin.get_predicted_state(days_ahead=1))
```

The twin is a **data-driven software representation** of hospital state from
the dataset and ML outputs — not a physical IoT-connected hospital.
