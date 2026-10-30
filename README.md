# AI-Based Battery Health Monitoring System Using Machine Learning

A beginner-friendly capstone implementation that summarizes NASA PCoE lithium-ion battery discharge cycles, estimates capacity-based State of Health (SOH), classifies condition where labels exist, flags configured abnormal measurements, and serves predictions through Streamlit.

> **Important:** This is an academic software demonstration, not a certified battery management or safety system. Never use its predictions as a substitute for manufacturer limits, a BMS, protective circuitry, or qualified inspection. Demo warning thresholds must be replaced with the exact cell/pack specifications before any real-world interpretation.

## 1. Problem and objectives
Battery capacity fades with use. This project uses measured discharge-cycle data to estimate retained capacity and identify broad health categories. SOH = available discharge capacity / nominal rated capacity × 100. SOH is different from SOC (remaining charge).

## 2. Features
- NASA MATLAB dataset loader and capacity-derived SOH target.
- Cycle-level voltage/current/temperature feature summaries.
- Train-only imputation/scaling pipelines.
- Dummy, linear, tree, random forest and gradient boosting regression comparisons.
- Decision tree, random forest and gradient boosting classification comparisons (when supported by class distribution).
- Battery-group held-out evaluation; JSON metrics and test predictions.
- Saved Joblib pipelines, single-record and CSV batch prediction.
- Streamlit dashboard, warning rules, tests, notebook and project documentation.
- Hardware integration is optional and documented only as a future extension.

## 3. Dataset
NASA Ames PCoE Li-ion Battery Data Set. Source and citation details: [`data/DATASET_README.md`](data/DATASET_README.md). The raw ZIP is not included. Run `python scripts/download_data.py`, or download from https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip and place it in `data/raw/NASA_Battery_Data_Set.zip`.

## 4. Folder structure
```text
app.py
src/                 reusable loading, preprocessing, training and prediction modules
scripts/             download and train commands
data/                dataset notes; raw and processed data are git-ignored
notebooks/           EDA and model workflow notebook
models/              generated Joblib models and metadata (not committed)
outputs/             generated figures, metrics and reports
tests/               pytest tests
docs/                 report, architecture, methodology, viva and presentation
hardware/             optional hardware notes
```

## 5. Setup
Python 3.10+ is recommended.

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Download and train
```bash
python scripts/download_data.py
python scripts/train_all.py
```
The training script extracts MATLAB files if needed, writes cycle summaries to `data/processed/nasa_discharge_cycles.csv`, compares candidate estimators, saves the selected regression and classification pipelines under `models/`, and records test metrics under `outputs/metrics/`. It may skip classification if fewer than two classes are available. A group split holds out one battery when battery IDs exist. With only a few cells, the resulting test score is a small-sample estimate, not a universal performance guarantee.

If automatic download is blocked, follow the manual instructions in the dataset document. The repository intentionally contains no fabricated experimental records or prefilled metric scores.

## 7. Launch dashboard
```bash
streamlit run app.py
```
The dashboard requires a trained `models/regression_model.joblib`. The manual form accepts the exact 12 cycle-summary features listed in `src/config.py`. CSV upload requires the same feature columns; additional columns are preserved. The optional classification model is shown only when generated.

## 8. Notebook
Open `notebooks/battery_health_analysis.ipynb` in Jupyter or upload it to Google Colab. It loads the NASA data after setup, generates EDA figures, trains/evaluates models and explains the results. Notebook cells must be run in order after the dataset is present.

## 9. Tests
```bash
pytest -q
```
The 10 tests cover feature validation, cleaning, missing model behavior, health labels, anomaly rules and training mechanics. Two tests use artificial fixtures only to verify code execution; they are not research observations and do not validate model quality against the unavailable raw NASA data.

## 10. Input schema
Required columns and units:
- `cycle_index` (cycle number)
- `voltage_mean_v`, `voltage_min_v`, `voltage_max_v`, `voltage_std_v` (V)
- `current_mean_a`, `current_min_a`, `current_max_a`, `current_std_a` (A)
- `temperature_mean_c`, `temperature_max_c`, `temperature_std_c` (°C)

## 11. Limitations and future work
Only four cells in the classic dataset; controlled lab aging; limited temperature variation; potential distribution shift; capacity-derived labels; no live sensor link; demonstration thresholds. Future work: larger multi-chemistry datasets, chronological/group nested validation, uncertainty estimates, manufacturer-specific rules, calibration, validated pack sensors, and hardware-in-the-loop safety review.

## 12. References
1. Saha, B. and Goebel, C. (2007), NASA PCoE Battery Data Set.
2. NASA PCoE Data Set Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
3. NASA Open Data: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
4. NASA data license guidance: https://science.data.nasa.gov/about/license
