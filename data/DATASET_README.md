# Dataset documentation

## Selected source
NASA Ames Prognostics Center of Excellence (PCoE), **Li-ion Battery Aging / Battery Data Set** (the four-cell B0005, B0006, B0007 and B0018 MATLAB release).

- Official NASA repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA Open Data description: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
- Direct archive: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- Dataset citation: B. Saha and C. Goebel (2007), *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- Access: public research dataset. NASA's catalog marks the related data resource public. Check the archive's own notices and NASA data-use guidance before redistribution; cite NASA and the original authors. General NASA data guidance: https://science.data.nasa.gov/about/license.

## Data and units
The MATLAB records contain cycle type and nested measurement data. Discharge records include measured voltage (V), measured current (A), measured temperature (°C), time series, and discharge capacity (Ah). Charge and impedance cycles are present but the baseline implementation extracts discharge cycles only. Each output row is one discharge cycle summarized into mean/minimum/maximum/standard-deviation features.

## Target definition
`soh_percent = discharge_capacity_ah / 2.0 Ah * 100`. The nominal 2.0 Ah rating is used as the reference for this project. SOH is an estimate of retained capacity; SOC is remaining charge and is not predicted here. Values are not clipped during target construction. The cleaner excludes physically impossible target values outside 0–120% but does not remove ordinary unusual observations.

## Preprocessing
The loader parses MATLAB structures with SciPy, selects discharge cycles, drops non-finite samples within each cycle, summarizes time-series readings, and skips malformed cycles. The ML pipeline imputes missing numeric features with training-set medians and standardizes them. The split holds out a battery identity when possible to reduce within-cell leakage. No scaler or imputer is fit on test data.

## Feature fields
`cycle_index`; voltage mean/min/max/std (V); current mean/min/max/std (A); temperature mean/max/std (°C). Capacity is used only to derive the target and is intentionally excluded from model inputs to prevent direct target leakage.

## Limitations
The classic release contains only a small number of cells under controlled accelerated aging conditions; temperature variation is limited. It is not representative of every chemistry, pack design, environment, or field application. The cycle summaries are not a real-time sensor stream. Battery-level holdout results can vary greatly with the chosen held-out cell. Health classes (Healthy >=80%, Moderate 60–<80%, Degraded <60%) are project-defined categories, not universal safety thresholds. The NASA dataset may stop around its defined end-of-life criterion, so the Degraded class may be absent; the training script reports this rather than fabricating examples.

## Obtaining the data
Run `python scripts/download_data.py`. If network access fails, manually download the official ZIP from the direct archive link and place it in `data/raw/NASA_Battery_Data_Set.zip`; then run `python scripts/train_all.py`. Raw data is ignored by Git because of size and redistribution considerations.
