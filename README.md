# London Smart-Meter Forecasting

This repository is a compact public package for the Predictive Analytics Review-II submission on daily smart-meter energy forecasting.

## Student Details

- Name: Rohith Ganesh Kanchi

## Project Title

Energy Consumption Forecasting Using Machine Learning and Deep Learning Models: A Comparative Study on London Smart Meter Data

## What This Repository Contains

This public repository is intentionally compact. It does not expose the full raw research workspace. Instead, it keeps the most important material needed to understand the project quickly:

- a `submission_package/core_code/` folder with 5 core source files
- a `submission_package/results_images/` folder with the main result figures
- a `submission_package/results_tables/` folder with the main result tables

## Submission Package

### Core Code

The main compact code folder is:

- `submission_package/core_code/`

It contains these 5 core files:

1. `01_data_loader.py`
   Loads and validates the daily dataset and handles raw-to-daily aggregation logic.
2. `02_feature_engineering.py`
   Builds the leakage-safe lag, rolling, calendar, and change features.
3. `03_preprocessing.py`
   Implements the strict chronological split and supervised matrix preparation.
4. `04_modeling_ml.py`
   Contains the main benchmark machine learning models, including CatBoost, XGBoost, LightGBM, and Prophet helpers.
5. `05_evaluation.py`
   Shows exactly how RMSE, MAE, MAPE, SMAPE, R2, and direction accuracy are calculated.

### Result Images

The main image folder is:

- `submission_package/results_images/`

It contains the main visual results used in the report and paper:

- `benchmark_leaderboard.png`
- `benchmark_heatmap.png`
- `hard_case_slice.png`
- `high_volatility_slice.png`
- `care_confidence_vs_gain.png`
- `retrieval_summary.png`
- `07_catboost_forecast.png`

## Main Finding

The strongest overall model in the fair leakage-safe benchmark is:

- **CatBoost**

Main verified benchmark result:

- RMSE: `5151.33`
- MAE: `1907.47`
- R2: `0.254`

Key interpretation:

- Tree-based models performed best on this small, structured, feature-rich daily dataset.
- Deep learning models were tested fairly but did not beat the strongest tabular models.
- Hybrid residual models were useful research extensions, but they still did not beat CatBoost on held-out RMSE.

## Dataset

Canonical processed dataset used for modeling:

- `data/processed/daily_agg.parquet` in the private local workspace
- not published here as a file; linked below from the official source

Public dataset source:

- [London Datastore: SmartMeter Energy Use Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d/)

## Important Documents

- Updated Review-II report:
  - `reports/Review2_RohithGaneshKanchi_23BCE5049.pdf`
- Detailed IEEE-style paper:
  - `paper/Review2_RohithGaneshKanchi_23BCE5049.pdf`

## Repository Structure

```text
.
|-- README.md
|-- submission_package/
|   |-- core_code/
|   |-- results_images/
|   |-- results_tables/
|   `-- docs/
`-- paper/
```

## Reproducibility Note

The public repo is designed for review and understanding. It contains the most important code and output artifacts, while large raw data files, local virtual environments, cached outputs, and heavy model artifacts are not published.
