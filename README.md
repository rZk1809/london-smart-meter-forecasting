# Smart-Meter Forecasting Review-II Package

This repository is a public teacher-facing package for the Predictive Analytics Review-II submission on daily smart-meter energy forecasting.

## Student Details

- Name: Rohith Ganesh Kanchi
- Register Number: 23BCE5049
- Phone: +91 7305636052
- Email: rohithganesh.kanchi2023@vitstudent.ac.in

## Project Title

Energy Consumption Forecasting Using Machine Learning and Deep Learning Models: A Comparative Study on London Smart Meter Data

## What This Repository Contains

This public repository is intentionally compact. It does not expose the full raw research workspace. Instead, it keeps the most important material needed for a faculty review:

- a `teacher_package/core_code/` folder with 5 core source files
- a `teacher_package/results_images/` folder with the main result figures
- a `teacher_package/docs/` folder with the final report and paper PDFs
- a `reports/` folder with the updated Review-II report
- a `paper/` folder with the detailed IEEE-style paper

## Teacher Package

### Core Code

The main teacher-facing code folder is:

- `teacher_package/core_code/`

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

The teacher-facing image folder is:

- `teacher_package/results_images/`

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

Low Carbon London reference report:

- [Low Carbon London: A Learning Journey](https://library.ukpowernetworks.co.uk/library/en/gb/files/innovation/Low%20Carbon%20London/Project%20Publications/2014-03-14%20LCL%20Learning%20Journey%20Final.pdf)

## Important Documents

- Updated Review-II report:
  - `reports/Review_II_Assignment_III.md`
- Detailed IEEE-style paper:
  - `paper/ieee_small_sample_hybrid_paper.tex`
  - `paper/Review2_RohithGaneshKanchi_23BCE5049.pdf`

## Repository Structure

```text
.
|-- README.md
|-- teacher_package/
|   |-- core_code/
|   |-- results_images/
|   |-- results_tables/
|   `-- docs/
|-- reports/
`-- paper/
```

## Reproducibility Note

The public repo is designed for review and understanding. It contains the most important code and output artifacts, while large raw data files, local virtual environments, cached outputs, and heavy model artifacts are not published.
