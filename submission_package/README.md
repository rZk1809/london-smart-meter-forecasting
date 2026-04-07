# Submission Package

This folder is a compact submission package prepared for quick project review.

## 1. Core Code

Location:

- `submission_package/core_code/`

Included files:

1. `01_data_loader.py`
   Shows how the dataset is loaded and validated.

2. `02_feature_engineering.py`
   Shows the leakage-safe lag, rolling, calendar, difference, and percentage-change features.

3. `03_preprocessing.py`
   Shows the strict chronological split and supervised matrix preparation.

4. `04_modeling_ml.py`
   Shows the main benchmark machine learning models, including CatBoost, LightGBM, XGBoost, and Prophet helpers.

5. `05_evaluation.py`
   Shows exactly how RMSE, MAE, MAPE, SMAPE, R2, and direction accuracy are calculated.

These 5 files were selected because they explain the strongest final benchmark path in the clearest and shortest way.

## 2. Result Images

Location:

- `submission_package/results_images/`

Included images:

- `benchmark_leaderboard.png`
- `benchmark_heatmap.png`
- `hard_case_slice.png`
- `high_volatility_slice.png`
- `care_confidence_vs_gain.png`
- `retrieval_summary.png`
- `07_catboost_forecast.png`

## 3. Result Tables

Location:

- `submission_package/results_tables/`

Included files:

- `metrics_table.csv`
- `harpv2_ablation_results.csv`
- `care_ablation_results.csv`
- `experiment_summary.json`

## 4. Documents

Location:

- `submission_package/docs/`

Included documents:

- `Review_II_Assignment_III.md`
- `Review2_PredictiveAnalytics.pdf`
- `Review2_RohithGaneshKanchi_23BCE5049.pdf`
- `IEEE_Paper_RohithGaneshKanchi.pdf`

## 5. Main Result

The strongest verified final model is:

- **CatBoost**

Main benchmark result:

- RMSE: `5151.33`
- MAE: `1907.47`

## 6. Public Links

- Repository: [https://github.com/rZk1809/london-smart-meter-forecasting](https://github.com/rZk1809/london-smart-meter-forecasting)
- Dataset: [London Datastore SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d/)
