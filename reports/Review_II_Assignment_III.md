# Review - II [Assignment - III]

## 1. Project Title

**Energy Consumption Forecasting Using Machine Learning and Deep Learning Models: A Comparative Study on London Smart Meter Data**

## 2. Member(s)

- Name: Rohith Ganesh Kanchi
- Regd. No.: 23BCE5049
- Email: rohithganesh.kanchi2023@vitstudent.ac.in
- Phone No.: +91 7305636052

## 3. Problem Statement

The goal of this project is to forecast **daily aggregate energy consumption** using smart-meter data from London households. The problem is important because better energy forecasting helps in load planning, reserve management, and power system balancing.

The project uses the Low Carbon London smart-meter dataset. The raw data contain half-hourly records from 138 households. These records were converted into one daily aggregate time series. The forecasting task is:

- use only past information
- predict the next daily energy value
- compare multiple model families fairly
- identify the best model for this dataset

The main research question is:

**Which model performs best on this small daily smart-meter dataset under a strict leakage-safe chronological evaluation setup?**

## 4. Methodology (Description / Diagram)

The project follows a clear pipeline:

1. Load and validate the raw or processed daily dataset.
2. Build a leakage-safe daily feature frame.
3. Split the data chronologically into train, validation, and test sets.
4. Train benchmark models.
5. Compare results on the same held-out test set.
6. Analyse hard cases, high-volatility cases, and hybrid residual extensions.

### Practical Method Flow

- Raw half-hourly smart-meter readings were aggregated into daily totals.
- Leakage-safe features were created using only past values.
- A strict 70/15/15 time split was used.
- Statistical, machine learning, and deep learning models were trained.
- CatBoost was selected as the strongest benchmark model.
- Residual hybrid extensions such as DynamicRetrievalResidual, HARP-v2, and CARE were then evaluated on top of the CatBoost backbone.

### Core Logic

The project treats this problem as a **small structured forecasting task**. Since the dataset is small after preprocessing, the methodology focuses on:

- strong feature engineering
- strict temporal validation
- fair comparison across models
- careful interpretation of what worked and what failed

## 5. Data Description Used for the Purpose

### Raw Data

- Source: Low Carbon London smart-meter dataset
- Raw records: 4,999,863 half-hourly rows
- Households: 138
- Raw date range: 23-11-2011 to 28-02-2014

### Final Modeling Dataset

- File used: `data/processed/daily_agg.parquet` in the local private workspace
- Schema: `ds`, `y`
- Daily rows: 829
- Missing values: 0
- Duplicate dates: 0

### Feature-Ready Dataset

After creating lag and rolling features, the usable modeling rows become:

- 739 rows

Chronological split:

- Train: 517 rows
- Validation: 110 rows
- Test: 112 rows

### Why the Data is Valid

The dataset is valid for this project because:

- it has consistent daily timestamps
- it has no missing daily targets
- it has no duplicate dates
- it shows strong weekly and monthly structure
- it is appropriate for daily forecasting after aggregation

Public dataset link:

- [London Datastore: SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d/)

Related source report:

- [Low Carbon London: A Learning Journey](https://library.ukpowernetworks.co.uk/library/en/gb/files/innovation/Low%20Carbon%20London/Project%20Publications/2014-03-14%20LCL%20Learning%20Journey%20Final.pdf)

## 6. Implementation (with relevant coding)

The full public GitHub repository is:

- [https://github.com/rZk1809/london-smart-meter-forecasting](https://github.com/rZk1809/london-smart-meter-forecasting)

The most important implementation files are kept in:

- `submission_package/core_code/`

### Core Code Files

1. `01_data_loader.py`
   Handles data loading, dataset validation, and raw-to-daily aggregation support.

2. `02_feature_engineering.py`
   Creates leakage-safe calendar, lag, rolling, difference, and percentage-change features.

3. `03_preprocessing.py`
   Applies strict chronological splitting and supervised matrix preparation.

4. `04_modeling_ml.py`
   Contains the benchmark machine learning models, including CatBoost, XGBoost, LightGBM, and Prophet helpers.

5. `05_evaluation.py`
   Contains the final metric calculations used for fair comparison.

### Why These Files Were Chosen

These files were selected because they explain the main successful pipeline:

- how the data were loaded
- how leakage was prevented
- how the feature matrix was built
- how the chronological split was applied
- how the strongest models were trained
- how the final benchmark metrics were calculated

The hybrid extensions are discussed in the report and paper, but the 5 core files above are the clearest code path for understanding the main benchmark result.

## 7. Result

### Main Benchmark Result

The strongest overall model is:

- **CatBoost**

Main metrics:

- RMSE: `5151.33`
- MAE: `1907.47`
- MAPE: `24.07`
- R2: `0.254`

Other benchmark models:

- LightGBM: RMSE `5348.21`
- XGBoost: RMSE `5387.16`
- GRU: RMSE `5610.57`
- BiLSTM: RMSE `5704.57`
- Prophet: RMSE `37716.75`

### Hybrid Extension Results

The project also tested residual hybrid extensions:

- DynamicRetrievalResidual
- HARP-v2
- CARE

Key hybrid findings:

- DynamicRetrievalResidual nearly tied CatBoost but did not beat it
- HARP-v2 slightly improved MAE but not RMSE
- CARE improved some secondary behavior but reduced overall RMSE robustness

### Result Images

Important figures are kept in:

- `submission_package/results_images/`

These include:

- benchmark leaderboard
- benchmark heatmap
- hard-case slice
- high-volatility slice
- CARE confidence versus gain
- retrieval summary

## 8. Interpretation / Discussion

The project result is meaningful and easy to explain:

### Why CatBoost Won

CatBoost performed best because:

- the dataset is small after preprocessing
- the important patterns are already exposed through the engineered features
- boosted trees are very strong on structured tabular data
- CatBoost handled nonlinear relationships better than the simpler baselines and the deep models

### Why Deep Learning Did Not Win

GRU and BiLSTM were tested fairly, but they did not beat CatBoost. The likely reasons are:

- the dataset is too small for deep sequence models to show their full advantage
- the feature engineering already exposes most of the useful temporal structure
- the task is a single daily aggregate forecasting problem, not a large multivariate sequence problem

### Why Hybrid Models Still Matter

The hybrid models were not useless. They helped answer an important research question:

**Can CatBoost's remaining error be corrected using retrieval-based or residual hybrid methods?**

The answer is:

- partially, but not enough to beat CatBoost on held-out RMSE

### Main Remaining Challenge

The hardest errors are concentrated in a few rare shock-like days. One extreme day near the end of the test window causes a very large error because the actual demand collapses sharply while the model still predicts a normal high-demand pattern.

This means the next research step is likely:

- anomaly-aware forecasting
- shock-day handling
- or external variable integration

and not just using a larger or more complex neural model.

## 9. Conclusion

This project successfully built and evaluated a fair, leakage-safe daily smart-meter forecasting pipeline.

The final conclusion is:

- **CatBoost is the best model for this dataset under the current canonical setup**

The project also showed that:

- strong feature engineering matters a lot
- strict chronological evaluation is necessary
- deep models do not automatically outperform tree models on small daily datasets
- hybrid residual extensions are useful research experiments, but they are not yet better than the CatBoost backbone

The project approach was correct, methodical, and reproducible. The final public repository, report, and paper now clearly show:

- what was done
- why it was done
- what worked
- what failed
- and what should be improved in future work

