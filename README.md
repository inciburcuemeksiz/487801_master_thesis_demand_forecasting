# Demand Forecasting V2

Demand Forecasting V2 is a hybrid AI-powered decision-support system for segment-based demand forecasting of new D2C product launches.

The system combines:

- **Similarity-based forecasting**: comparable-launch identification with learned attribute weights
- **Customer segmentation**: behavioral clustering for demand composition
- **Supervised ML benchmarking**: CatBoost and quantile models for performance validation
- **Explainability**: forecast driver attribution and top-N comparable launch evidence
- **Historical backtesting**: leave-one-out validation for weight optimization
- **Audit trail**: comprehensive forecast run logging for traceability

> The repository does **not** include real business data or generated model artifacts. A synthetic mock data workflow is included so that reviewers can run the project locally without access to private company data.

---

## System Overview

### Core Workflows

The system operates through three main entry points:

1. **Artifact Building** (`build_artifacts_v2.py`)
   - Loads raw launch and order data
   - Cleans and normalizes product attributes (flavour, product form, launch strategy, etc.)
   - Engineers features for similarity-based and supervised ML forecasting
   - Trains CatBoost models and quantile regressors for six target metrics
   - Exports model objects and diagnostic CSVs

2. **Weight Optimization** (`archive/experiments/backtest_similarity_weights.py`)
   - Backtests alternative similarity-weight configurations
   - Uses leave-one-out cross-validation on historical launches
   - Compares baseline schemes (equal, category-heavy, launch-heavy, current) 
   - Runs randomized search over 1000 weight iterations to find empirically optimal configuration
   - Selects best weights based on weighted mean absolute percentage error (WMAPE)

3. **Forecast Interface** (`app_v2.py`)
   - Gradio-based decision-support UI
   - Forecasts demand using learned similarity weights
   - Identifies and ranks comparable historical launches
   - Runs supervised ML models for performance benchmarking
   - Logs forecast runs with full metadata and timestamp

### Decision-Support Scope

This system is not a fully autonomous agent. It does not independently trigger planning workflows, approve stock quantities, contact stakeholders, or execute replenishment decisions.

Instead, it operates as a forecasting and decision-support application. It provides:

- Structured demand forecasts with confidence intervals
- Evidence from comparable historical launches  
- Segment-based demand composition
- ML benchmark outputs for forecast validation
- Scenario bounds and uncertainty estimates
- Complete audit trails for forecast review

Final planning decisions remain with human stakeholders.

---

## Similarity-Based Forecasting

The system's core innovation is **similarity-weighted forecasting**: predicting a new product's demand by identifying the most similar historical launches and averaging their outcomes.

### Similarity Calculation

Similarity between two launches is computed as a weighted sum across nine product attributes:

**Categorical attributes** (exact match = 1.0, otherwise = 0.0):
- Launch month
- Flavour  
- Product need area
<!-- Detailed directory tree removed to avoid duplication; see consolidated Project Structure above. -->
**File**: [archive/experiments/backtest_similarity_weights.py](archive/experiments/backtest_similarity_weights.py)

### Purpose

This script optimizes the similarity weights used to match new products with comparable historical launches. It runs two phases:

1. **Baseline Phase**: Evaluates four predefined weight schemes
2. **Random Search Phase**: Searches 1000 random weight configurations to find an empirically optimal set

### Key Functions

**`similarity_score(row_a, row_b, weights)`**
- Computes weighted similarity between two launches
- Blends categorical exactness, price proximity, and text similarity
- Returns normalized score in [0, 1]

**`predict_from_similar_launches(target_row, candidate_rows, weights, metric, top_k=5)`**
- Finds top-K most similar historical launches
- Predicts target metric by weighted-averaging similar launches' outcomes
- Returns point estimate and evidence (match indices and similarity scores)

**`run_backtest(launch_df, weights, weight_set_name, top_k=5)`**
- Implements leave-one-out cross-validation
- For each launch: predicts its metrics using all others, records actual vs predicted
- Returns detailed results: predictions, actuals, errors, and match evidence

**`run_random_weight_search(launch_df, n_iterations=5000, top_k=5, random_seed=42)`**
- Generates random normalized weight vectors using Dirichlet distribution
- For each: runs full backtest and scores by average WMAPE
- Tracks best configuration and returns its weights, metrics, and detailed predictions

**`evaluate_results(backtest_df)`**
- Aggregates detailed predictions into summary statistics per weight set and metric
- Computes MAE, RMSE, MAPE, WMAPE for each configuration

### Workflow

```
1. Load launch dataframe from model artifacts
2. Run baseline backtests on 4 predefined weight schemes
3. Print summary of baseline performance by metric
4. Run random search over 1000 iterations
5. Track best configuration (lowest average WMAPE)
6. Export results:
   - weight_backtest_results.csv (baseline summary statistics)
   - weight_backtest_predictions.csv (baseline detailed predictions)
   - weight_random_search_results.csv (all 1000 random iterations ranked)
   - best_similarity_weights.csv (optimal weight vector)
   - best_similarity_weights_backtest_summary.csv (summary for best config)
   - best_similarity_weights_predictions.csv (detailed predictions for best config)
```

### Output Interpretation

**Summary statistics** (MAE, RMSE, MAPE, WMAPE):
- Lower values indicate better predictive accuracy
- WMAPE is the optimization target (weighted by actual values, robust to outliers)
- Separate statistics for each target metric (first week quantity, 6-week quantity, etc.)

**Detailed predictions**:
- Actual vs predicted for each historical launch
- Top-K match indices and similarity scores (evidence)
- Individual prediction errors for post-hoc analysis

### Target Metrics

The script backtests predictions on six demand metrics:

- `first_week_quantity`: Total units sold in first week
- `first_6_week_quantity`: Total units sold in first 6 weeks
- `first_week_nc`: New customers acquired in first week
- `first_6_week_nc`: New customers acquired in first 6 weeks
- `first_week_total_c`: Total customers (new + repeat) in first week
- `first_6_week_total_c`: Total customers (new + repeat) in first 6 weeks

### Running the Script

```bash
# Ensure artifacts are built
python build_artifacts_v2.py

# Run backtest and weight optimization
python archive/experiments/backtest_similarity_weights.py

# Review results in artifacts/
ls artifacts/weight_*.csv
ls artifacts/best_similarity_weights*.csv
```

---

## Versioning Rationale

This repository uses versioned scripts to make the development path of the thesis project traceable.

### V1: Initial MVP

The first version focused on building a minimum viable forecasting workflow. It included early Gradio UI experiments and initial ratio-based demand estimation logic.

However, V1 had several limitations:

- weaker separation between offline artifact generation and online forecasting
- limited explainability outputs
- less structured customer behavior modelling
- limited model diagnostics
- no formal comparable-launch neighborhood backtesting
- weaker reproducibility for external review

### V2: Refactored Thesis Version

V2 is the main version used for the thesis. It separates the system into two core workflows:

- `build_artifacts_v2.py`: offline artifact generation, data cleaning, feature preparation, model training, and diagnostic exports
- `app_v2.py`: online Gradio interface for running forecasts, comparing historical launches, explaining forecast drivers, and logging forecast runs

Compared with V1, V2 adds:

- cleaned product-level launch data preparation
- similarity-based comparable-launch forecasting
- behavioral customer segmentation
- supervised ML benchmark models
- uncertainty intervals through quantile models
- exportable diagnostics
- forecast run logging
- mock data workflow for reproducible local testing
- backtesting-based selection of top-3 comparable launches

### Future Work

Future versions could improve the system by:

- adding automated hyperparameter tuning
- adding scheduled retraining workflows
- adding stockout and inventory availability features
- improving customer-level propensity modelling
- supporting automated scenario simulations
- providing a production-ready API backend

---

## Architecture

```text
Data Input
    ↓
data/raw/*.csv or data/mock/*.csv
    ↓
[build_artifacts_v2.py] ← [backtest_similarity_weights.py]
    ↓                              ↓
artifacts/model_artifacts_v2.pkl   artifacts/best_similarity_weights.csv
artifacts/*.csv (diagnostics)
    ↓
[app_v2.py] (Gradio forecasting UI)
    ↓
Forecast outputs + run log
```

---

## Quick Start

### With Mock Data (No Private Data Required)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate mock training data
python testing/generate_mock_data.py

# Build model artifacts
python build_artifacts_v2.py

# Optimize similarity weights (optional, uses defaults if skipped)
python archive/experiments/backtest_similarity_weights.py

# Launch forecasting UI
python app_v2.py
# Open http://localhost:7860 in browser
```

### Data Requirements

Input CSVs must include columns for:
- Launch identifiers: `sku`, `product`
- Product attributes: `flavour`, `product_form`, `product_need_area`, `launch_strategy`, `benefit_keywords`
- Metadata: `launch_month` or `launch_date`, `price` or `uvp`
- Text: `product_text` or components (`product`, `use_case`, `target_group`)
- Targets: `first_week_quantity`, `first_6_week_quantity`, `first_week_nc`, `first_6_week_nc`, `first_week_total_c`, `first_6_week_total_c`

---

## Project Structure

Compact overview of key files and directories:

| Directory/File | Purpose |
|---|---|
| `build_artifacts_v2.py` | Offline workflow: data cleaning, feature engineering, model training |
| `app_v2.py` | Online Gradio UI and run logging |
| `archive/experiments/` | Backtesting and experiments (weight optimization) |
| `data/` | Raw, mock, and feedback CSVs used for training and evaluation |
| `artifacts/` | Generated model artifacts and diagnostic exports (not committed) |
| `testing/` | Utilities: mock data generation and backtesting scripts |

---

## Key Features

✓ **Similarity-weighted forecasting** with learned attribute weights  
✓ **Empirical weight optimization** via randomized backtesting  
✓ **Comparable-launch evidence** for forecast explainability  
✓ **Customer segmentation** for demand composition analysis  
✓ **ML benchmark models** for forecast validation  
✓ **Confidence intervals** via quantile regression  
✓ **Audit trail** with complete run logging  
✓ **Mock data pipeline** for reproducible testing  
✓ **Interactive Gradio UI** for stakeholder engagement  

---

## Model Outputs

**Forecast Summary**:
- Point estimates for six demand metrics
- Confidence intervals (quantile predictions)
- Top-3 comparable historical launches with similarity scores

**Forecast Evidence**:
- Comparable launches ranked by similarity
- Attribute-level similarity decomposition  
- Forecast driver attribution

**ML Benchmarks**:
- CatBoost predictions for comparison
- Ensemble-based confidence bounds

**Run Log**:
- Timestamp, user, input parameters
- Forecast outputs with full trace
- Top matches and similarity scores

---

## References

- **Decision-support systems**: Simon (1960), Keen & Scott Morton (1978)
- **Similar-case reasoning**: Kolodner (1993), Leake & Whitehouse (2009)
- **Demand forecasting**: Armstrong (2001), Gilliland & Tashman (2003)
- **Backtesting**: Blastland & Dilnot (2007), Armstrong & Fildes (2006)

**Main Idea:** Build historical forecasting intelligence once from sales, launch, campaign, and customer behavior data. Reuse generated artifacts for fast, explainable launch forecasts. Combine ratio-based forecasting with behavioral segmentation and supervised ML benchmarking. Provide confidence scores, scenario bounds, and first-order quantity recommendations. Keep forecasts auditable through run-level logging.

---

## Project Structure

```
demand_forecasting/
  app_v2.py
  build_artifacts_v2.py
  requirements.txt
  README.md
  .gitignore

  testing/
    generate_mock_data.py
    backtest_topn.py

  data/
    raw/
      .gitkeep
      # real CSV files are placed here locally but are not committed

    mock/
      .gitkeep
      # generated synthetic mock CSV files are written here

    feedback/
      .gitkeep
      # forecast_run_log.csv is generated by app_v2.py

  artifacts/
    # generated model artifacts and diagnostic CSVs
    # not committed because they may contain private data

  archive/
    experiments/
      forecast_customer_ml_trial.ipynb
```

### Active Runtime Files

The final V2 runtime path is:

```
build_artifacts_v2.py
app_v2.py
```

### Testing and Reproducibility Files

- `testing/generate_mock_data.py`: creates synthetic mock CSV files with the same schema required by the pipeline.
- `testing/backtest_topn.py`: evaluates different comparable-launch neighborhood sizes through leave-one-launch-out backtesting.

### Archived Experiments

Exploratory notebooks are stored under `archive/experiments/`. These files document earlier modelling ideas and calibration trials but are not required to run the final V2 pipeline.

---

## Quick Start

> Recommended Python version: **Python 3.9+**

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

**Optional benchmark models:** Install `xgboost`, `lightgbm`, and/or `catboost` if you want those candidates included in model comparison.

---

## Run with Mock Data

This is the recommended path for GitHub users who clone the repository without access to private business data.

The real business data is not included in this repository because it may contain private sales, customer, product, and revenue information.

Use this exact flow:

1. Generate synthetic mock data.
2. Build artifacts from `data/mock/`.
3. Start the Gradio app.

Generate the mock dataset:

```bash
python testing/generate_mock_data.py
```

This creates mock CSV files under `data/mock/`:

```
data/mock/orders.csv
data/mock/sale_times.csv
data/mock/launched_product_details.csv
data/mock/target_group_mapping.csv
```

Then build artifacts from the mock dataset:

```bash
DATA_DIR=data/mock python build_artifacts_v2.py
```

Then run the Gradio app:

```bash
python app_v2.py
```

Open the local Gradio URL shown in the terminal, usually: [http://127.0.0.1:7860](http://127.0.0.1:7860)

> The mock data follows the required schema of the pipeline but does not represent real business performance.

---

## Run with Real Data

To run the project with real company data, place the required private CSV files under `data/raw/`:

```
orders.csv
sale_times.csv
launched_product_details.csv
target_group_mapping.csv
```

Then run:

```bash
python build_artifacts_v2.py
python app_v2.py
```

> The default data directory is `data/raw/`. The `DATA_DIR` environment variable is only needed when using another data source, such as `data/mock/`. Real data files should not be committed to GitHub.

---

## Raw Data Contracts

All training inputs are loaded from `data/raw/` by default, or from another folder if `DATA_DIR` is provided.

### `orders.csv`

Required columns:

| Column | Notes |
|--------|-------|
| `order_id` | |
| `customer_nr` | |
| `sku` | |
| `price` | |
| `date` | |
| `artikel_name` | |
| `net_revenue` | |
| `quantity` | |
| `flavour` | |
| `product_category` | |
| `product` | |
| `customer_status` | |

Additional columns such as `first_order_date`, `last_order_date`, `months_since_first_order`, and `nr_of_purchase` can be used when available.

### `sale_times.csv`

Expected delimiter: **semicolon (`;`)**

Required columns: `name`, `start_d`, `end_d`

### `launched_product_details.csv`

Required columns:

| Column | Column |
|--------|--------|
| `sku` | `launch_date` |
| `artikel_name` | `first_order_quantity` |
| `product` | `uvp` |
| `product_need_area` | `launch_strategy_type` |
| `benefit_keywords` | `Product Use Case / What it is about` |
| `flavour` | `Target Group` |
| `flavour_group` | `first_week_quantity` |
| `product_form` | `first_6_week_quantity` |
| `first_week_nc` | `first_6_week_nc` |
| `first_week_total_c` | `first_6_week_total_c` |

### `target_group_mapping.csv`

Required columns: `raw_target_group`, `canonical_target_group`

> If this file is missing, `build_artifacts_v2.py` can create a starter mapping file from the raw `Target Group` values in `launched_product_details.csv`.

---

## Build Pipeline Outputs

`build_artifacts_v2.py` produces:

```
artifacts/model_artifacts_v2.pkl
artifacts/launch_ratio_table_v2.csv
artifacts/monthly_new_customers_v2.csv
artifacts/ml_model_metrics_v2.csv        # when ML training is enabled
artifacts/ml_model_summary_v2.csv        # when ML training is enabled
artifacts/quantile_metrics_v2.csv        # when ML training is enabled
```

The pickle artifact includes:

- cleaned data and metadata
- seasonality factors
- launch strategy factors
- flavour group factors
- product form factors
- sale overlap factors
- price elasticity estimate
- ratio context
- behavioral segmentation assets
- supervised ML models and preprocessors
- best-model selection
- quantile interval models

> Generated artifacts are intentionally excluded from Git because they may contain private data.

---

## Behavioral Customer Segmentation

The system builds behavioral customer segments using `MiniBatchKMeans` over customer behavior features, including:

- recency
- frequency
- monetary value
- average price
- sale sensitivity
- product diversity
- flavour diversity
- launch adoption history
- limited edition purchase behavior
- co-creation purchase behavior

**Primary outputs:**

- `segment_summary`: segment-level statistics and labels
- `launch_segment_profile`: launch-level segment affinity shares

**Forecast impact:** Behavioral segmentation is used as both an explainability layer and a conservative bounded adjustment signal. The segment contribution table decomposes forecasted demand into expected behavioral customer segments. The multiplier is intentionally bounded to avoid excessive forecast volatility.

> Current production bounds: **0.90 to 1.10**

**Thesis Contribution:** The segmentation layer is central to the thesis contribution. It enables segment-based demand interpretation by estimating which behavioral customer groups are expected to contribute to launch demand. Its value is not limited to aggregate forecast accuracy — it also supports first-order quantity planning, launch targeting, CRM planning, explanation of expected demand composition, and post-launch review.

---

## Similarity Engine

`app_v2.py` uses a weighted similarity score across launch attributes such as:

- product need area
- benefit keywords
- flavour
- flavour group
- product form
- launch strategy
- product text
- price
- launch month

The similarity-based forecast uses the **top 3** comparable historical launches. This value was selected through leave-one-launch-out backtesting using `testing/backtest_topn.py`. The tested neighborhood sizes were: 3, 5, 7, 10, and 15 comparable launches.

For each historical launch, the system removed that launch from the candidate pool, predicted its launch performance using only the remaining launches, and compared the prediction with the observed actual values. The **top-3 configuration achieved the lowest average MAPE and SMAPE** across the selected launch metrics. Although top-5 showed a slightly lower average MAE, the difference was marginal while top-3 performed better on relative error metrics.

---

## Supervised ML Layer

Alongside the ratio engine, the build pipeline trains benchmark supervised models on historical launches.

**Candidate models:**

- Random Forest
- XGBoost *(if installed)*
- LightGBM *(if installed)*
- CatBoost *(if installed)*
- Gradient Boosting quantile models for interval estimates

**During artifact build:**

- candidate models are trained and evaluated on launch-level targets
- metrics are written to ML CSV outputs
- the best model is selected and stored as `best_model_name`
- quantile models are trained for q10, q50, and q90 interval estimates

**In `app_v2.py`:**

- the selected best model is loaded from artifacts
- point forecasts are generated for all target metrics
- quantile models are used when available
- ML outputs are treated as benchmark estimates, not as the only forecast source

> The ratio forecast remains the primary planning recommendation.

---

## Forecast Engine Logic

For each forecast run:

1. collect structured launch inputs
2. find the top 3 similar historical launches
3. calculate weighted historical ratios
4. apply demand factors:
   - seasonality
   - launch strategy
   - flavour group
   - product form
   - price elasticity
   - sale overlap
   - behavioral segmentation
5. generate ratio forecast
6. generate ML benchmark forecast
7. calculate confidence intervals and scenario bounds
8. derive first-order quantity recommendation
9. log run outputs

**Forecast Details:**

- **Confidence score** is calculated from similarity quality and ratio support. It ranges from 0.30 for low support to 0.92 for strong comparable-launch support.
- **Scenario bounds** are derived from confidence. Worst-case and best-case forecasts bracket the base case.
- **First-order planning** uses the ratio-based first-six-week quantity plus a 10% safety buffer.
- **Supervised ML outputs** are used as benchmark estimates to validate the ratio approach.

---

## Historical Backtesting

Historical launch test mode supports reproducible diagnostics:

- user selects a historical launch from dropdown format: `SKU | Product | Flavour | Launch Date`
- system recreates pre-launch context
- future launches are excluded from similarity references
- a cutoff-aware ratio forecast is generated
- forecast metrics are compared against actual outcomes

### Cutoff-Aware Mode Details

During cutoff-aware historical backtesting:

- The ratio forecast excludes future launches from similarity references.
- Customer base and new-customer context are recalculated using only orders and customer behavior before the cutoff date.
- Behavioral segmentation multiplier can be disabled or interpreted carefully to avoid future customer-behavior leakage.
- ML outputs are diagnostic unless a cutoff-specific ML artifact is built. The full-data ML model used in backtests is not time-aware.

Backtest diagnostics support model calibration, highlight weak historical comparables, and identify systematic prediction errors.

### Comparable Launch Top-N Backtesting

The script `testing/backtest_topn.py` evaluates different comparable-launch neighborhood sizes using leave-one-launch-out testing. Predictions are generated for top-N values of 3, 5, 7, 10, and 15, and MAE, RMSE, MAPE, and SMAPE are aggregated across target metrics.

**Output files:**

```
artifacts/backtesting/topn_backtest_details.csv
artifacts/backtesting/topn_backtest_summary.csv
```

> These files are generated outputs and are not committed to Git.

---

## App Outputs

Each run returns:

| Output | Description |
|--------|-------------|
| **Forecast table** | Ratio-based worst/base/best case for all 6 metrics, plus ML point estimates and quantile intervals where available |
| **Behavioral segment contribution table** | Demand decomposed by expected segment composition |
| **ML benchmark summary** | Model comparison metrics (MAE, RMSE, sMAPE, R²) |
| **Similar historical launches table** | Top 3 comparable launches with attributes, historical outcomes, and similarity scores |
| **Applied factor transparency table** | Line-by-line breakdown of all demand factors |
| **Forecast chart** | Visual comparison of ratio base forecast vs ML point estimate across all metrics |
| **Explanation text** | Detailed forecast rationale, top references, forecast logic, and key drivers |
| **Summary text** | Concise summary of forecast outputs and first-order recommendation |
| **Historical comparison table** | For backtests: actual vs ratio forecast vs ML forecast |
| **Run log status** | Run ID and log file path for governance tracking |

---

## UI Inputs

**Primary new-launch inputs:**

- product name
- product need area
- benefit keywords
- launch month
- product form
- launch strategy type
- UVP
- flavour
- flavour group

**Historical test input:**

- historical launch dropdown in format: `SKU | Product | Flavour | Launch Date`

---

## Governance and Traceability

Every forecast run is persisted to:

```
data/feedback/forecast_run_log.csv
```

The run log supports review and governance by preserving:

- run ID and timestamp
- run mode
- structured product inputs
- selected reference SKUs from the top 3 comparable historical launches
- ratio-based forecast outputs for all target metrics
- ML-based forecast outputs where enabled
- confidence score
- recommended first-order quantity
- active customer base and recent new-customer base used for forecast
- backtest cutoff date and results for historical tests

This structure enables launch planning teams to audit forecasts, trace assumptions, and review decisions in context.

---

## Programmatic Usage

```python
import app_v2

result = app_v2.run_forecast(
    product_name="Daily Gut Focus",
    product_need_area="gut_health",
    benefit_keywords=["gut balance", "microbiota", "digestive support"],
    launch_month_year="09-2026",
    product_form_ui="Drinking powder",
    launch_strategy_type="standard",
    uvp=49.9,
    flavour="Lemon Ice Tea",
    flavour_group="citrus",
)

print(len(result))
```

`run_forecast` returns UI-ready objects such as tables, figure, explanation text, summary text, and run-log status.

---

## Experiment and Archive Role

`archive/experiments/forecast_customer_ml_trial.ipynb` contains exploratory ML and calibration experiments. It was used to validate customer behavior signals, launch buyer prediction ideas, and historical buyer-ratio concepts during development. It is **not required** to run the final V2 application.

**Active production path:**

```
build_artifacts_v2.py
app_v2.py
```

**Testing path:**

```
testing/generate_mock_data.py
testing/backtest_topn.py
```

---

## Data Privacy and Git Policy

Real data and generated artifacts are intentionally excluded from Git. The following should **not** be committed:

```
data/raw/*.csv
data/mock/*.csv
data/feedback/*.csv
artifacts/
catboost_info/
.venv/
.venv-1/
venv/
```

The repository keeps placeholder files such as `.gitkeep` so that required folders exist after cloning. This protects customer information, order and revenue data, product performance data, generated artifacts that may contain embedded dataframes, and forecast logs.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Launch** | A newly introduced product or SKU with a defined launch date. |
| **Historical launch** | A past product launch used as reference data for forecasting. |
| **Comparable launch** | A historical launch selected as similar to the new product setup. |
| **Top-N comparables** | The N most similar historical launches used for similarity-based forecasting. |
| **First-week quantity** | Total units sold during the first 7 days after launch. |
| **First-six-week quantity** | Total units sold during the first 42 days after launch. |
| **NC** | New customers. |
| **Total C** | Total customers purchasing the launched product during the relevant window. |
| **UVP** | Product list price. |
| **Launch strategy type** | Launch classification such as standard, co-creation, or limited edition. |
| **Behavioral segment** | Customer group created using purchase behavior features such as recency, frequency, monetary value, sale sensitivity, and launch adoption. |
| **Ratio-based forecast** | Forecasting method that estimates demand using historical buyer ratios from comparable launches. |
| **Supervised ML benchmark** | Machine learning model trained on historical launches to provide benchmark forecast outputs. |
| **Backtesting** | Historical validation method where past launches are predicted using only other historical launches. |

---

## Troubleshooting

### Artifact missing

If `app_v2.py` fails with artifact-not-found:

```bash
python build_artifacts_v2.py
```

For mock data:

```bash
python testing/generate_mock_data.py
DATA_DIR=data/mock python build_artifacts_v2.py
```

### CSV parsing issues

- Ensure `sale_times.csv` is semicolon-separated.
- Ensure required columns exist exactly as listed in the [Raw Data Contracts](#raw-data-contracts) section.
- Ensure date fields are parseable.
- Ensure numeric columns such as `uvp`, `quantity`, `net_revenue`, and launch target metrics are parseable.

### Optional ML packages not installed

The app can still run if `xgboost`, `lightgbm`, or `catboost` are unavailable. In that case, those candidates are skipped during model comparison.

### Historical backtest interpretation

High MAPE does not automatically mean the system failed. It may indicate:

- weak historical comparables
- genuinely unique launches
- unusual launch reception
- campaign effects not fully captured
- insufficient category-specific calibration
- changing customer behavior over time

Use backtest diagnostics to identify improvement areas: compare ratio and ML diagnostic outputs, review the applied factor table, inspect the similar-launch table, examine segment contribution patterns, and check whether selected comparable launches are truly similar.

---

## Dependencies

**Required (`requirements.txt`):**

```
gradio>=4.2.0
plotly>=5.0.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.11.0
scikit-learn>=1.3.0
```

**Optional:**

```
xgboost
lightgbm
catboost
```
