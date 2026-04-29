# Demand Forecasting V2

Demand Forecasting V2 is a hybrid AI-powered decision-support system for segment-based demand forecasting of new D2C product launches.

The solution combines:

- ratio-based forecasting from similar historical launches
- behavioral customer segmentation
- supervised ML benchmark models
- explainability outputs
- historical backtesting

Core scripts:

- `build_artifacts_v2.py` (offline pipeline): prepares cleaned/calibrated assets and exports model diagnostics
- `app_v2.py` (online app): loads artifacts, runs forecasts in a Gradio UI, and logs every run

---

## Thesis Positioning

This project supports an AI-powered decision-support approach for segment-based demand forecasting and product launch planning in D2C. The system assists planners by generating demand forecasts, explaining forecast drivers, decomposing expected demand by behavioral customer segments, benchmarking supervised ML outputs, and logging forecast runs for traceability. Final planning decisions remain with human stakeholders.

---

## Architecture

```text
data/raw/*.csv
    -> build_artifacts_v2.py
    -> artifacts/model_artifacts_v2.pkl (+ CSV exports)
    -> app_v2.py (Gradio UI + traceable run logging)
```

### Main idea

1. Build historical forecasting intelligence once from raw sales and launch data.
2. Reuse the generated artifacts for fast, explainable launch forecasts.
3. Combine ratio-based forecasting with behavioral segmentation and supervised ML benchmarking.
4. Provide confidence scores, scenario bounds, and first-order quantity recommendations.
5. Keep all forecasts auditable through run-level logging.

---

## Decision-Support Scope

This system is not a fully autonomous agent. It does not independently trigger planning workflows, approve first-order quantities, contact stakeholders, or execute replenishment decisions. Instead, it operates as an AI-powered forecasting and decision-support application. It provides structured forecasts, comparable-launch evidence, segment-based demand composition, ML benchmarks, and traceable outputs to support human launch planning decisions.

---

## Project Structure

```text
demand_forecasting/
  app_v2.py
  build_artifacts_v2.py
  forecast_customer_ml_trial.ipynb
  requirements.txt
  README.md

  artifacts/
    model_artifacts_v2.pkl
    launch_ratio_table_v2.csv
    monthly_new_customers_v2.csv
    ml_model_metrics_v2.csv              (if supervised ML is enabled)
    ml_model_summary_v2.csv              (if supervised ML is enabled)
    quantile_metrics_v2.csv              (if supervised ML is enabled)

  data/
    raw/
      orders.csv
      sale_times.csv
      launched_product_details.csv
      target_group_mapping.csv (auto-created if missing)

    feedback/
      forecast_run_log.csv (auto-created by app_v2.py)

  archive/
    forecast_trial_1.ipynb
```

---

## Quick Start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional benchmark models:

- Install `xgboost`, `lightgbm`, and/or `catboost` if you want those candidates included in model comparison.

### 2) Build artifacts from raw data

```bash
python build_artifacts_v2.py
```

This creates/updates:

- `artifacts/model_artifacts_v2.pkl`
- `artifacts/launch_ratio_table_v2.csv`
- `artifacts/monthly_new_customers_v2.csv`
- `artifacts/ml_model_metrics_v2.csv` (when ML training is enabled)
- `artifacts/ml_model_summary_v2.csv` (when ML training is enabled)
- `artifacts/quantile_metrics_v2.csv` (when ML training is enabled)

### 3) Run the forecasting app

```bash
python app_v2.py
```

The app opens locally with Gradio (usually `http://127.0.0.1:XXX`).

---

## Raw Data Contracts

All training inputs are loaded from `data/raw/`.

### `orders.csv`

Required columns:

- `order_id`
- `customer_nr`
- `sku`
- `price`
- `date`
- `artikel_name`
- `net_revenue`
- `quantity`
- `flavour`
- `product_category`
- `product`
- `customer_status`

### `sale_times.csv`

- Expected delimiter: semicolon (`;`)
- Required columns: `name`, `start_d`, `end_d`

### `launched_product_details.csv`

Required columns:

- `sku`
- `artikel_name`
- `product`
- `product_need_area`
- `benefit_keywords`
- `flavour_group`
- `flavour`
- `product_form`
- `launch_date`
- `first_order_quantity`
- `uvp`
- `launch_strategy_type`
- `Product Use Case / What it is about`
- `Target Group`
- `first_week_quantity`
- `first_6_week_quantity`
- `first_week_nc`
- `first_6_week_nc`
- `first_week_total_c`
- `first_6_week_total_c`

---

## Build Pipeline Outputs (`build_artifacts_v2.py`)

The artifact build pipeline produces:

- `model_artifacts_v2.pkl`
- `launch_ratio_table_v2.csv`
- `monthly_new_customers_v2.csv`
- `ml_model_metrics_v2.csv` (if ML enabled)
- `ml_model_summary_v2.csv` (if ML enabled)
- `quantile_metrics_v2.csv` (if ML enabled)

The pickle artifact includes:

- cleaned data and metadata
- seasonality / strategy / flavour / product form / sale / price factor tables
- ratio context (active customer base and recent NC scale)
- behavioral segmentation assets
- supervised ML models, preprocessors, best-model selection, and quantile models

---

## Behavioral Customer Segmentation

The system builds behavioral customer segments using MiniBatchKMeans over customer behavior features, including:

- Recency
- Frequency
- Monetary
- Launch adoption history (24-month launch purchase behavior and diversity signals)

Primary outputs:

- `segment_summary`: segment-level statistics and labels
- `launch_segment_profile`: launch-level segment affinity shares (`sku x segment_key x launch_segment_share`)

Forecast impact:

- Behavioral segmentation is used both as an explainability layer and as a conservative bounded adjustment signal.
- The segment contribution table decomposes forecasted demand into expected behavioral customer segments.
- The multiplier is intentionally bounded to avoid excessive forecast volatility.
- Current production bounds: **0.90 to 1.10**.
- This range was selected after notebook-based historical testing showed that wider ranges increased forecast volatility.

### Thesis Contribution

The segmentation layer is central to the thesis contribution. It enables segment-based demand interpretation by estimating which behavioral customer groups are expected to contribute to launch demand. Therefore, its value is not limited to aggregate forecast accuracy; it also supports first order quantity, launch targeting, CRM planning, and explainability.

---

## Supervised ML Layer

Alongside the ratio engine, the build pipeline trains benchmark supervised models on historical launches.

Candidate models include:

- Random Forest
- Gradient Boosting (quantile interval layer)
- HistGradientBoosting (benchmark/experimental variant in research workflows)
- XGBoost / LightGBM / CatBoost (if installed)

During artifact build:

1. candidate models are trained and evaluated on launch-level targets
2. metrics are written to ML CSV outputs
3. the best model is selected and stored as `best_model_name`

In `app_v2.py`:

- the selected best model is loaded from artifacts
- point forecasts are generated for all target metrics
- quantile models (q10/q50/q90) are used when available

---

## Similarity Engine

`app_v2.py` uses a weighted structured similarity score across launch attributes (need area, benefit keywords, flavour, flavour group, product form, strategy, product text, price, launch month).

Similarity weights were empirically tuned through notebook-based historical backtesting and then fixed in production logic. The selected setup uses the top 7 comparable historical launches.

---

## Forecast Engine Logic

For each forecast run:

1. collect structured launch inputs
2. find top similar historical launches
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

### Forecast Details

- **Confidence score** is calculated from similarity quality and ratio support. It ranges from 0.30 (low, few comparable launches) to 0.92 (high, strong comparable launches).
- **Scenario bounds** are derived from confidence: worst-case and best-case forecasts bracket the base case.
- **First-order planning**: for supply planning, the system recommends the ratio-based first-6-week quantity plus a 10% safety buffer.
- **Supervised ML outputs** are used as benchmark estimates to validate the ratio approach, not as the only forecast source. The ratio forecast remains the primary recommendation.

---

## Historical Backtesting

Historical launch test mode supports reproducible diagnostics:

1. user selects a historical launch from dropdown format:
   - `SKU | Product | Flavour | Launch Date`
2. system recreates pre-launch context only
3. future launches are excluded from similarity references
4. a cutoff-aware ratio forecast is generated
5. forecast metrics are compared against actual outcomes

### Cutoff-Aware Mode Details

During cutoff-aware historical backtesting:

- The ratio forecast excludes future launches from similarity references.
- Customer base and new-customer context are recalculated using only orders and customer behavior before the cutoff date.
- Behavioral segmentation multiplier can be disabled or interpreted carefully to avoid future customer-behavior leakage.
- ML outputs are diagnostic unless a cutoff-specific ML artifact is built (i.e., the full-data ML model used in backtests is not time-aware).

Backtest diagnostics support model calibration, highlight weak historical comparables, and identify systematic prediction errors.

---

## App Outputs

Each run returns:

- **Forecast table**: Ratio-based worst/base/best case for all 6 metrics, plus ML point estimates and quantile intervals (q10/q50/q90) where available.
- **Behavioral segment contribution table**: Demand decomposed by expected segment composition, showing which customer groups are expected to drive adoption.
- **ML benchmark summary**: Model comparison metrics (MAE, RMSE, sMAPE, R²) for the selected best model.
- **Similar historical launches table**: Top 7 comparable launches with their attributes, historical outcomes, and similarity scores.
- **Applied factor transparency table**: Line-by-line breakdown of seasonality, strategy, flavour group, product form, price, sale, and behavioral segment factors applied to each metric.
- **Forecast chart**: Visual comparison of ratio base vs ML point estimate across all metrics.
- **Explanation text**: Detailed forecast rationale, top references, forecast logic, and key drivers.
- **Summary text**: Concise summary of forecast outputs and first-order recommendation.
- **Historical comparison table** (for backtests): Side-by-side comparison of actual outcomes vs ratio forecast vs ML forecast, with error percentages and MAPE.
- **Run log status**: Run ID and log file path for governance tracking.

---

## UI Inputs

Primary new-launch inputs:

- Product name
- Product need area
- Benefit keywords
- Launch month
- Product form
- Launch strategy type
- UVP
- Flavour
- Flavour group

Historical test input:

- Historical launch dropdown in format: `SKU | Product | Flavour | Launch Date`

---

## Governance and Traceability

Every forecast run is persisted to:

- `data/feedback/forecast_run_log.csv`

The run log supports review and governance by preserving:

- Run ID and timestamp
- Run mode (new launch forecast vs historical backtest)
- All structured inputs (product attributes, UVP, flavour, strategy, etc.)
- Selected reference SKUs (top 5 from similarity ranking)
- Ratio-based forecast outputs for all 6 metrics
- ML-based forecast outputs (point + quantiles) where enabled
- Confidence score
- Recommended first-order quantity
- Active customer base and recent NC base used for forecast
- Backtest cutoff date and results (for historical tests)

This structure enables launch planning teams to audit forecasts, trace assumptions, and review decisions in context.

---

## Programmatic Usage (Optional)

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

`run_forecast` returns UI-ready objects (tables, figure, explanation text, summary text, and run-log status).

---

## Notebook Role

### `forecast_customer_ml_trial.ipynb`

- exploratory ML and calibration notebook
- validates customer behavior signals and launch buyer prediction ideas
- computes historical buyer-ratio calibration outputs used in the ratio framework

### `archive/forecast_trial_1.ipynb`

- archived experimentation notebook used during earlier iteration and backtesting phases

Production runtime path remains:

1. `build_artifacts_v2.py`
2. `app_v2.py`

---

## Troubleshooting

### Artifact missing

If `app_v2.py` fails with artifact-not-found:

```bash
python build_artifacts_v2.py
```

### CSV parsing issues

- Ensure `sale_times.csv` is semicolon-separated.
- Ensure required columns exist exactly as listed above.
- Ensure date fields are parseable.

### Optional ML packages not installed

- The app still runs if `xgboost`, `lightgbm`, or `catboost` are unavailable.
- In that case, those candidates are skipped during model comparison.

### Historical backtest interpretation

- **High MAPE does not automatically mean the system failed.** It may indicate:
  - Weak historical comparables (genuinely unique launches with few similar predecessors)
  - Unusual launch reception (viral/seasonal/external factors not captured)
  - Insufficient category-specific calibration (buyer ratios or NC ratios may need category refinement)
  - Changing customer behavior (historical baselines outdated vs current market dynamics)

- Use backtest diagnostics to identify improvement areas:
  - Compare ratio and ML diagnostic outputs to cross-validate direction and magnitude.
  - Review the factor table to spot misfires in seasonality, strategy, or price adjustments.
  - Examine the similar-launch table to assess whether the best references are truly comparable.
  - Segment-contribution insights may reveal if demand concentration differs from historical patterns.

---

## Dependencies

Current dependencies in `requirements.txt`:

- `gradio>=4.2.0`
- `plotly>=5.0.0`
- `pandas>=2.0.0`
- `numpy>=1.26.0`
- `scipy>=1.11.0`
- `scikit-learn>=1.3.0`
