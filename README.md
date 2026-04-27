# Demand Forecasting V2

Demand Forecasting V2 is a raw-data-driven forecasting workflow for new product launches.
It has two production scripts:

- `build_artifacts_v2.py` (offline pipeline): cleans raw CSV files, builds model/calibration assets, and writes artifacts.
- `app_v2.py` (online app): loads artifacts, accepts structured new-launch inputs in a Gradio UI, and returns explainable forecasts.

The notebook `forecast_trial_1.ipynb` is kept for experimentation and iteration, not as the production runtime.

---

## Architecture

```text
data/raw/*.csv
    -> build_artifacts_v2.py
    -> artifacts/model_artifacts_v2.pkl (+ CSV exports)
    -> app_v2.py (Gradio UI + run logging)
```

### Main idea

1. Build historical intelligence once from raw data.
2. Reuse that intelligence for fast interactive forecasts.
3. Log each forecast run for governance and review.
4. Keep the UI focused on structured launch attributes instead of free-text use-case or target-group entry.

---

## Project Structure

```text
demand_forecasting/
  app_v2.py
  build_artifacts_v2.py
  forecast_trial_1.ipynb
  requirements.txt
  README.md

  artifacts/
    model_artifacts_v2.pkl
    launch_ratio_table_v2.csv
    monthly_new_customers_v2.csv

  data/
    raw/
      orders.csv
      sale_times.csv
      launched_product_details.csv
      target_group_mapping.csv (auto-created if missing)

    feedback/
      forecast_run_log.csv (auto-created by app_v2.py)

  app_old.py
  build_artifacts_old.py
```

Legacy files are intentionally kept for comparison, but the active flow is `*_v2.py`.

---

## Quick Start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build artifacts from raw data

```bash
python build_artifacts_v2.py
```

This creates/updates:

- `artifacts/model_artifacts_v2.pkl`
- `artifacts/launch_ratio_table_v2.csv`
- `artifacts/monthly_new_customers_v2.csv`

### 3) Run the forecasting app

```bash
python app_v2.py
```

The app opens locally with Gradio (usually `http://127.0.0.1:7860`).

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

## What `build_artifacts_v2.py` Produces

The artifact file stores:

- cleaned data tables
- calibration factors (seasonality, strategy, flavour, product form, sale, price)
- growth and ratio context
- historical ratio table for forecast metrics
- semantic similarity assets (TF-IDF based)
- behavioral customer segmentation (MiniBatchKMeans based, recomputed on each artifact build)
- metadata for training window and row counts

Note: the artifact still preserves target-group assets for backward compatibility in the data layer, but the current app UI does not expose a target-group recommendation block.

The script also exports ratio-support CSV files to `artifacts/` for inspection.

---

## What `app_v2.py` Does

At startup:

1. Loads `artifacts/model_artifacts_v2.pkl`.
2. Raises `FileNotFoundError` if artifact is missing.

For each forecast run:

1. Reads structured launch inputs from UI.
2. Finds similar historical launches using structured attribute scoring.
3. Computes ratio-based forecast metrics:
   - first-week quantity
   - first-6-week quantity
   - first-week NC
   - first-6-week NC
   - first-week total customers
   - first-6-week total customers
4. Applies adjustment factors (seasonality, strategy, flavour group, product form, price, sale overlap).
5. Calculates confidence + scenario bounds.
6. Generates explainability tables and chart.
7. Appends run details to `data/feedback/forecast_run_log.csv`.

### UI Inputs

- Product name: free-text textbox, empty by default.
- Product need area: dropdown sourced from historical launch data, empty by default.
- Benefit keywords: multiselect dropdown parsed from historical launch data, empty by default.
- Launch month: dropdown, empty by default.
- Product form: dropdown, empty by default.
- Launch strategy type: dropdown, empty by default.
- UVP: number input, empty by default.
- Flavour: dropdown sourced from historical launch data plus `New Flavour`, empty by default.
- Flavour group: dropdown sourced from historical launch data, empty by default.

The app does not ask for a free-text use case anymore and does not show a target-group recommendation block in the UI.

---

## Programmatic Usage (Optional)

You can call the core function directly from Python:

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

`run_forecast` returns multiple outputs used by the UI (tables, plot, explanation, summary, run status).

---

## Notebook Role

`forecast_trial_1.ipynb` is a sandbox for experimentation and prototyping.
Production usage should run through:

1. `build_artifacts_v2.py`
2. `app_v2.py`

This keeps runtime behavior consistent and reproducible.

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
- Check date fields are parseable.

### Input selection issues

- If a flavour exists historically, select it from the flavour dropdown.
- If the flavour is new, select `New Flavour` and still provide a flavour group.
- Benefit keywords should be selected as structured tags rather than entered as free text.

### Empty or weak model components

Some sub-models (for example target-group inference or segmentation) can be disabled automatically when data is insufficient. This is expected behavior and is reflected in output tables/reason fields.

---

## Dependencies

Current dependencies in `requirements.txt`:

- `gradio>=4.2.0`
- `plotly>=5.0.0`
- `pandas>=2.0.0`
- `numpy>=1.26.0`
- `scikit-learn>=1.3.0`
