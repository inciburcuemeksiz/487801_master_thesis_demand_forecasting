import os
import re
import pickle
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from scipy import sparse


# ============================================================
# CONFIG
# ============================================================

ARTIFACT_PATH = "artifacts/model_artifacts_v2.pkl"

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join([c for c in x if not unicodedata.combining(c)])
    x = re.sub(r"[^a-z0-9äöüß\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_strategy(x):
    if pd.isna(x):
        return "standard"

    x = normalize_text(x)
    x = x.replace("-", "_").replace(" ", "_")

    mapping = {
        "standard": "standard",
        "standart": "standard",
        "co_creation": "co_creation",
        "cocreation": "co_creation",
        "co": "co_creation",
        "limited_edition": "limited_edition",
        "limited": "limited_edition",
    }

    return mapping.get(x, x)


def safe_divide(a, b, default=1.0):
    if b is None or pd.isna(b) or b == 0:
        return default
    if a is None or pd.isna(a):
        return default
    return a / b


def clip_factor(x, low=0.5, high=1.8):
    if pd.isna(x):
        return 1.0
    return float(np.clip(x, low, high))


def token_similarity(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def string_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def month_circular_similarity(m1, m2):
    if pd.isna(m1) or pd.isna(m2):
        return 0.0

    m1 = int(m1)
    m2 = int(m2)

    distance = abs(m1 - m2)
    distance = min(distance, 12 - distance)

    return 1 - (distance / 6)


def price_similarity(p1, p2):
    if pd.isna(p1) or pd.isna(p2) or p1 <= 0 or p2 <= 0:
        return 0.5

    return float(min(p1, p2) / max(p1, p2))


def strategy_similarity(s1, s2):
    s1 = normalize_strategy(s1)
    s2 = normalize_strategy(s2)

    if s1 == s2:
        return 1.0

    pair = {s1, s2}

    if pair == {"co_creation", "limited_edition"}:
        return 0.50

    if pair == {"standard", "limited_edition"}:
        return 0.35

    if pair == {"standard", "co_creation"}:
        return 0.30

    return 0.25


def get_factor_from_dict(factor_dict, key, metric, default=1.0):
    key = normalize_text(key)

    if not key:
        return default

    if key in factor_dict and metric in factor_dict[key]:
        return factor_dict[key][metric]

    return default


def to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build_target_group_table(product_name, use_case, flavour, launch_strategy_type, uvp):
    """
    Predict top-3 target groups for the new launch profile.
    """
    enabled = TARGET_GROUP_INFERENCE.get("enabled", False)

    if not enabled:
        return pd.DataFrame(
            [
                {
                    "Predicted Target Group": "N/A",
                    "Confidence Score": np.nan,
                    "Coverage (orders/share)": TARGET_GROUP_INFERENCE.get(
                        "reason",
                        "Target-group model not available",
                    ),
                }
            ]
        )

    model = TARGET_GROUP_INFERENCE.get("model")
    vectorizer = TARGET_GROUP_INFERENCE.get("vectorizer")

    if model is None or vectorizer is None:
        return pd.DataFrame(
            [
                {
                    "Predicted Target Group": "N/A",
                    "Confidence Score": np.nan,
                    "Coverage (orders/share)": "Model artifacts missing",
                }
            ]
        )

    numeric_cols = TARGET_GROUP_INFERENCE.get("numeric_feature_columns", [])
    numeric_defaults = TARGET_GROUP_INFERENCE.get("numeric_feature_defaults", {})
    coverage_by_group = TARGET_GROUP_INFERENCE.get("coverage_by_group", {})

    uvp_value = to_float(uvp, default=numeric_defaults.get("uvp", 0.0))

    feature_text = " ".join(
        [
            normalize_text(product_name),
            normalize_text(use_case),
            normalize_text(flavour),
            normalize_strategy(launch_strategy_type),
        ]
    ).strip()

    row_values = []
    for col in numeric_cols:
        if col == "uvp":
            row_values.append(uvp_value)
        else:
            row_values.append(float(numeric_defaults.get(col, 0.0)))

    x_text = vectorizer.transform([feature_text])
    x_num = sparse.csr_matrix([row_values]) if numeric_cols else sparse.csr_matrix((1, 0))
    x_input = sparse.hstack([x_text, x_num], format="csr")

    probabilities = model.predict_proba(x_input)[0]
    classes = model.classes_
    top_idx = np.argsort(probabilities)[::-1][:3]

    rows = []
    for idx in top_idx:
        label = str(classes[idx])
        score = float(probabilities[idx])

        coverage = coverage_by_group.get(label, {})
        cov_count = int(coverage.get("count", 0))
        cov_share = float(coverage.get("share", 0.0))

        rows.append(
            {
                "Predicted Target Group": label,
                "Confidence Score": round(score, 3),
                "Coverage (orders/share)": f"{cov_count} launches | {cov_share:.1%}",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(
            f"{ARTIFACT_PATH} not found. Run: python build_artifacts_v2.py"
        )

    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)

    return artifacts


ARTIFACTS = load_artifacts()

LAUNCHES = ARTIFACTS["data"]["launches"].copy()
SALE_TIMES = ARTIFACTS["data"]["sale_times"].copy()
CALIBRATION = ARTIFACTS["calibration"]
METADATA = ARTIFACTS["metadata"]

SEASONALITY_INDEX = CALIBRATION["seasonality_index"]
STRATEGY_FACTORS = CALIBRATION["strategy_factors"]
FLAVOUR_FACTORS = CALIBRATION["flavour_factors"]
PRODUCT_FORM_FACTORS = CALIBRATION["product_form_factors"]
SALE_FACTORS = CALIBRATION["sale_factors"]
PRICE_ELASTICITY = CALIBRATION["price_elasticity"]
GROWTH_CONTEXT = CALIBRATION["growth_context"]
TARGET_GROUP_INFERENCE = ARTIFACTS.get("target_group_inference", {})


# ============================================================
# SIMILARITY ENGINE
# ============================================================

def score_launch_similarity(
    row,
    product_name,
    use_case,
    target_group,
    flavour,
    product_form,
    launch_month,
    launch_strategy_type,
    uvp,
):
    """
    Computes similarity between the new launch input and one historical launch.
    """

    product_score = max(
        token_similarity(product_name, row.get("product_norm", "")),
        token_similarity(product_name, row.get("artikel_name_norm", "")),
        string_similarity(product_name, row.get("product_norm", "")),
    )

    use_case_score = max(
        token_similarity(use_case, row.get("use_case_norm", "")),
        token_similarity(use_case, row.get("target_group_norm", "")),
    )

    target_group_score = max(
        token_similarity(target_group, row.get("target_group_norm", "")),
        token_similarity(target_group, row.get("use_case_norm", "")),
    )

    flavour_score = max(
        token_similarity(flavour, row.get("flavour_norm", "")),
        string_similarity(flavour, row.get("flavour_norm", "")),
    )

    product_form_score = max(
        token_similarity(product_form, row.get("product_form_norm", "")),
        string_similarity(product_form, row.get("product_form_norm", "")),
    )

    strategy_score = strategy_similarity(
        launch_strategy_type,
        row.get("launch_strategy_type", "standard"),
    )

    month_score = month_circular_similarity(
        launch_month,
        row.get("launch_month", np.nan),
    )

    price_score = price_similarity(
        uvp,
        row.get("uvp", np.nan),
    )

    # Weighted similarity.
    # Sum should be 1.00.
    total_score = (
        0.18 * product_score
        + 0.20 * use_case_score
        + 0.14 * target_group_score
        + 0.14 * flavour_score
        + 0.12 * product_form_score
        + 0.10 * strategy_score
        + 0.07 * price_score
        + 0.05 * month_score
    )

    return {
        "similarity_score": float(total_score),
        "product_score": float(product_score),
        "use_case_score": float(use_case_score),
        "target_group_score": float(target_group_score),
        "flavour_score": float(flavour_score),
        "product_form_score": float(product_form_score),
        "strategy_score": float(strategy_score),
        "price_score": float(price_score),
        "month_score": float(month_score),
    }


def find_similar_launches(
    product_name,
    use_case,
    target_group,
    flavour,
    product_form,
    launch_month,
    launch_strategy_type,
    uvp,
    top_n=7,
):
    scored_rows = []

    for _, row in LAUNCHES.iterrows():
        scores = score_launch_similarity(
            row=row,
            product_name=product_name,
            use_case=use_case,
            target_group=target_group,
            flavour=flavour,
            product_form=product_form,
            launch_month=launch_month,
            launch_strategy_type=launch_strategy_type,
            uvp=uvp,
        )

        row_dict = row.to_dict()
        row_dict.update(scores)
        scored_rows.append(row_dict)

    scored_df = pd.DataFrame(scored_rows)
    scored_df = scored_df.sort_values("similarity_score", ascending=False)

    return scored_df.head(top_n).copy()


# ============================================================
# FORECAST ENGINE
# ============================================================

def weighted_average(values, weights):
    values = pd.Series(values, dtype="float")
    weights = pd.Series(weights, dtype="float")

    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return float(np.average(values[valid], weights=weights[valid]))


def calculate_sale_overlap_for_new_launch(launch_date):
    """
    Calculates sale overlap for a future launch date.
    """

    if pd.isna(launch_date):
        return {
            "launch_during_sale": 0,
            "first_week_sale_days": 0,
            "first_6_week_sale_days": 0,
            "sale_name_overlap": "",
        }

    launch_date = pd.to_datetime(launch_date)
    first_week_end = launch_date + pd.Timedelta(days=6)
    first_6w_end = launch_date + pd.Timedelta(days=41)

    launch_during_sale = 0
    first_week_days = 0
    first_6w_days = 0
    overlap_names = []

    for _, sale in SALE_TIMES.iterrows():
        sale_start = pd.to_datetime(sale["start_d"])
        sale_end = pd.to_datetime(sale["end_d"])

        if sale_start <= launch_date <= sale_end:
            launch_during_sale = 1
            overlap_names.append(str(sale["name"]))

        overlap_1w_start = max(launch_date, sale_start)
        overlap_1w_end = min(first_week_end, sale_end)

        if overlap_1w_start <= overlap_1w_end:
            first_week_days += (overlap_1w_end - overlap_1w_start).days + 1
            overlap_names.append(str(sale["name"]))

        overlap_6w_start = max(launch_date, sale_start)
        overlap_6w_end = min(first_6w_end, sale_end)

        if overlap_6w_start <= overlap_6w_end:
            first_6w_days += (overlap_6w_end - overlap_6w_start).days + 1
            overlap_names.append(str(sale["name"]))

    return {
        "launch_during_sale": launch_during_sale,
        "first_week_sale_days": first_week_days,
        "first_6_week_sale_days": first_6w_days,
        "sale_name_overlap": ", ".join(sorted(set(overlap_names))),
    }


def calculate_price_factor(uvp):
    avg_uvp = LAUNCHES["uvp"].dropna().mean()

    if pd.isna(uvp) or uvp <= 0 or pd.isna(avg_uvp) or avg_uvp <= 0:
        return 1.0

    raw_factor = (uvp / avg_uvp) ** PRICE_ELASTICITY

    return clip_factor(raw_factor, 0.75, 1.25)


def calculate_metric_forecast(
    metric,
    similar_launches,
    launch_month,
    launch_strategy_type,
    flavour,
    product_form,
    uvp,
    launch_date,
):
    """
    Calculates forecast for one metric.
    """

    # 1. Similar launch baseline.
    base = weighted_average(
        similar_launches[metric],
        similar_launches["similarity_score"],
    )

    if pd.isna(base):
        base = LAUNCHES[metric].mean()

    # 2. Factors.
    seasonality_factor = SEASONALITY_INDEX.get(int(launch_month), 1.0)

    strategy = normalize_strategy(launch_strategy_type)
    strategy_factor = (
        STRATEGY_FACTORS
        .get(strategy, {})
        .get(metric, 1.0)
    )

    flavour_factor = get_factor_from_dict(
        FLAVOUR_FACTORS,
        flavour,
        metric,
        default=1.0,
    )

    product_form_factor = get_factor_from_dict(
        PRODUCT_FORM_FACTORS,
        product_form,
        metric,
        default=1.0,
    )

    price_factor = calculate_price_factor(uvp)

    growth_factor = GROWTH_CONTEXT.get("company_growth_factor", 1.0)

    sale_overlap = calculate_sale_overlap_for_new_launch(launch_date)

    sale_factor = 1.0

    if sale_overlap["launch_during_sale"] == 1:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("launch_during_sale", 1.0)

    if "first_week" in metric and sale_overlap["first_week_sale_days"] > 0:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("first_week_sale_overlap", 1.0)

    if "first_6_week" in metric and sale_overlap["first_6_week_sale_days"] > 0:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("first_6_week_sale_overlap", 1.0)

    sale_factor = clip_factor(sale_factor, 0.7, 1.5)

    # 3. Final adjusted forecast.
    adjusted = (
        base
        * seasonality_factor
        * strategy_factor
        * flavour_factor
        * product_form_factor
        * price_factor
        * growth_factor
        * sale_factor
    )

    # Keep individual metrics stable.
    adjusted = max(0, adjusted)

    factor_details = {
        "base": base,
        "seasonality_factor": seasonality_factor,
        "strategy_factor": strategy_factor,
        "flavour_factor": flavour_factor,
        "product_form_factor": product_form_factor,
        "price_factor": price_factor,
        "growth_factor": growth_factor,
        "sale_factor": sale_factor,
    }

    return adjusted, factor_details


def calculate_confidence(similar_launches):
    """
    Simple confidence score based on top similarity and amount of usable references.
    """

    if similar_launches.empty:
        return 0.35

    top_score = float(similar_launches["similarity_score"].max())
    avg_top_3 = float(similar_launches.head(3)["similarity_score"].mean())

    usable_refs = int((similar_launches["similarity_score"] >= 0.25).sum())
    ref_bonus = min(0.15, usable_refs * 0.025)

    confidence = 0.35 + 0.35 * top_score + 0.15 * avg_top_3 + ref_bonus

    return float(np.clip(confidence, 0.30, 0.90))


def scenario_bounds(base_forecast, confidence):
    """
    Creates low/base/high range.
    Lower confidence gives wider interval.
    """

    uncertainty = 0.35 - (confidence - 0.30) * 0.25
    uncertainty = float(np.clip(uncertainty, 0.12, 0.35))

    low = base_forecast * (1 - uncertainty)
    high = base_forecast * (1 + uncertainty)

    return int(round(low)), int(round(base_forecast)), int(round(high))


def build_explanation(
    product_name,
    similar_launches,
    factor_details_by_metric,
    confidence,
    sale_overlap,
):
    top = similar_launches.head(3)

    lines = []

    lines.append(f"Forecast generated for: {product_name}")
    lines.append("")
    lines.append("Top similar historical launches:")

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i}. {row.get('product', 'N/A')} - {row.get('flavour', '')} "
            f"({row.get('launch_strategy_type', '')}) | similarity: {row['similarity_score']:.2f}"
        )

    lines.append("")
    lines.append("Main forecast drivers:")

    fwq = factor_details_by_metric["first_week_quantity"]

    lines.append(f"- Seasonality factor: {fwq['seasonality_factor']:.2f}")
    lines.append(f"- Launch strategy factor: {fwq['strategy_factor']:.2f}")
    lines.append(f"- Flavour factor: {fwq['flavour_factor']:.2f}")
    lines.append(f"- Product form factor: {fwq['product_form_factor']:.2f}")
    lines.append(f"- Price factor: {fwq['price_factor']:.2f}")
    lines.append(f"- Company growth factor: {fwq['growth_factor']:.2f}")
    lines.append(f"- Sale factor: {fwq['sale_factor']:.2f}")

    if sale_overlap["sale_name_overlap"]:
        lines.append("")
        lines.append(f"Sale overlap detected: {sale_overlap['sale_name_overlap']}")
        lines.append(f"- First week sale days: {sale_overlap['first_week_sale_days']}")
        lines.append(f"- First 6 week sale days: {sale_overlap['first_6_week_sale_days']}")

    lines.append("")
    lines.append(f"Confidence score: {confidence:.2f}")

    if confidence < 0.50:
        lines.append("Confidence note: Low. Similar historical references are weak or limited.")
    elif confidence < 0.70:
        lines.append("Confidence note: Medium. Forecast is usable but should be reviewed with business context.")
    else:
        lines.append("Confidence note: Good. Similar historical references are relatively strong.")

    return "\n".join(lines)


def run_forecast(
    product_name,
    use_case,
    target_group,
    launch_month_name,
    launch_date,
    flavour,
    product_form,
    uvp,
    launch_strategy_type,
    first_order_quantity,
):
    launch_month = MONTH_MAP.get(launch_month_name, 1)
    strategy = normalize_strategy(launch_strategy_type)

    if not product_name:
        product_name = "New Product"

    uvp = to_float(uvp, default=np.nan)
    if pd.isna(uvp) or uvp <= 0:
        uvp = LAUNCHES["uvp"].dropna().mean()

    first_order_quantity = to_float(first_order_quantity, default=np.nan)
    if pd.isna(first_order_quantity):
        first_order_quantity = 0

    launch_date_parsed = pd.to_datetime(launch_date, errors="coerce")

    if pd.isna(launch_date_parsed):
        # If exact launch date is missing, use first day of selected month in next available year.
        current_year = datetime.now().year
        launch_date_parsed = pd.Timestamp(year=current_year, month=launch_month, day=1)

    similar_launches = find_similar_launches(
        product_name=product_name,
        use_case=use_case,
        target_group=target_group,
        flavour=flavour,
        product_form=product_form,
        launch_month=launch_month,
        launch_strategy_type=strategy,
        uvp=uvp,
        top_n=7,
    )

    forecasts = {}
    factor_details_by_metric = {}

    for metric in TARGET_METRICS:
        value, factor_details = calculate_metric_forecast(
            metric=metric,
            similar_launches=similar_launches,
            launch_month=launch_month,
            launch_strategy_type=strategy,
            flavour=flavour,
            product_form=product_form,
            uvp=uvp,
            launch_date=launch_date_parsed,
        )

        forecasts[metric] = value
        factor_details_by_metric[metric] = factor_details

    confidence = calculate_confidence(similar_launches)

    fw_qty_low, fw_qty_base, fw_qty_high = scenario_bounds(
        forecasts["first_week_quantity"],
        confidence,
    )

    six_qty_low, six_qty_base, six_qty_high = scenario_bounds(
        forecasts["first_6_week_quantity"],
        confidence,
    )

    fw_nc_low, fw_nc_base, fw_nc_high = scenario_bounds(
        forecasts["first_week_nc"],
        confidence,
    )

    six_nc_low, six_nc_base, six_nc_high = scenario_bounds(
        forecasts["first_6_week_nc"],
        confidence,
    )

    stock_coverage = safe_divide(
        first_order_quantity,
        forecasts["first_6_week_quantity"],
        default=np.nan,
    )

    if pd.isna(stock_coverage) or first_order_quantity <= 0:
        stock_risk = "No stock quantity provided"
    elif stock_coverage < 0.80:
        stock_risk = "High stockout risk"
    elif stock_coverage <= 1.20:
        stock_risk = "Balanced stock coverage"
    else:
        stock_risk = "Potential overstock risk"

    # Output table.
    forecast_table = pd.DataFrame([
        {
            "Metric": "First week quantity",
            "Low": fw_qty_low,
            "Base": fw_qty_base,
            "High": fw_qty_high,
        },
        {
            "Metric": "First 6 week quantity",
            "Low": six_qty_low,
            "Base": six_qty_base,
            "High": six_qty_high,
        },
        {
            "Metric": "First week new customers",
            "Low": fw_nc_low,
            "Base": fw_nc_base,
            "High": fw_nc_high,
        },
        {
            "Metric": "First 6 week new customers",
            "Low": six_nc_low,
            "Base": six_nc_base,
            "High": six_nc_high,
        },
        {
            "Metric": "First week total customers",
            "Low": "",
            "Base": int(round(forecasts["first_week_total_c"])),
            "High": "",
        },
        {
            "Metric": "First 6 week total customers",
            "Low": "",
            "Base": int(round(forecasts["first_6_week_total_c"])),
            "High": "",
        },
        {
            "Metric": "Stock coverage ratio",
            "Low": "",
            "Base": "" if pd.isna(stock_coverage) else round(stock_coverage, 2),
            "High": stock_risk,
        },
    ])

    # Similar launch table.
    similar_table_cols = [
        "sku",
        "product",
        "flavour",
        "product_form",
        "launch_strategy_type",
        "launch_date",
        "uvp",
        "first_week_quantity",
        "first_6_week_quantity",
        "first_week_nc",
        "first_6_week_nc",
        "similarity_score",
    ]

    similar_table = similar_launches[similar_table_cols].copy()
    similar_table["similarity_score"] = similar_table["similarity_score"].round(3)
    similar_table["launch_date"] = pd.to_datetime(similar_table["launch_date"]).dt.date.astype(str)

    # Factor table.
    factor_rows = []

    for metric in ["first_week_quantity", "first_6_week_quantity", "first_week_nc", "first_6_week_nc"]:
        details = factor_details_by_metric[metric]
        factor_rows.append({
            "Metric": metric,
            "Similar launch baseline": round(details["base"], 1),
            "Seasonality": round(details["seasonality_factor"], 2),
            "Strategy": round(details["strategy_factor"], 2),
            "Flavour": round(details["flavour_factor"], 2),
            "Product form": round(details["product_form_factor"], 2),
            "Price": round(details["price_factor"], 2),
            "Growth": round(details["growth_factor"], 2),
            "Sale": round(details["sale_factor"], 2),
        })

    factor_table = pd.DataFrame(factor_rows)

    target_group_table = build_target_group_table(
        product_name=product_name,
        use_case=use_case,
        flavour=flavour,
        launch_strategy_type=strategy,
        uvp=uvp,
    )

    # Chart.
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["First week qty", "First 6 week qty", "First week NC", "First 6 week NC"],
        y=[fw_qty_base, six_qty_base, fw_nc_base, six_nc_base],
        name="Base forecast",
    ))

    fig.update_layout(
        title=f"Launch Forecast | Confidence: {confidence:.2f}",
        xaxis_title="Metric",
        yaxis_title="Forecast",
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.35,
        margin=dict(l=40, r=30, t=70, b=50),
    )

    sale_overlap = calculate_sale_overlap_for_new_launch(launch_date_parsed)

    explanation = build_explanation(
        product_name=product_name,
        similar_launches=similar_launches,
        factor_details_by_metric=factor_details_by_metric,
        confidence=confidence,
        sale_overlap=sale_overlap,
    )

    summary_text = (
        f"Base first week quantity: {fw_qty_base}\n"
        f"Base first 6 week quantity: {six_qty_base}\n"
        f"Base first week NC: {fw_nc_base}\n"
        f"Base first 6 week NC: {six_nc_base}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Stock risk: {stock_risk}"
    )

    return (
        forecast_table,
        target_group_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
    )


# ============================================================
# GRADIO UI
# ============================================================

CSS = """
body, .gradio-container {
    font-family: Inter, Arial, sans-serif !important;
    background: #f6f7f9 !important;
}

.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #555;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin: 10px 0 6px 0;
}

.header-box {
    background: white;
    border: 1px solid #e7e7e7;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
}

.small-muted {
    font-size: 0.78rem;
    color: #777;
}
"""


with gr.Blocks(css=CSS, title="Demand Forecasting Agent V2") as demo:
    gr.HTML(
        f"""
        <div class="header-box">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:1.35rem; font-weight:800; color:#111;">
                        Demand Forecasting Agent V2
                    </div>
                    <div class="small-muted">
                        Similar-launch based forecast with seasonality, strategy, flavour, price, sale and growth adjustments
                    </div>
                </div>
                <div class="small-muted" style="text-align:right;">
                    Artifact version: {METADATA.get("version", "N/A")}<br>
                    Created: {METADATA.get("created_at", "N/A")}<br>
                    Historical launches: {METADATA.get("launch_count", "N/A")}
                </div>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='section-title'>New Launch Input</div>")

            inp_product_name = gr.Textbox(
                label="Product name",
                value="Daily Carnitin",
            )

            inp_use_case = gr.Textbox(
                label="Use case",
                value="For individuals who train a lot and sporty people",
                lines=3,
            )

            inp_target_group = gr.Textbox(
                label="Target group",
                value="Sporty people, active individuals, fitness-focused consumers",
                lines=3,
            )

            with gr.Row():
                inp_month = gr.Dropdown(
                    label="Launch month",
                    choices=list(MONTH_MAP.keys()),
                    value="Dec",
                )

                inp_launch_date = gr.Textbox(
                    label="Launch date",
                    value="2026-12-01",
                    placeholder="YYYY-MM-DD",
                )

            with gr.Row():
                inp_flavour = gr.Textbox(
                    label="Flavour",
                    value="Orange",
                )

                inp_product_form = gr.Textbox(
                    label="Product form",
                    value="Capsules",
                )

            with gr.Row():
                inp_uvp = gr.Number(
                    label="UVP",
                    value=29.90,
                    minimum=0.1,
                )

                inp_first_order_quantity = gr.Number(
                    label="First order quantity / launch stock",
                    value=3000,
                    minimum=0,
                )

            inp_strategy = gr.Dropdown(
                label="Launch strategy type",
                choices=["standard", "co_creation", "limited_edition"],
                value="standard",
            )

            btn = gr.Button("Generate Forecast", variant="primary")

            gr.HTML("<div class='section-title'>Summary</div>")
            out_summary = gr.Textbox(
                label="",
                lines=7,
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.HTML("<div class='section-title'>Forecast Output</div>")

            out_forecast_table = gr.DataFrame(
                label="Forecast table",
                interactive=False,
                wrap=True,
            )

            gr.HTML("<div class='section-title'>Target Group Recommendation</div>")
            out_target_group_table = gr.DataFrame(
                label="Target-group inference",
                interactive=False,
                wrap=True,
            )

            out_chart = gr.Plot(label="Forecast chart")

            gr.HTML("<div class='section-title'>Forecast Explanation</div>")

            out_explanation = gr.Textbox(
                label="",
                lines=14,
                interactive=False,
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='section-title'>Similar Historical Launches</div>")
            out_similar_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=1):
            gr.HTML("<div class='section-title'>Applied Factors</div>")
            out_factor_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True,
            )

    btn.click(
        fn=run_forecast,
        inputs=[
            inp_product_name,
            inp_use_case,
            inp_target_group,
            inp_month,
            inp_launch_date,
            inp_flavour,
            inp_product_form,
            inp_uvp,
            inp_strategy,
            inp_first_order_quantity,
        ],
        outputs=[
            out_forecast_table,
            out_target_group_table,
            out_chart,
            out_similar_table,
            out_factor_table,
            out_explanation,
            out_summary,
        ],
    )


if __name__ == "__main__":
    demo.launch(debug=True)