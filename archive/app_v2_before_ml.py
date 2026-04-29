import os
import re
import pickle
import unicodedata
import uuid
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
LOG_DIR = "data/feedback"
FORECAST_RUN_LOG_PATH = os.path.join(LOG_DIR, "forecast_run_log.csv")

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]

PRODUCT_FORM_CHOICES = [
    "Capsules",
    "Drinking powder",
    "Oils",
    "Sprays",
    "Gummies",
]

PRODUCT_FORM_MODEL_VALUE_MAP = {
    "Capsules": "Capsules",
    "Drinking powder": "Drinking Powder",
    "Oils": "Oil",
    "Sprays": "Spray",
    "Gummies": "Gummies",
}

LAUNCH_STRATEGY_CHOICES = [
    "standard",
    "co_creation",
    "limited_edition",
]


# ============================================================
# BASIC HELPERS
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
    if pd.isna(x) or not str(x).strip():
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


def canonical_product_form(product_form):
    return PRODUCT_FORM_MODEL_VALUE_MAP.get(product_form, product_form)


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


def to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


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


def weighted_average(values, weights):
    values = pd.Series(values, dtype="float")
    weights = pd.Series(weights, dtype="float")

    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return float(np.average(values[valid], weights=weights[valid]))


def normalize_keyword_list(x):
    if x is None:
        return []

    if isinstance(x, list):
        raw_items = x
    else:
        raw_items = str(x).split(",")

    return sorted(
        {
            normalize_text(item).replace(" ", "_")
            for item in raw_items
            if str(item).strip()
        }
    )


def keyword_overlap_score(a, b):
    a_set = set(normalize_keyword_list(a))
    b_set = set(normalize_keyword_list(b))

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def get_factor_from_dict(factor_dict, key, metric, default=1.0):
    key = normalize_text(key)

    if not key:
        return default

    if key in factor_dict and metric in factor_dict[key]:
        return factor_dict[key][metric]

    return default


def build_future_month_year_choices(n_months=12):
    today = pd.Timestamp.today().normalize()
    start = (today + pd.offsets.MonthBegin(1)).replace(day=1)

    return [
        (start + pd.DateOffset(months=i)).strftime("%m-%Y")
        for i in range(n_months)
    ]


def parse_launch_month_year(launch_month_year):
    dt = pd.to_datetime(f"01-{launch_month_year}", format="%d-%m-%Y", errors="coerce")

    next_month = (
        pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1)
    ).replace(day=1)

    if pd.isna(dt):
        return next_month

    dt = dt.replace(day=1)

    if dt < next_month:
        return next_month

    return dt


def format_historical_launch_choice(row):
    launch_date = pd.to_datetime(row.get("launch_date"), errors="coerce")
    launch_date_text = "unknown-date"

    if pd.notna(launch_date):
        launch_date_text = launch_date.date().isoformat()

    return f"{row.get('sku', '')} | {row.get('product', '')} | {launch_date_text}"


def historical_launch_choice_to_sku(choice):
    if pd.isna(choice) or not str(choice).strip():
        return ""

    return str(choice).split("|", 1)[0].strip()


def normalize_review_value(x):
    x_norm = normalize_text(x)
    if x_norm in {"yes", "y", "true", "ok", "pass"}:
        return "yes"
    if x_norm in {"no", "n", "false", "fail"}:
        return "no"
    return "needs_review"


def append_forecast_run_log(payload):
    os.makedirs(LOG_DIR, exist_ok=True)
    row_df = pd.DataFrame([payload])

    if not os.path.exists(FORECAST_RUN_LOG_PATH):
        row_df.to_csv(FORECAST_RUN_LOG_PATH, index=False)
    else:
        row_df.to_csv(FORECAST_RUN_LOG_PATH, mode="a", index=False, header=False)


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
LAUNCH_RATIO_TABLE = ARTIFACTS["data"].get("launch_ratio_table", pd.DataFrame()).copy()
MONTHLY_NEW_CUSTOMERS = ARTIFACTS["data"].get("monthly_new_customers", pd.DataFrame()).copy()

CALIBRATION = ARTIFACTS["calibration"]
METADATA = ARTIFACTS["metadata"]

SEASONALITY_INDEX = CALIBRATION["seasonality_index"]
STRATEGY_FACTORS = CALIBRATION["strategy_factors"]
FLAVOUR_FACTORS = CALIBRATION["flavour_factors"]
PRODUCT_FORM_FACTORS = CALIBRATION["product_form_factors"]
SALE_FACTORS = CALIBRATION["sale_factors"]
PRICE_ELASTICITY = CALIBRATION["price_elasticity"]
RATIO_CONTEXT = CALIBRATION.get("ratio_context", {})

TARGET_GROUP_INFERENCE = ARTIFACTS.get("target_group_inference", {})
BEHAVIORAL_SEGMENTATION = ARTIFACTS.get("behavioral_segmentation", {})

LAUNCHES["sku"] = LAUNCHES["sku"].astype(str)

if not LAUNCH_RATIO_TABLE.empty and "sku" in LAUNCH_RATIO_TABLE.columns:
    LAUNCH_RATIO_TABLE["sku"] = LAUNCH_RATIO_TABLE["sku"].astype(str)

HISTORICAL_LAUNCH_CHOICES = []

if not LAUNCHES.empty and {"sku", "product", "launch_date"}.issubset(LAUNCHES.columns):
    historical_launches_sorted = LAUNCHES.sort_values(
        ["launch_date", "sku"], ascending=[False, True]
    )
    HISTORICAL_LAUNCH_CHOICES = [
        format_historical_launch_choice(row)
        for _, row in historical_launches_sorted.iterrows()
    ]


# ============================================================
# CHOICES FROM ARTIFACT DATA
# ============================================================

def build_unique_choices(df, col):
    if col not in df.columns:
        return []

    return sorted(
        [
            str(x).strip()
            for x in df[col].dropna().unique().tolist()
            if str(x).strip()
        ]
    )


def split_keywords(x):
    if pd.isna(x):
        return []

    return [
        k.strip()
        for k in str(x).split(",")
        if k.strip()
    ]


def build_keyword_choices(df, col="benefit_keywords"):
    if col not in df.columns:
        return []

    keywords = set()

    for value in df[col].dropna().tolist():
        for keyword in split_keywords(value):
            keywords.add(keyword)

    return sorted(keywords)


MONTH_YEAR_CHOICES = build_future_month_year_choices(12)

PRODUCT_NEED_AREA_CHOICES = build_unique_choices(LAUNCHES, "product_need_area")
BENEFIT_KEYWORD_CHOICES = build_keyword_choices(LAUNCHES, "benefit_keywords")
FLAVOUR_GROUP_CHOICES = build_unique_choices(LAUNCHES, "flavour_group")

FLAVOUR_CHOICES = build_unique_choices(LAUNCHES, "flavour")

if "New Flavour" not in FLAVOUR_CHOICES:
    FLAVOUR_CHOICES.append("New Flavour")


# SIMILARITY ENGINE
# ============================================================

def score_launch_similarity(
    row,
    product_name,
    product_need_area,
    benefit_keywords,
    flavour,
    flavour_group,
    product_form,
    launch_month,
    launch_strategy_type,
    uvp,
):
    product_score = max(
        token_similarity(product_name, row.get("product_norm", "")),
        token_similarity(product_name, row.get("artikel_name_norm", "")),
        string_similarity(product_name, row.get("product_norm", "")),
    )

    need_area_score = (
        1.0
        if normalize_text(product_need_area) == normalize_text(row.get("product_need_area_norm", row.get("product_need_area", "")))
        else 0.0
    )

    benefit_score = keyword_overlap_score(
        benefit_keywords,
        row.get("benefit_keywords_norm", row.get("benefit_keywords", "")),
    )

    flavour_score = max(
        token_similarity(flavour, row.get("flavour_norm", "")),
        string_similarity(flavour, row.get("flavour_norm", "")),
    )

    if normalize_text(flavour) == "new flavour":
        flavour_score = 0.0

    flavour_group_score = (
        1.0
        if normalize_text(flavour_group) == normalize_text(row.get("flavour_group_norm", row.get("flavour_group", "")))
        else 0.0
    )

    product_form_score = max(
        token_similarity(product_form, row.get("product_form_norm", "")),
        string_similarity(product_form, row.get("product_form_norm", "")),
    )

    strategy_score = strategy_similarity(
        launch_strategy_type,
        row.get("launch_strategy_type", "standard"),
    )

    price_score = price_similarity(
        uvp,
        row.get("uvp", np.nan),
    )

    month_score = month_circular_similarity(
        launch_month,
        row.get("launch_month", np.nan),
    )

    similarity_score = (
        0.30 * need_area_score
        + 0.25 * benefit_score
        + 0.12 * product_form_score
        + 0.10 * flavour_group_score
        + 0.08 * flavour_score
        + 0.07 * strategy_score
        + 0.04 * product_score
        + 0.02 * price_score
        + 0.02 * month_score
    )

    return {
        "similarity_score": float(similarity_score),
        "need_area_score": float(need_area_score),
        "benefit_score": float(benefit_score),
        "product_form_score": float(product_form_score),
        "flavour_group_score": float(flavour_group_score),
        "flavour_score": float(flavour_score),
        "strategy_score": float(strategy_score),
        "product_score": float(product_score),
        "price_score": float(price_score),
        "month_score": float(month_score),
    }


def find_similar_launches(
    product_name,
    product_need_area,
    benefit_keywords,
    flavour,
    flavour_group,
    product_form,
    launch_month,
    launch_strategy_type,
    uvp,
    top_n=7,
    exclude_sku=None,
    reference_launch_date=None,
):
    scored_rows = []

    if reference_launch_date is not None:
        reference_launch_date = pd.to_datetime(reference_launch_date, errors="coerce")

    for _, row in LAUNCHES.iterrows():
        row_sku = str(row.get("sku", ""))

        if exclude_sku is not None and row_sku == str(exclude_sku):
            continue

        if pd.notna(reference_launch_date):
            row_launch_date = pd.to_datetime(row.get("launch_date"), errors="coerce")

            if pd.notna(row_launch_date) and row_launch_date >= reference_launch_date:
                continue

        scores = score_launch_similarity(
            row=row,
            product_name=product_name,
            product_need_area=product_need_area,
            benefit_keywords=benefit_keywords,
            flavour=flavour,
            flavour_group=flavour_group,
            product_form=product_form,
            launch_month=launch_month,
            launch_strategy_type=launch_strategy_type,
            uvp=uvp,
        )

        row_dict = row.to_dict()
        row_dict.update(scores)
        scored_rows.append(row_dict)

    scored_df = pd.DataFrame(scored_rows)
    scored_df["sku"] = scored_df["sku"].astype(str)

    if not LAUNCH_RATIO_TABLE.empty:
        ratio_cols = [
            "sku",
            "eligible_customers_before_launch",
            "buyer_ratio_1w_existing",
            "buyer_ratio_6w_existing",
            "buyer_ratio_1w_existing_clipped",
            "buyer_ratio_6w_existing_clipped",
            "monthly_new_customers_at_launch",
            "nc_ratio_1w_vs_monthly_nc",
            "nc_ratio_6w_vs_monthly_nc",
            "nc_ratio_1w_vs_monthly_nc_clipped",
            "nc_ratio_6w_vs_monthly_nc_clipped",
            "units_per_customer_1w",
            "units_per_customer_6w",
            "units_per_customer_1w_clipped",
            "units_per_customer_6w_clipped",
            "flag_nc_ratio_too_high",
        ]

        existing_ratio_cols = [c for c in ratio_cols if c in LAUNCH_RATIO_TABLE.columns]

        scored_df = scored_df.merge(
            LAUNCH_RATIO_TABLE[existing_ratio_cols],
            on="sku",
            how="left",
        )

    scored_df = scored_df.sort_values("similarity_score", ascending=False)

    return scored_df.head(top_n).copy()


# ============================================================
# FORECAST ENGINE
# ============================================================

def calculate_sale_overlap_for_new_launch(launch_date):
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


def calculate_sale_factor(metric, launch_date):
    sale_overlap = calculate_sale_overlap_for_new_launch(launch_date)
    sale_factor = 1.0

    if sale_overlap["launch_during_sale"] == 1:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("launch_during_sale", 1.0)

    if "first_week" in metric and sale_overlap["first_week_sale_days"] > 0:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("first_week_sale_overlap", 1.0)

    if "first_6_week" in metric and sale_overlap["first_6_week_sale_days"] > 0:
        sale_factor *= SALE_FACTORS.get(metric, {}).get("first_6_week_sale_overlap", 1.0)

    return clip_factor(sale_factor, 0.7, 1.5), sale_overlap


def calculate_demand_factor(
    metric,
    launch_month,
    launch_strategy_type,
    flavour_group,
    product_form,
    uvp,
    launch_date,
    behavioral_segment_multiplier=1.0,
):
    seasonality_factor = SEASONALITY_INDEX.get(int(launch_month), 1.0)

    strategy = normalize_strategy(launch_strategy_type)
    strategy_factor = STRATEGY_FACTORS.get(strategy, {}).get(metric, 1.0)

    flavour_factor = get_factor_from_dict(
        FLAVOUR_FACTORS,
        flavour_group,
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
    sale_factor, sale_overlap = calculate_sale_factor(metric, launch_date)

    total_factor = (
        seasonality_factor
        * strategy_factor
        * flavour_factor
        * product_form_factor
        * price_factor
        * sale_factor
        * behavioral_segment_multiplier
    )

    total_factor = clip_factor(total_factor, 0.35, 2.50)

    return {
        "seasonality_factor": seasonality_factor,
        "strategy_factor": strategy_factor,
        "flavour_group_factor": flavour_factor,
        "product_form_factor": product_form_factor,
        "price_factor": price_factor,
        "sale_factor": sale_factor,
        "behavioral_segment_factor": behavioral_segment_multiplier,
        "total_factor": total_factor,
        "sale_overlap": sale_overlap,
    }


def weighted_ratio(similar_launches, clipped_col, raw_col, fallback):
    if clipped_col in similar_launches.columns:
        value = weighted_average(
            similar_launches[clipped_col],
            similar_launches["similarity_score"],
        )
    else:
        value = np.nan

    if pd.isna(value) and raw_col in similar_launches.columns:
        value = weighted_average(
            similar_launches[raw_col],
            similar_launches["similarity_score"],
        )

    if pd.isna(value):
        value = fallback

    return float(value)


def calculate_ratio_based_forecast(
    similar_launches,
    launch_month,
    launch_strategy_type,
    flavour_group,
    product_form,
    uvp,
    launch_date,
):
    behavioral_segment_multiplier, behavioral_segment_details = calculate_behavioral_segment_multiplier(
        similar_launches=similar_launches,
        launch_strategy_type=launch_strategy_type,
    )

    active_customer_base = float(
        RATIO_CONTEXT.get(
            "active_customer_count_12m",
            RATIO_CONTEXT.get("known_customer_count", 0),
        )
    )

    if active_customer_base <= 0:
        active_customer_base = float(RATIO_CONTEXT.get("known_customer_count", 1))

    recent_monthly_nc_base = float(
        RATIO_CONTEXT.get("recent_3m_avg_new_customers", 1.0)
    )

    if recent_monthly_nc_base <= 0:
        recent_monthly_nc_base = 1.0

    fallback_buyer_1w = 0.003
    fallback_buyer_6w = 0.007
    fallback_nc_1w = 0.06
    fallback_nc_6w = 0.11
    fallback_upc_1w = 1.30
    fallback_upc_6w = 1.30

    if not LAUNCH_RATIO_TABLE.empty:
        fallback_buyer_1w = float(
            LAUNCH_RATIO_TABLE["buyer_ratio_1w_existing_clipped"].median()
            if "buyer_ratio_1w_existing_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["buyer_ratio_1w_existing"].median()
        )

        fallback_buyer_6w = float(
            LAUNCH_RATIO_TABLE["buyer_ratio_6w_existing_clipped"].median()
            if "buyer_ratio_6w_existing_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["buyer_ratio_6w_existing"].median()
        )

        fallback_nc_1w = float(
            LAUNCH_RATIO_TABLE["nc_ratio_1w_vs_monthly_nc_clipped"].median()
            if "nc_ratio_1w_vs_monthly_nc_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["nc_ratio_1w_vs_monthly_nc"].median()
        )

        fallback_nc_6w = float(
            LAUNCH_RATIO_TABLE["nc_ratio_6w_vs_monthly_nc_clipped"].median()
            if "nc_ratio_6w_vs_monthly_nc_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["nc_ratio_6w_vs_monthly_nc"].median()
        )

        fallback_upc_1w = float(
            LAUNCH_RATIO_TABLE["units_per_customer_1w_clipped"].median()
            if "units_per_customer_1w_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["units_per_customer_1w"].median()
        )

        fallback_upc_6w = float(
            LAUNCH_RATIO_TABLE["units_per_customer_6w_clipped"].median()
            if "units_per_customer_6w_clipped" in LAUNCH_RATIO_TABLE.columns
            else LAUNCH_RATIO_TABLE["units_per_customer_6w"].median()
        )

    buyer_ratio_1w = weighted_ratio(
        similar_launches,
        "buyer_ratio_1w_existing_clipped",
        "buyer_ratio_1w_existing",
        fallback_buyer_1w,
    )

    buyer_ratio_6w = weighted_ratio(
        similar_launches,
        "buyer_ratio_6w_existing_clipped",
        "buyer_ratio_6w_existing",
        fallback_buyer_6w,
    )

    nc_ratio_1w = weighted_ratio(
        similar_launches,
        "nc_ratio_1w_vs_monthly_nc_clipped",
        "nc_ratio_1w_vs_monthly_nc",
        fallback_nc_1w,
    )

    nc_ratio_6w = weighted_ratio(
        similar_launches,
        "nc_ratio_6w_vs_monthly_nc_clipped",
        "nc_ratio_6w_vs_monthly_nc",
        fallback_nc_6w,
    )

    upc_1w = weighted_ratio(
        similar_launches,
        "units_per_customer_1w_clipped",
        "units_per_customer_1w",
        fallback_upc_1w,
    )

    upc_6w = weighted_ratio(
        similar_launches,
        "units_per_customer_6w_clipped",
        "units_per_customer_6w",
        fallback_upc_6w,
    )

    factors_existing_1w = calculate_demand_factor(
        metric="first_week_total_c",
        launch_month=launch_month,
        launch_strategy_type=launch_strategy_type,
        flavour_group=flavour_group,
        product_form=product_form,
        uvp=uvp,
        launch_date=launch_date,
        behavioral_segment_multiplier=behavioral_segment_multiplier,
    )

    factors_existing_6w = calculate_demand_factor(
        metric="first_6_week_total_c",
        launch_month=launch_month,
        launch_strategy_type=launch_strategy_type,
        flavour_group=flavour_group,
        product_form=product_form,
        uvp=uvp,
        launch_date=launch_date,
        behavioral_segment_multiplier=behavioral_segment_multiplier,
    )

    factors_nc_1w = calculate_demand_factor(
        metric="first_week_nc",
        launch_month=launch_month,
        launch_strategy_type=launch_strategy_type,
        flavour_group=flavour_group,
        product_form=product_form,
        uvp=uvp,
        launch_date=launch_date,
        behavioral_segment_multiplier=behavioral_segment_multiplier,
    )

    factors_nc_6w = calculate_demand_factor(
        metric="first_6_week_nc",
        launch_month=launch_month,
        launch_strategy_type=launch_strategy_type,
        flavour_group=flavour_group,
        product_form=product_form,
        uvp=uvp,
        launch_date=launch_date,
        behavioral_segment_multiplier=behavioral_segment_multiplier,
    )

    factors_existing_1w["behavioral_segment_details"] = behavioral_segment_details
    factors_existing_6w["behavioral_segment_details"] = behavioral_segment_details
    factors_nc_1w["behavioral_segment_details"] = behavioral_segment_details
    factors_nc_6w["behavioral_segment_details"] = behavioral_segment_details

    existing_1w_base = active_customer_base * buyer_ratio_1w
    existing_6w_base = active_customer_base * buyer_ratio_6w

    nc_1w_base = recent_monthly_nc_base * nc_ratio_1w
    nc_6w_base = recent_monthly_nc_base * nc_ratio_6w

    existing_1w = existing_1w_base * factors_existing_1w["total_factor"]
    existing_6w = existing_6w_base * factors_existing_6w["total_factor"]

    nc_1w = nc_1w_base * factors_nc_1w["total_factor"]
    nc_6w = nc_6w_base * factors_nc_6w["total_factor"]

    total_c_1w = max(0, existing_1w + nc_1w)
    total_c_6w = max(0, existing_6w + nc_6w)

    nc_1w = min(max(0, nc_1w), total_c_1w)
    nc_6w = min(max(0, nc_6w), total_c_6w)

    quantity_1w = total_c_1w * upc_1w
    quantity_6w = total_c_6w * upc_6w

    forecasts = {
        "first_week_quantity": quantity_1w,
        "first_6_week_quantity": quantity_6w,
        "first_week_nc": nc_1w,
        "first_6_week_nc": nc_6w,
        "first_week_total_c": total_c_1w,
        "first_6_week_total_c": total_c_6w,
        "first_week_existing_c": max(0, existing_1w),
        "first_6_week_existing_c": max(0, existing_6w),
    }

    factor_details = {}

    for metric, base, buyer_ratio, nc_ratio, upc, factors in [
        ("first_week_quantity", quantity_1w, buyer_ratio_1w, nc_ratio_1w, upc_1w, factors_existing_1w),
        ("first_6_week_quantity", quantity_6w, buyer_ratio_6w, nc_ratio_6w, upc_6w, factors_existing_6w),
        ("first_week_nc", nc_1w_base, buyer_ratio_1w, nc_ratio_1w, upc_1w, factors_nc_1w),
        ("first_6_week_nc", nc_6w_base, buyer_ratio_6w, nc_ratio_6w, upc_6w, factors_nc_6w),
        ("first_week_total_c", existing_1w_base + nc_1w_base, buyer_ratio_1w, nc_ratio_1w, upc_1w, factors_existing_1w),
        ("first_6_week_total_c", existing_6w_base + nc_6w_base, buyer_ratio_6w, nc_ratio_6w, upc_6w, factors_existing_6w),
    ]:
        factor_details[metric] = {
            "base": base,
            "forecast_logic": (
                "total customers × weighted units/customer"
                if "quantity" in metric
                else "existing customers + new customers"
                if "total_c" in metric
                else "recent monthly NC base × weighted NC ratio"
            ),
            "active_customer_base": active_customer_base,
            "recent_monthly_nc_base": recent_monthly_nc_base,
            "buyer_ratio": buyer_ratio,
            "nc_ratio": nc_ratio,
            "units_per_customer": upc,
            **factors,
        }

    return forecasts, factor_details


def calculate_confidence(similar_launches):
    if similar_launches.empty:
        return 0.35

    top_score = float(similar_launches["similarity_score"].max())
    avg_top_3 = float(similar_launches.head(3)["similarity_score"].mean())

    usable_refs = int((similar_launches["similarity_score"] >= 0.25).sum())
    ref_bonus = min(0.15, usable_refs * 0.025)

    ratio_bonus = 0.0
    if "buyer_ratio_6w_existing_clipped" in similar_launches.columns:
        usable_ratio_refs = int(similar_launches["buyer_ratio_6w_existing_clipped"].notna().sum())
        ratio_bonus = min(0.08, usable_ratio_refs * 0.01)

    confidence = 0.35 + 0.35 * top_score + 0.15 * avg_top_3 + ref_bonus + ratio_bonus

    return float(np.clip(confidence, 0.30, 0.92))


def scenario_bounds(base_forecast, confidence):
    uncertainty = 0.35 - (confidence - 0.30) * 0.25
    uncertainty = float(np.clip(uncertainty, 0.12, 0.35))

    low = base_forecast * (1 - uncertainty)
    high = base_forecast * (1 + uncertainty)

    return int(round(low)), int(round(base_forecast)), int(round(high))


# ============================================================
# OUTPUT BUILDERS
# ============================================================

def build_behavioral_segment_table(
    similar_launches,
    fw_qty_base,
    six_qty_base,
    fw_nc_base,
    six_nc_base,
):
    if not BEHAVIORAL_SEGMENTATION.get("enabled", False):
        return pd.DataFrame(
            [
                {
                    "Segment": "N/A",
                    "Segment label": "N/A",
                    "Affinity score": np.nan,
                    "First week units (base)": np.nan,
                    "First 6 week units (base)": np.nan,
                    "First week NC (base)": np.nan,
                    "First 6 week NC (base)": np.nan,
                    "Interpretation": BEHAVIORAL_SEGMENTATION.get(
                        "reason",
                        "Behavioral segmentation not available",
                    ),
                }
            ]
        )

    profile = BEHAVIORAL_SEGMENTATION.get("launch_segment_profile")
    seg_summary = BEHAVIORAL_SEGMENTATION.get("segment_summary")

    if profile is None or seg_summary is None or profile.empty:
        return pd.DataFrame(
            [
                {
                    "Segment": "N/A",
                    "Segment label": "N/A",
                    "Affinity score": np.nan,
                    "First week units (base)": np.nan,
                    "First 6 week units (base)": np.nan,
                    "First week NC (base)": np.nan,
                    "First 6 week NC (base)": np.nan,
                    "Interpretation": "Segment profile table is empty",
                }
            ]
        )

    ref = similar_launches[["sku", "similarity_score"]].copy()
    ref["sku"] = ref["sku"].astype(str)
    ref["similarity_score"] = pd.to_numeric(ref["similarity_score"], errors="coerce").fillna(0.0)
    ref = ref[ref["similarity_score"] > 0]

    profile = profile.copy()
    profile["sku"] = profile["sku"].astype(str)

    merged = ref.merge(profile, on="sku", how="left")
    merged = merged.dropna(subset=["segment_key"]).copy()

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "Segment": "N/A",
                    "Segment label": "N/A",
                    "Affinity score": np.nan,
                    "First week units (base)": np.nan,
                    "First 6 week units (base)": np.nan,
                    "First week NC (base)": np.nan,
                    "First 6 week NC (base)": np.nan,
                    "Interpretation": "No segment affinity from similar launches",
                }
            ]
        )

    merged["weighted_share"] = merged["similarity_score"] * merged["launch_segment_share"]

    agg = (
        merged.groupby("segment_key", as_index=False)
        .agg(weighted_share=("weighted_share", "sum"))
        .sort_values("weighted_share", ascending=False)
    )

    total = max(agg["weighted_share"].sum(), 1e-9)
    agg["affinity"] = agg["weighted_share"] / total

    agg["First week units (base)"] = (agg["affinity"] * fw_qty_base).round().astype(int)
    agg["First 6 week units (base)"] = (agg["affinity"] * six_qty_base).round().astype(int)
    agg["First week NC (base)"] = (agg["affinity"] * fw_nc_base).round().astype(int)
    agg["First 6 week NC (base)"] = (agg["affinity"] * six_nc_base).round().astype(int)

    needed_summary_cols = [
        "segment_key",
        "segment_label",
        "segment_description",
        "global_share",
        "avg_recency_days",
        "avg_frequency",
        "avg_monetary",
        "avg_sale_share",
        "avg_launch_purchase_count_24m",
        "avg_unique_launch_skus_24m",
        "avg_launch_share_24m",
        "avg_unique_product_count_24m",
        "avg_unique_flavour_count_24m",
        "avg_product_diversity_ratio_24m",
    ]

    existing_summary_cols = [c for c in needed_summary_cols if c in seg_summary.columns]
    seg_summary_small = seg_summary[existing_summary_cols].copy()

    agg = agg.merge(seg_summary_small, on="segment_key", how="left")

    agg["Segment"] = agg["segment_key"]
    agg["Segment label"] = agg.get("segment_label", "")
    agg["Affinity score"] = agg["affinity"].round(3)
    agg["Global customer share"] = agg.get("global_share", 0).fillna(0).round(3)

    if "global_share" in agg.columns:
        agg["Interpretation"] = np.where(
            agg["affinity"] >= agg["global_share"].fillna(0),
            "Over-indexed vs global mix",
            "Under-indexed vs global mix",
        )
    else:
        agg["Interpretation"] = ""

    final_cols = [
        "Segment",
        "Segment label",
        "Affinity score",
        "Global customer share",
        "First week units (base)",
        "First 6 week units (base)",
        "First week NC (base)",
        "First 6 week NC (base)",
        "Interpretation",
    ]

    existing_final_cols = [c for c in final_cols if c in agg.columns]

    return agg[existing_final_cols]


def calculate_behavioral_segment_multiplier(similar_launches, launch_strategy_type):
    if not BEHAVIORAL_SEGMENTATION.get("enabled", False):
        return 1.0, {
            "enabled": False,
            "reason": BEHAVIORAL_SEGMENTATION.get(
                "reason",
                "Behavioral segmentation not available",
            ),
        }

    profile = BEHAVIORAL_SEGMENTATION.get("launch_segment_profile")
    seg_summary = BEHAVIORAL_SEGMENTATION.get("segment_summary")

    if profile is None or seg_summary is None or profile.empty:
        return 1.0, {
            "enabled": False,
            "reason": "Segment profile table is empty",
        }

    ref = similar_launches[["sku", "similarity_score"]].copy()
    ref["sku"] = ref["sku"].astype(str)
    ref["similarity_score"] = pd.to_numeric(ref["similarity_score"], errors="coerce").fillna(0.0)
    ref = ref[ref["similarity_score"] > 0]

    profile = profile.copy()
    profile["sku"] = profile["sku"].astype(str)

    merged = ref.merge(profile, on="sku", how="left")
    merged = merged.dropna(subset=["segment_key"]).copy()

    if merged.empty:
        return 1.0, {
            "enabled": False,
            "reason": "No segment affinity from similar launches",
        }

    merged["weighted_share"] = merged["similarity_score"] * merged["launch_segment_share"]
    affinity = (
        merged.groupby("segment_key", as_index=False)
        .agg(weighted_share=("weighted_share", "sum"))
        .sort_values("weighted_share", ascending=False)
    )

    total = max(affinity["weighted_share"].sum(), 1e-9)
    affinity["affinity"] = affinity["weighted_share"] / total

    strategy = normalize_strategy(launch_strategy_type)
    if strategy == "co_creation":
        propensity_col = "avg_co_creation_purchase_count_24m"
    elif strategy == "limited_edition":
        propensity_col = "avg_limited_edition_purchase_count_24m"
    else:
        propensity_col = "avg_launch_purchase_count_24m"

    if propensity_col not in seg_summary.columns:
        return 1.0, {
            "enabled": False,
            "reason": f"Segment summary missing {propensity_col}",
        }

    global_propensity = float(seg_summary[propensity_col].mean())
    if not np.isfinite(global_propensity) or global_propensity <= 0:
        global_propensity = 1.0

    seg_small = seg_summary[["segment_key", propensity_col]].copy()
    affinity = affinity.merge(seg_small, on="segment_key", how="left")
    affinity[propensity_col] = pd.to_numeric(affinity[propensity_col], errors="coerce").fillna(global_propensity)
    affinity["propensity_ratio"] = affinity[propensity_col] / global_propensity

    multiplier = float((affinity["affinity"] * affinity["propensity_ratio"]).sum())
    multiplier = clip_factor(multiplier, 0.75, 1.35)

    details = {
        "enabled": True,
        "strategy": strategy,
        "propensity_col": propensity_col,
        "global_propensity": global_propensity,
        "multiplier": multiplier,
        "top_segments": affinity[["segment_key", "affinity", "propensity_ratio"]].head(5).to_dict("records"),
    }

    return multiplier, details


def build_explanation(
    product_name,
    product_need_area,
    benefit_keywords,
    flavour,
    flavour_group,
    similar_launches,
    factor_details_by_metric,
    confidence,
    sale_overlap,
):
    top = similar_launches.head(3)
    keyword_text = ", ".join(benefit_keywords) if isinstance(benefit_keywords, list) else str(benefit_keywords)

    lines = []

    lines.append(f"Forecast generated for: {product_name}")
    lines.append(f"Product need area: {product_need_area}")
    lines.append(f"Benefit keywords: {keyword_text}")
    lines.append(f"Flavour: {flavour}")
    lines.append(f"Flavour group: {flavour_group}")
    lines.append("")
    lines.append("Top similar historical launches:")

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i}. {row.get('product', 'N/A')} | "
            f"need area: {row.get('product_need_area', 'N/A')} | "
            f"flavour: {row.get('flavour', 'N/A')} | "
            f"group: {row.get('flavour_group', 'N/A')} | "
            f"score: {row.get('similarity_score', np.nan):.2f}"
        )

    fwq = factor_details_by_metric["first_week_quantity"]
    sixq = factor_details_by_metric["first_6_week_quantity"]

    lines.append("")
    lines.append("How this output is calculated:")
    lines.append("1) Similar historical launches are selected using structured product attributes.")
    lines.append("2) Matching is based on product need area, benefit keyword overlap, product form, flavour, flavour group, strategy, price, and launch month.")
    lines.append("3) Existing-customer demand is estimated from historical buyer penetration ratios.")
    lines.append("4) New-customer demand is estimated from historical NC ratios versus monthly NC scale.")
    lines.append("5) Quantity = total customers × weighted historical units per customer.")
    lines.append("6) Forecast is adjusted by seasonality, strategy, flavour group, product form, price, sale overlap, and behavioral segmentation.")

    lines.append("")
    lines.append("Main ratio drivers:")
    lines.append(f"- Active customer base used: {fwq['active_customer_base']:.0f}")
    lines.append(f"- Recent monthly NC base used: {fwq['recent_monthly_nc_base']:.0f}")
    lines.append(f"- Weighted first-week buyer ratio: {fwq['buyer_ratio']:.4f}")
    lines.append(f"- Weighted first-6-week buyer ratio: {sixq['buyer_ratio']:.4f}")
    lines.append(f"- Weighted first-week NC ratio vs monthly NC: {fwq['nc_ratio']:.4f}")
    lines.append(f"- Weighted first-6-week NC ratio vs monthly NC: {sixq['nc_ratio']:.4f}")
    lines.append(f"- Weighted units/customer first week: {fwq['units_per_customer']:.2f}")
    lines.append(f"- Weighted units/customer first 6 weeks: {sixq['units_per_customer']:.2f}")

    lines.append("")
    lines.append("Main adjustment factors:")
    lines.append(f"- Seasonality factor: {fwq['seasonality_factor']:.2f}")
    lines.append(f"- Launch strategy factor: {fwq['strategy_factor']:.2f}")
    lines.append(f"- Flavour group factor: {fwq['flavour_group_factor']:.2f}")
    lines.append(f"- Product form factor: {fwq['product_form_factor']:.2f}")
    lines.append(f"- Price factor: {fwq['price_factor']:.2f}")
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


def build_plausibility_table(
    demand_fit,
    supply_fit,
    confidence_fit,
    review_decision,
    reviewer_name,
    review_notes,
):
    return pd.DataFrame(
        [
            {
                "Review Criterion": "Demand estimate appears plausible",
                "Review Result": demand_fit,
                "Reviewer": reviewer_name,
                "Decision": review_decision,
                "Notes": review_notes,
            },
            {
                "Review Criterion": "Proposed first-order quantity is operationally feasible",
                "Review Result": supply_fit,
                "Reviewer": reviewer_name,
                "Decision": review_decision,
                "Notes": review_notes,
            },
            {
                "Review Criterion": "Confidence level is acceptable for decision",
                "Review Result": confidence_fit,
                "Reviewer": reviewer_name,
                "Decision": review_decision,
                "Notes": review_notes,
            },
        ]
    )


# ============================================================
# MAIN FORECAST FUNCTION
# ============================================================

def run_forecast(
    product_name,
    product_need_area,
    benefit_keywords,
    launch_month_year,
    product_form_ui,
    launch_strategy_type,
    uvp,
    flavour,
    flavour_group,
    assumption_market="",
    assumption_media="",
    assumption_channel="",
    assumption_supply="",
    assumption_override="",
    plausibility_demand_fit="Needs review",
    plausibility_supply_fit="Needs review",
    plausibility_confidence_fit="Needs review",
    review_decision="Needs revision",
    reviewer_name="",
    review_notes="",
    launch_date_override=None,
    historical_reference_sku=None,
):
    if not product_name:
        product_name = "New Product"

    if not product_need_area:
        product_need_area = ""

    if not benefit_keywords:
        benefit_keywords = []

    if not launch_month_year:
        launch_month_year = MONTH_YEAR_CHOICES[0] if MONTH_YEAR_CHOICES else None

    if not product_form_ui:
        product_form_ui = ""

    if not launch_strategy_type:
        launch_strategy_type = "standard"

    if not flavour:
        flavour = "New Flavour"

    if not flavour_group:
        flavour_group = ""

    if launch_date_override is not None:
        launch_date_parsed = pd.to_datetime(launch_date_override, errors="coerce")
    else:
        launch_date_parsed = pd.Timestamp("NaT")

    if pd.isna(launch_date_parsed):
        launch_date_parsed = parse_launch_month_year(launch_month_year)

    launch_month = int(launch_date_parsed.month)

    strategy = normalize_strategy(launch_strategy_type)
    product_form = canonical_product_form(product_form_ui)

    uvp = to_float(uvp, default=np.nan)
    if pd.isna(uvp) or uvp <= 0:
        uvp = LAUNCHES["uvp"].dropna().mean()

    similar_launches = find_similar_launches(
        product_name=product_name,
        product_need_area=product_need_area,
        benefit_keywords=benefit_keywords,
        flavour=flavour,
        flavour_group=flavour_group,
        product_form=product_form,
        launch_month=launch_month,
        launch_strategy_type=strategy,
        uvp=uvp,
        top_n=7,
        exclude_sku=historical_reference_sku,
        reference_launch_date=launch_date_parsed if historical_reference_sku else None,
    )

    forecasts, factor_details_by_metric = calculate_ratio_based_forecast(
        similar_launches=similar_launches,
        launch_month=launch_month,
        launch_strategy_type=strategy,
        flavour_group=flavour_group,
        product_form=product_form,
        uvp=uvp,
        launch_date=launch_date_parsed,
    )

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

    fw_total_low, fw_total_base, fw_total_high = scenario_bounds(
        forecasts["first_week_total_c"],
        confidence,
    )

    six_total_low, six_total_base, six_total_high = scenario_bounds(
        forecasts["first_6_week_total_c"],
        confidence,
    )

    fw_existing_base = int(round(forecasts["first_week_existing_c"]))
    six_existing_base = int(round(forecasts["first_6_week_existing_c"]))

    proposed_first_order_qty = int(round(six_qty_base * 1.10))
    proposed_rationale = "Base 6-week quantity + 10% safety buffer"

    stock_coverage = safe_divide(
        proposed_first_order_qty,
        forecasts["first_6_week_quantity"],
        default=np.nan,
    )

    stock_risk = "Recommended launch stock includes 10% safety buffer"

    forecast_table = pd.DataFrame(
        [
            {
                "Metric": "First week quantity (units)",
                "Worst case": fw_qty_low,
                "Base case": fw_qty_base,
                "Best case": fw_qty_high,
            },
            {
                "Metric": "First 6 week quantity (units)",
                "Worst case": six_qty_low,
                "Base case": six_qty_base,
                "Best case": six_qty_high,
            },
            {
                "Metric": "First week total customers",
                "Worst case": fw_total_low,
                "Base case": fw_total_base,
                "Best case": fw_total_high,
            },
            {
                "Metric": "First 6 week total customers",
                "Worst case": six_total_low,
                "Base case": six_total_base,
                "Best case": six_total_high,
            },
            {
                "Metric": "First week new customers",
                "Worst case": fw_nc_low,
                "Base case": fw_nc_base,
                "Best case": fw_nc_high,
            },
            {
                "Metric": "First 6 week new customers",
                "Worst case": six_nc_low,
                "Base case": six_nc_base,
                "Best case": six_nc_high,
            },
            {
                "Metric": "First week existing customers",
                "Worst case": "",
                "Base case": fw_existing_base,
                "Best case": "",
            },
            {
                "Metric": "First 6 week existing customers",
                "Worst case": "",
                "Base case": six_existing_base,
                "Best case": "",
            },
            {
                "Metric": "Recommended first-order quantity",
                "Worst case": "",
                "Base case": proposed_first_order_qty,
                "Best case": proposed_rationale,
            },
            {
                "Metric": "Recommended stock coverage ratio",
                "Worst case": "",
                "Base case": "" if pd.isna(stock_coverage) else round(stock_coverage, 2),
                "Best case": stock_risk,
            },
        ]
    )

    similar_table_cols = [
        "sku",
        "product",
        "product_need_area",
        "benefit_keywords",
        "flavour",
        "flavour_group",
        "product_form",
        "launch_strategy_type",
        "launch_date",
        "uvp",
        "first_week_quantity",
        "first_6_week_quantity",
        "first_week_nc",
        "first_6_week_nc",
        "buyer_ratio_6w_existing_clipped",
        "nc_ratio_6w_vs_monthly_nc_clipped",
        "units_per_customer_6w_clipped",
        "need_area_score",
        "benefit_score",
        "product_form_score",
        "flavour_group_score",
        "flavour_score",
        "strategy_score",
        "product_score",
        "price_score",
        "month_score",
        "similarity_score",
    ]

    existing_similar_cols = [col for col in similar_table_cols if col in similar_launches.columns]
    similar_table = similar_launches[existing_similar_cols].copy()

    for score_col in [
        "need_area_score",
        "benefit_score",
        "product_form_score",
        "flavour_group_score",
        "flavour_score",
        "strategy_score",
        "product_score",
        "price_score",
        "month_score",
        "similarity_score",
    ]:
        if score_col in similar_table.columns:
            similar_table[score_col] = similar_table[score_col].round(3)

    for ratio_col in [
        "buyer_ratio_6w_existing_clipped",
        "nc_ratio_6w_vs_monthly_nc_clipped",
        "units_per_customer_6w_clipped",
    ]:
        if ratio_col in similar_table.columns:
            similar_table[ratio_col] = similar_table[ratio_col].round(4)

    if "launch_date" in similar_table.columns:
        similar_table["launch_date"] = pd.to_datetime(similar_table["launch_date"]).dt.date.astype(str)

    factor_rows = []

    for metric in TARGET_METRICS:
        details = factor_details_by_metric[metric]
        factor_rows.append(
            {
                "Metric": metric,
                "Forecast logic": details["forecast_logic"],
                "Base before factors": round(details["base"], 1),
                "Active customer base": round(details["active_customer_base"], 0),
                "Recent monthly NC base": round(details["recent_monthly_nc_base"], 0),
                "Weighted buyer ratio": round(details["buyer_ratio"], 5),
                "Weighted NC ratio": round(details["nc_ratio"], 5),
                "Units/customer": round(details["units_per_customer"], 3),
                "Seasonality": round(details["seasonality_factor"], 2),
                "Strategy": round(details["strategy_factor"], 2),
                "Flavour group": round(details["flavour_group_factor"], 2),
                "Product form": round(details["product_form_factor"], 2),
                "Price": round(details["price_factor"], 2),
                "Sale": round(details["sale_factor"], 2),
                "Behavioral segment": round(details.get("behavioral_segment_factor", 1.0), 2),
                "Total factor": round(details["total_factor"], 2),
            }
        )

    factor_table = pd.DataFrame(factor_rows)

    
    behavioral_segment_table = build_behavioral_segment_table(
        similar_launches=similar_launches,
        fw_qty_base=fw_qty_base,
        six_qty_base=six_qty_base,
        fw_nc_base=fw_nc_base,
        six_nc_base=six_nc_base,
    )

    plausibility_table = build_plausibility_table(
        demand_fit=plausibility_demand_fit,
        supply_fit=plausibility_supply_fit,
        confidence_fit=plausibility_confidence_fit,
        review_decision=review_decision,
        reviewer_name=reviewer_name,
        review_notes=review_notes,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                "1W qty",
                "6W qty",
                "1W total C",
                "6W total C",
                "1W NC",
                "6W NC",
            ],
            y=[
                fw_qty_base,
                six_qty_base,
                fw_total_base,
                six_total_base,
                fw_nc_base,
                six_nc_base,
            ],
            name="Base forecast",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Ratio-Based Launch Forecast | Confidence: {confidence:.2f}",
            font=dict(color="#4a392d", size=18),
        ),
        xaxis_title="Metric",
        yaxis_title="Forecast",
        height=420,
        plot_bgcolor="#fffaf4",
        paper_bgcolor="#fffaf4",
        bargap=0.35,
        margin=dict(l=40, r=30, t=70, b=50),
        font=dict(color="#4a392d"),
    )

    sale_overlap = calculate_sale_overlap_for_new_launch(launch_date_parsed)

    explanation = build_explanation(
        product_name=product_name,
        product_need_area=product_need_area,
        benefit_keywords=benefit_keywords,
        flavour=flavour,
        flavour_group=flavour_group,
        similar_launches=similar_launches,
        factor_details_by_metric=factor_details_by_metric,
        confidence=confidence,
        sale_overlap=sale_overlap,
    )

    benefit_keywords_text = (
        ", ".join(benefit_keywords)
        if isinstance(benefit_keywords, list)
        else str(benefit_keywords)
    )

    summary_text = (
        f"Base first week quantity: {fw_qty_base}\n"
        f"Base first 6 week quantity: {six_qty_base}\n"
        f"Base first week total customers: {fw_total_base}\n"
        f"Base first 6 week total customers: {six_total_base}\n"
        f"Base first week new customers: {fw_nc_base}\n"
        f"Base first 6 week new customers: {six_nc_base}\n"
        f"Base first week existing customers: {fw_existing_base}\n"
        f"Base first 6 week existing customers: {six_existing_base}\n"
        f"Recommended first-order quantity: {proposed_first_order_qty}\n"
        f"Rationale: {proposed_rationale}\n"
        f"Product need area: {product_need_area}\n"
        f"Benefit keywords: {benefit_keywords_text}\n"
        f"Product form: {product_form_ui}\n"
        f"Flavour: {flavour}\n"
        f"Flavour group: {flavour_group}\n"
        f"Confidence: {confidence:.2f}"
    )

    run_id = f"FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


    reference_skus = (
        "|".join(similar_table["sku"].astype(str).head(5).tolist())
        if "sku" in similar_table.columns
        else ""
    )

    append_forecast_run_log(
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "product_name": product_name,
            "product_need_area": product_need_area,
            "benefit_keywords": benefit_keywords_text,
            "flavour": flavour,
            "flavour_group": flavour_group,
            "product_form_input": product_form_ui,
            "product_form_model": product_form,
            "uvp": uvp,
            "launch_strategy_type": strategy,
            "launch_date": str(launch_date_parsed.date()),
            "launch_month_year": launch_month_year,
            "launch_month": int(launch_month),
            "first_order_qty_proposed": int(proposed_first_order_qty),
            "first_order_qty_rationale": proposed_rationale,
            "recommended_stock_coverage_ratio": "" if pd.isna(stock_coverage) else round(stock_coverage, 4),
            "base_first_week_qty": int(fw_qty_base),
            "base_first_6w_qty": int(six_qty_base),
            "base_first_week_total_c": int(fw_total_base),
            "base_first_6w_total_c": int(six_total_base),
            "base_first_week_nc": int(fw_nc_base),
            "base_first_6w_nc": int(six_nc_base),
            "base_first_week_existing_c": int(fw_existing_base),
            "base_first_6w_existing_c": int(six_existing_base),
            "first_week_low": int(fw_qty_low),
            "first_week_high": int(fw_qty_high),
            "first_6w_low": int(six_qty_low),
            "first_6w_high": int(six_qty_high),
            "confidence": float(round(confidence, 4)),
            "stock_risk": stock_risk,
            "review_decision": review_decision,
            "reviewer_name": reviewer_name,
            "plausibility_demand_fit": normalize_review_value(plausibility_demand_fit),
            "plausibility_supply_fit": normalize_review_value(plausibility_supply_fit),
            "plausibility_confidence_fit": normalize_review_value(plausibility_confidence_fit),
            "review_notes": review_notes,
            "assumption_market": assumption_market,
            "assumption_media": assumption_media,
            "assumption_channel": assumption_channel,
            "assumption_supply": assumption_supply,
            "assumption_override": assumption_override,
            "reference_skus": reference_skus,
            "active_customer_base": RATIO_CONTEXT.get("active_customer_count_12m", ""),
            "recent_monthly_nc_base": RATIO_CONTEXT.get("recent_3m_avg_new_customers", ""),
        }
    )

    run_record_status = (
        f"Run ID: {run_id}\n"
        f"Review decision: {review_decision}\n"
        f"Saved to: {FORECAST_RUN_LOG_PATH}"
    )

    return (
        forecast_table,
        behavioral_segment_table,
        plausibility_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_record_status,
    )


# ============================================================
# GRADIO UI WRAPPER
# ============================================================

def run_forecast_ui(
    product_name,
    product_need_area,
    benefit_keywords,
    launch_month_year,
    product_form_ui,
    launch_strategy_type,
    uvp,
    flavour,
    flavour_group,
):
    result = run_forecast(
        product_name=product_name,
        product_need_area=product_need_area,
        benefit_keywords=benefit_keywords,
        launch_month_year=launch_month_year,
        product_form_ui=product_form_ui,
        launch_strategy_type=launch_strategy_type,
        uvp=uvp,
        flavour=flavour,
        flavour_group=flavour_group,
    )

    (
        forecast_table,
        behavioral_segment_table,
        _plausibility_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_record_status,
    ) = result

    return (
        forecast_table,
        behavioral_segment_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_record_status,
    )


def get_historical_launch_row(historical_launch_choice):
    sku = historical_launch_choice_to_sku(historical_launch_choice)

    if not sku:
        raise ValueError("Please select a historical launch.")

    match = LAUNCHES[LAUNCHES["sku"] == sku]

    if match.empty:
        raise ValueError(f"Historical launch not found for SKU: {sku}")

    return match.iloc[0]


def build_historical_comparison_table(actual_launch, forecast_table):
    metric_map = [
        ("First week quantity (units)", "first_week_quantity"),
        ("First 6 week quantity (units)", "first_6_week_quantity"),
        ("First week total customers", "first_week_total_c"),
        ("First 6 week total customers", "first_6_week_total_c"),
        ("First week new customers", "first_week_nc"),
        ("First 6 week new customers", "first_6_week_nc"),
    ]

    rows = []

    for forecast_label, actual_col in metric_map:
        actual_value = to_float(actual_launch.get(actual_col), default=np.nan)

        forecast_match = forecast_table[forecast_table["Metric"] == forecast_label]
        forecast_value = np.nan

        if not forecast_match.empty and "Base case" in forecast_match.columns:
            forecast_value = to_float(forecast_match.iloc[0]["Base case"], default=np.nan)

        error_value = np.nan
        error_pct = np.nan

        if not pd.isna(actual_value) and not pd.isna(forecast_value):
            error_value = forecast_value - actual_value
            error_pct = safe_divide(error_value, actual_value, default=np.nan) * 100

        rows.append(
            {
                "Metric": forecast_label,
                "Actual": actual_value,
                "Forecast": forecast_value,
                "Error": error_value,
                "Error %": error_pct,
            }
        )

    comparison_table = pd.DataFrame(rows)

    for col in ["Actual", "Forecast", "Error", "Error %"]:
        if col in comparison_table.columns:
            comparison_table[col] = comparison_table[col].round(2)

    return comparison_table


def run_historical_launch_test(historical_launch_choice):
    launch_row = get_historical_launch_row(historical_launch_choice)

    result = run_forecast(
        product_name=launch_row.get("product", ""),
        product_need_area=launch_row.get("product_need_area", ""),
        benefit_keywords=split_keywords(launch_row.get("benefit_keywords", "")),
        launch_month_year=pd.to_datetime(launch_row.get("launch_date")).strftime("%m-%Y"),
        product_form_ui=launch_row.get("product_form", ""),
        launch_strategy_type=launch_row.get("launch_strategy_type", "standard"),
        uvp=launch_row.get("uvp", np.nan),
        flavour=launch_row.get("flavour", ""),
        flavour_group=launch_row.get("flavour_group", ""),
        launch_date_override=launch_row.get("launch_date"),
        historical_reference_sku=launch_row.get("sku", ""),
    )

    (
        forecast_table,
        behavioral_segment_table,
        _plausibility_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_record_status,
    ) = result

    comparison_table = build_historical_comparison_table(launch_row, forecast_table)

    valid_errors = comparison_table["Error %"].dropna().abs()
    mape = float(valid_errors.mean()) if not valid_errors.empty else np.nan

    actual_summary_lines = [
        f"Historical launch SKU: {launch_row.get('sku', '')}",
        f"Historical launch date: {pd.to_datetime(launch_row.get('launch_date')).date()}",
        f"Actual first week quantity: {to_float(launch_row.get('first_week_quantity'), default=np.nan)}",
        f"Actual first 6 week quantity: {to_float(launch_row.get('first_6_week_quantity'), default=np.nan)}",
        f"Actual first week total customers: {to_float(launch_row.get('first_week_total_c'), default=np.nan)}",
        f"Actual first 6 week total customers: {to_float(launch_row.get('first_6_week_total_c'), default=np.nan)}",
        f"Actual first week new customers: {to_float(launch_row.get('first_week_nc'), default=np.nan)}",
        f"Actual first 6 week new customers: {to_float(launch_row.get('first_6_week_nc'), default=np.nan)}",
    ]

    if not pd.isna(mape):
        actual_summary_lines.append(f"MAPE across available metrics: {mape:.2f}%")

    historical_summary = "\n".join(actual_summary_lines)

    return (
        forecast_table,
        behavioral_segment_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        f"{summary_text}\n\n{historical_summary}",
        f"Historical launch test completed for {launch_row.get('sku', '')}. {run_record_status}",
        comparison_table,
        historical_summary,
    )


# ============================================================
# UI
# ============================================================

CSS = """
body, .gradio-container {
    font-family: Inter, Arial, sans-serif !important;
    background: linear-gradient(135deg, #fbf7f1 0%, #f3eadf 45%, #eadccb 100%) !important;
    color: #3f342c !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: auto !important;
}

.header-box {
    background: rgba(255, 252, 247, 0.92);
    border: 1px solid #e3d2bd;
    border-radius: 24px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 18px 45px rgba(121, 92, 63, 0.13);
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 1.55rem;
    font-weight: 900;
    color: #4a392d;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 0.86rem;
    color: #7f6754;
    margin-top: 4px;
}

.section-title {
    font-size: 0.78rem;
    font-weight: 800;
    color: #8a6f59;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin: 12px 0 8px 0;
}

.input-card, .output-card {
    background: rgba(255, 252, 247, 0.94);
    border: 1px solid #e2d1bd;
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 14px 35px rgba(121, 92, 63, 0.11);
}

.small-muted {
    font-size: 0.78rem;
    color: #8d7460;
}

button.primary {
    background: linear-gradient(90deg, #b89572 0%, #8d6f55 100%) !important;
    border: none !important;
    border-radius: 18px !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow: 0 12px 26px rgba(141, 111, 85, 0.24) !important;
}

textarea, input, select {
    border-radius: 16px !important;
    border-color: #d8c4ad !important;
    background: #fffaf4 !important;
    color: #3f342c !important;
}

label {
    color: #5a4637 !important;
    font-weight: 700 !important;
}

footer {
    visibility: hidden;
}
"""

theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="amber",
    neutral_hue="stone",
)


with gr.Blocks(
    title="Demand Forecasting Agent V2",
    theme=theme,
) as demo:

    gr.HTML(
        f"""
        <div class="header-box">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:18px;">
                <div>
                    <div class="hero-title">🌾 Demand Forecasting Agent V2</div>
                    <div class="hero-subtitle">
                        Structured launch forecast using product need area, benefit keywords,
                        flavour, flavour group, product form, buyer penetration, new-customer scale,
                        seasonality, strategy, price and campaign effects.
                    </div>
                </div>
                <div class="small-muted" style="text-align:right; min-width:230px;">
                    Artifact version: {METADATA.get("version", "N/A")}<br>
                    Created: {METADATA.get("created_at", "N/A")}<br>
                    Historical launches: {METADATA.get("launch_count", "N/A")}<br>
                    Ratio rows: {METADATA.get("launch_ratio_rows", "N/A")}
                </div>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='input-card'>")
            gr.HTML("<div class='section-title'>Historical Launch Test</div>")

            inp_historical_launch = gr.Dropdown(
                label="Historical launch",
                choices=HISTORICAL_LAUNCH_CHOICES,
                value=None,
            )

            btn_test_historical = gr.Button("🧪 Run Historical Test", variant="secondary")

            gr.HTML("<div class='section-title'>New Launch Input</div>")

            inp_product_name = gr.Textbox(
                label="Product name",
                value="",
                placeholder="Example: Daily Carnitin",
            )

            inp_product_need_area = gr.Dropdown(
                label="Product need area",
                choices=PRODUCT_NEED_AREA_CHOICES,
                value=None,
            )

            inp_benefit_keywords = gr.Dropdown(
                label="Benefit keywords",
                choices=BENEFIT_KEYWORD_CHOICES,
                value=[],
                multiselect=True,
            )

            inp_month_year = gr.Dropdown(
                label="Launch month",
                choices=MONTH_YEAR_CHOICES,
                value=None,
            )

            inp_product_form = gr.Dropdown(
                label="Product form",
                choices=PRODUCT_FORM_CHOICES,
                value=None,
            )

            inp_strategy = gr.Dropdown(
                label="Launch strategy type",
                choices=LAUNCH_STRATEGY_CHOICES,
                value=None,
            )

            inp_uvp = gr.Number(
                label="UVP in EUR",
                value=None,
            )

            inp_flavour = gr.Dropdown(
                label="Flavour",
                choices=FLAVOUR_CHOICES,
                value=None,
            )

            inp_flavour_group = gr.Dropdown(
                label="Flavour group",
                choices=FLAVOUR_GROUP_CHOICES,
                value=None,
            )

            btn = gr.Button("✨ Generate Forecast", variant="primary", size="lg")

            gr.HTML("<div class='section-title'>Summary</div>")

            out_summary = gr.Textbox(
                label="",
                lines=14,
                interactive=False,
            )

            out_run_record_status = gr.Textbox(
                label="Run record",
                lines=3,
                interactive=False,
            )

            gr.HTML("</div>")

        with gr.Column(scale=2):
            gr.HTML("<div class='output-card'>")
            gr.HTML("<div class='section-title'>Forecast Output</div>")

            out_forecast_table = gr.DataFrame(
                label="Forecast table",
                interactive=False,
                wrap=True,
            )

            out_chart = gr.Plot(label="Forecast chart")

        

            gr.HTML("<div class='section-title'>Behavioral Segment Impact</div>")

            out_behavioral_table = gr.DataFrame(
                label="Behavioral segmentation contribution",
                interactive=False,
                wrap=True,
            )

            gr.HTML("<div class='section-title'>Forecast Explanation</div>")

            out_explanation = gr.Textbox(
                label="",
                lines=18,
                interactive=False,
            )

            gr.HTML("</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")
            gr.HTML("<div class='section-title'>Similar Historical Launches</div>")

            out_similar_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True,
            )

            gr.HTML("</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")
            gr.HTML("<div class='section-title'>Historical Launch Comparison</div>")

            out_historical_comparison = gr.DataFrame(
                label="Historical vs forecast",
                interactive=False,
                wrap=True,
            )

            out_historical_summary = gr.Textbox(
                label="",
                lines=8,
                interactive=False,
            )

            gr.HTML("</div>")

        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")
            gr.HTML("<div class='section-title'>Applied Ratio Factors</div>")

            out_factor_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True,
            )

            gr.HTML("</div>")

    btn.click(
        fn=run_forecast_ui,
        inputs=[
            inp_product_name,
            inp_product_need_area,
            inp_benefit_keywords,
            inp_month_year,
            inp_product_form,
            inp_strategy,
            inp_uvp,
            inp_flavour,
            inp_flavour_group,
        ],
        outputs=[
            out_forecast_table,
            out_behavioral_table,
            out_chart,
            out_similar_table,
            out_factor_table,
            out_explanation,
            out_summary,
            out_run_record_status,
        ],
    )

    btn_test_historical.click(
        fn=run_historical_launch_test,
        inputs=[inp_historical_launch],
        outputs=[
            out_forecast_table,
            out_behavioral_table,
            out_chart,
            out_similar_table,
            out_factor_table,
            out_explanation,
            out_summary,
            out_run_record_status,
            out_historical_comparison,
            out_historical_summary,
        ],
    )


if __name__ == "__main__":
    demo.launch(debug=True, css=CSS)