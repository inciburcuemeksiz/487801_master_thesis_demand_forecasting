import os
import re
import pickle
import unicodedata
import uuid
from datetime import datetime
from difflib import SequenceMatcher

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

METRIC_LABELS = {
    "first_week_quantity": "First week quantity (units)",
    "first_6_week_quantity": "First 6 week quantity (units)",
    "first_week_nc": "First week new customers",
    "first_6_week_nc": "First 6 week new customers",
    "first_week_total_c": "First week total customers",
    "first_6_week_total_c": "First 6 week total customers",
}

PRODUCT_FORM_CHOICES = ["Capsules", "Drinking powder", "Oils", "Sprays", "Gummies"]
PRODUCT_FORM_MODEL_VALUE_MAP = {
    "Capsules": "Capsules",
    "Drinking powder": "Drinking Powder",
    "Oils": "Oil",
    "Sprays": "Spray",
    "Gummies": "Gummies",
}

LAUNCH_STRATEGY_CHOICES = ["standard", "co_creation", "limited_edition"]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"[^a-z0-9äöüß\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def normalize_strategy(x):
    if pd.isna(x) or not str(x).strip():
        return "standard"
    x = normalize_text(x).replace("-", "_").replace(" ", "_")
    mapping = {
        "standard": "standard",
        "standart": "standard",
        "co_creation": "co_creation",
        "cocreation": "co_creation",
        "co": "co_creation",
        "limited_edition": "limited_edition",
        "limited": "limited_edition",
    }
    return mapping.get(x, x or "standard")


def normalize_keyword_list(x):
    if x is None:
        return []
    raw_items = x if isinstance(x, list) else str(x).split(",")
    return sorted({normalize_text(item).replace(" ", "_") for item in raw_items if str(item).strip()})


def split_keywords(x):
    if pd.isna(x):
        return []
    return [k.strip() for k in str(x).split(",") if k.strip()]


def canonical_product_form(product_form):
    return PRODUCT_FORM_MODEL_VALUE_MAP.get(product_form, product_form or "")


def safe_divide(a, b, default=1.0):
    if b is None or pd.isna(b) or b == 0:
        return default
    if a is None or pd.isna(a):
        return default
    return a / b


def clip_factor(x, low=0.90, high=1.10):
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


def keyword_overlap_score(a, b):
    a_set = set(normalize_keyword_list(a))
    b_set = set(normalize_keyword_list(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def month_circular_similarity(m1, m2):
    if pd.isna(m1) or pd.isna(m2):
        return 0.0
    distance = abs(int(m1) - int(m2))
    distance = min(distance, 12 - distance)
    return 1 - distance / 6


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


def get_factor_from_dict(factor_dict, key, metric, default=1.0):
    key = normalize_text(key)
    if key and key in factor_dict and metric in factor_dict[key]:
        return factor_dict[key][metric]
    return default


def build_future_month_year_choices(n_months=12):
    today = pd.Timestamp.today().normalize()
    start = (today + pd.offsets.MonthBegin(1)).replace(day=1)
    return [(start + pd.DateOffset(months=i)).strftime("%m-%Y") for i in range(n_months)]


def parse_launch_month_year(launch_month_year):
    dt = pd.to_datetime(f"01-{launch_month_year}", format="%d-%m-%Y", errors="coerce")
    next_month = (pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1)).replace(day=1)
    if pd.isna(dt):
        return next_month
    dt = dt.replace(day=1)
    return max(dt, next_month)


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
        raise FileNotFoundError(f"{ARTIFACT_PATH} not found. Run: python build_artifacts_v2.py")
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


ARTIFACTS = load_artifacts()

ORDERS = ARTIFACTS["data"].get("orders", pd.DataFrame()).copy()
LAUNCHES = ARTIFACTS["data"]["launches"].copy()
SALE_TIMES = ARTIFACTS["data"]["sale_times"].copy()
LAUNCH_RATIO_TABLE = ARTIFACTS["data"].get("launch_ratio_table", pd.DataFrame()).copy()
MONTHLY_NEW_CUSTOMERS = ARTIFACTS["data"].get("monthly_new_customers", pd.DataFrame()).copy()

CALIBRATION = ARTIFACTS["calibration"]
METADATA = ARTIFACTS["metadata"]

SEASONALITY_INDEX = CALIBRATION.get("seasonality_index", {})
STRATEGY_FACTORS = CALIBRATION.get("strategy_factors", {})
FLAVOUR_FACTORS = CALIBRATION.get("flavour_factors", {})
PRODUCT_FORM_FACTORS = CALIBRATION.get("product_form_factors", {})
SALE_FACTORS = CALIBRATION.get("sale_factors", {})
PRICE_ELASTICITY = CALIBRATION.get("price_elasticity", -0.5)
RATIO_CONTEXT = CALIBRATION.get("ratio_context", {})

BEHAVIORAL_SEGMENTATION = ARTIFACTS.get("behavioral_segmentation", {})
SUPERVISED_ML = ARTIFACTS.get("supervised_ml", {})

LAUNCHES["sku"] = LAUNCHES["sku"].astype(str)
if not LAUNCH_RATIO_TABLE.empty and "sku" in LAUNCH_RATIO_TABLE.columns:
    LAUNCH_RATIO_TABLE["sku"] = LAUNCH_RATIO_TABLE["sku"].astype(str)
if not ORDERS.empty and "date" in ORDERS.columns:
    ORDERS["date"] = pd.to_datetime(ORDERS["date"], errors="coerce")
if not MONTHLY_NEW_CUSTOMERS.empty and "date" in MONTHLY_NEW_CUSTOMERS.columns:
    MONTHLY_NEW_CUSTOMERS["date"] = pd.to_datetime(MONTHLY_NEW_CUSTOMERS["date"], errors="coerce")


# ============================================================
# UI CHOICES
# ============================================================

def build_unique_choices(df, col):
    if col not in df.columns:
        return []
    return sorted([str(x).strip() for x in df[col].dropna().unique().tolist() if str(x).strip()])


def build_keyword_choices(df, col="benefit_keywords"):
    if col not in df.columns:
        return []
    keywords = set()
    for value in df[col].dropna().tolist():
        keywords.update(split_keywords(value))
    return sorted(keywords)


def format_historical_launch_choice(row):
    launch_date = pd.to_datetime(row.get("launch_date"), errors="coerce")
    launch_date_text = launch_date.date().isoformat() if pd.notna(launch_date) else "unknown-date"
    return f"{row.get('sku', '')} | {row.get('product', '')} | {row.get('flavour', '')} | {launch_date_text}"

def historical_launch_choice_to_sku(choice):
    if pd.isna(choice) or not str(choice).strip():
        return ""
    return str(choice).split("|", 1)[0].strip()


MONTH_YEAR_CHOICES = build_future_month_year_choices(12)
PRODUCT_NEED_AREA_CHOICES = build_unique_choices(LAUNCHES, "product_need_area")
BENEFIT_KEYWORD_CHOICES = build_keyword_choices(LAUNCHES, "benefit_keywords")
FLAVOUR_GROUP_CHOICES = build_unique_choices(LAUNCHES, "flavour_group")
FLAVOUR_CHOICES = build_unique_choices(LAUNCHES, "flavour")
if "New Flavour" not in FLAVOUR_CHOICES:
    FLAVOUR_CHOICES.append("New Flavour")

HISTORICAL_LAUNCH_CHOICES = []
if not LAUNCHES.empty and {"sku", "product", "launch_date"}.issubset(LAUNCHES.columns):
    historical_launches_sorted = LAUNCHES.sort_values(["launch_date", "sku"], ascending=[False, True])
    HISTORICAL_LAUNCH_CHOICES = [format_historical_launch_choice(row) for _, row in historical_launches_sorted.iterrows()]


# ============================================================
# CUTOFF-AWARE BACKTEST HELPERS
# ============================================================

def build_cutoff_ratio_context(cutoff_date):
    cutoff_date = pd.to_datetime(cutoff_date, errors="coerce")

    if pd.isna(cutoff_date) or ORDERS.empty:
        return RATIO_CONTEXT.copy()

    orders_before = ORDERS[ORDERS["date"] < cutoff_date].copy()

    if orders_before.empty:
        return {
            "latest_order_date": None,
            "known_customer_count": 1,
            "active_customer_count_12m": 1,
            "active_customer_count_6m": 1,
            "recent_3m_avg_new_customers": 1.0,
            "historical_avg_new_customers": 1.0,
            "new_customer_growth_factor": 1.0,
            "is_cutoff_context": True,
            "cutoff_date": str(cutoff_date.date()),
        }

    active_12m_start = cutoff_date - pd.DateOffset(months=12)
    active_6m_start = cutoff_date - pd.DateOffset(months=6)

    known_customer_count = int(orders_before["customer_nr"].nunique())
    active_customer_count_12m = int(orders_before.loc[orders_before["date"] >= active_12m_start, "customer_nr"].nunique())
    active_customer_count_6m = int(orders_before.loc[orders_before["date"] >= active_6m_start, "customer_nr"].nunique())

    monthly_before = MONTHLY_NEW_CUSTOMERS.copy()
    if not monthly_before.empty and "date" in monthly_before.columns:
        monthly_before = monthly_before[monthly_before["date"] < cutoff_date]

    if monthly_before.empty or "new_customers" not in monthly_before.columns:
        recent_3m_avg_new_customers = 1.0
        historical_avg_new_customers = 1.0
    else:
        recent_3m_avg_new_customers = float(monthly_before.tail(3)["new_customers"].mean())
        historical_avg_new_customers = float(monthly_before["new_customers"].mean())

    if not np.isfinite(recent_3m_avg_new_customers) or recent_3m_avg_new_customers <= 0:
        recent_3m_avg_new_customers = 1.0
    if not np.isfinite(historical_avg_new_customers) or historical_avg_new_customers <= 0:
        historical_avg_new_customers = 1.0

    return {
        "latest_order_date": str(orders_before["date"].max().date()),
        "known_customer_count": max(known_customer_count, 1),
        "active_customer_count_12m": max(active_customer_count_12m, 1),
        "active_customer_count_6m": max(active_customer_count_6m, 1),
        "recent_3m_avg_new_customers": recent_3m_avg_new_customers,
        "historical_avg_new_customers": historical_avg_new_customers,
        "new_customer_growth_factor": clip_factor(safe_divide(recent_3m_avg_new_customers, historical_avg_new_customers, 1.0), 0.5, 2.5),
        "is_cutoff_context": True,
        "cutoff_date": str(cutoff_date.date()),
    }


def build_cutoff_launch_ratio_table(cutoff_date):
    cutoff_date = pd.to_datetime(cutoff_date, errors="coerce")

    if pd.isna(cutoff_date) or LAUNCH_RATIO_TABLE.empty or "launch_date" not in LAUNCH_RATIO_TABLE.columns:
        return LAUNCH_RATIO_TABLE.copy()

    df = LAUNCH_RATIO_TABLE.copy()
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    return df[df["launch_date"] < cutoff_date].copy()


# ============================================================
# INPUT CONTEXT
# ============================================================

def build_input_context(
    product_name,
    product_need_area,
    benefit_keywords,
    launch_month_year,
    product_form_ui,
    launch_strategy_type,
    uvp,
    flavour,
    flavour_group,
    launch_date_override=None,
    backtest_cutoff_date=None,
):
    product_name = product_name or "New Product"
    product_need_area = product_need_area or ""
    benefit_keywords = benefit_keywords or []
    launch_month_year = launch_month_year or (MONTH_YEAR_CHOICES[0] if MONTH_YEAR_CHOICES else None)
    product_form_ui = product_form_ui or ""
    strategy = normalize_strategy(launch_strategy_type or "standard")
    flavour = flavour or "New Flavour"
    flavour_group = flavour_group or ""

    launch_date = pd.to_datetime(launch_date_override, errors="coerce") if launch_date_override is not None else pd.Timestamp("NaT")
    if pd.isna(launch_date):
        launch_date = parse_launch_month_year(launch_month_year)

    uvp = to_float(uvp, default=np.nan)
    if pd.isna(uvp) or uvp <= 0:
        uvp = LAUNCHES["uvp"].dropna().mean()

    product_form = canonical_product_form(product_form_ui)
    cutoff_date = pd.to_datetime(backtest_cutoff_date, errors="coerce") if backtest_cutoff_date is not None else pd.Timestamp("NaT")

    return {
        "product_name": product_name,
        "product_need_area": product_need_area,
        "benefit_keywords": benefit_keywords,
        "launch_month_year": launch_month_year,
        "launch_date": launch_date,
        "launch_month": int(launch_date.month),
        "product_form_ui": product_form_ui,
        "product_form": product_form,
        "strategy": strategy,
        "uvp": uvp,
        "flavour": flavour,
        "flavour_group": flavour_group,
        "backtest_cutoff_date": cutoff_date,
        "is_backtest": pd.notna(cutoff_date),
        "ratio_context": build_cutoff_ratio_context(cutoff_date) if pd.notna(cutoff_date) else RATIO_CONTEXT,
        "launch_ratio_table": build_cutoff_launch_ratio_table(cutoff_date) if pd.notna(cutoff_date) else LAUNCH_RATIO_TABLE,
    }


# ============================================================
# SIMILARITY ENGINE
# ============================================================

def score_launch_similarity(row, ctx):
    product_score = max(
        token_similarity(ctx["product_name"], row.get("product_norm", "")),
        token_similarity(ctx["product_name"], row.get("artikel_name_norm", "")),
        string_similarity(ctx["product_name"], row.get("product_norm", "")),
    )

    need_area_score = float(normalize_text(ctx["product_need_area"]) == normalize_text(row.get("product_need_area_norm", row.get("product_need_area", ""))))
    benefit_score = keyword_overlap_score(ctx["benefit_keywords"], row.get("benefit_keywords_norm", row.get("benefit_keywords", "")))

    flavour_score = max(token_similarity(ctx["flavour"], row.get("flavour_norm", "")), string_similarity(ctx["flavour"], row.get("flavour_norm", "")))
    if normalize_text(ctx["flavour"]) == "new flavour":
        flavour_score = 0.0

    flavour_group_score = float(normalize_text(ctx["flavour_group"]) == normalize_text(row.get("flavour_group_norm", row.get("flavour_group", ""))))
    product_form_score = max(token_similarity(ctx["product_form"], row.get("product_form_norm", "")), string_similarity(ctx["product_form"], row.get("product_form_norm", "")))
    strategy_score = strategy_similarity(ctx["strategy"], row.get("launch_strategy_type", "standard"))
    price_score = price_similarity(ctx["uvp"], row.get("uvp", np.nan))
    month_score = month_circular_similarity(ctx["launch_month"], row.get("launch_month", np.nan))

    similarity_score = (
    0.182740 * need_area_score
    + 0.123736 * benefit_score
    + 0.004193 * product_form_score
    + 0.035550 * flavour_group_score
    + 0.183200 * flavour_score
    + 0.148905 * strategy_score
    + 0.046785 * product_score
    + 0.053567 * price_score
    + 0.221323 * month_score
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


def find_similar_launches(ctx, top_n=3, exclude_sku=None, reference_launch_date=None):
    scored_rows = []
    reference_launch_date = pd.to_datetime(reference_launch_date, errors="coerce") if reference_launch_date is not None else pd.Timestamp("NaT")

    for _, row in LAUNCHES.iterrows():
        row_sku = str(row.get("sku", ""))
        if exclude_sku is not None and row_sku == str(exclude_sku):
            continue

        if pd.notna(reference_launch_date):
            row_launch_date = pd.to_datetime(row.get("launch_date"), errors="coerce")
            if pd.notna(row_launch_date) and row_launch_date >= reference_launch_date:
                continue

        row_dict = row.to_dict()
        row_dict.update(score_launch_similarity(row, ctx))
        scored_rows.append(row_dict)

    scored_df = pd.DataFrame(scored_rows)
    if scored_df.empty:
        return scored_df

    scored_df["sku"] = scored_df["sku"].astype(str)
    ratio_table = ctx.get("launch_ratio_table", LAUNCH_RATIO_TABLE)

    if ratio_table is not None and not ratio_table.empty:
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
        ]
        existing_ratio_cols = [c for c in ratio_cols if c in ratio_table.columns]
        scored_df = scored_df.merge(ratio_table[existing_ratio_cols], on="sku", how="left")

    return scored_df.sort_values("similarity_score", ascending=False).head(top_n).copy()


# ============================================================
# SALE + FACTORS
# ============================================================

def calculate_sale_overlap_for_new_launch(launch_date):
    if pd.isna(launch_date):
        return {"launch_during_sale": 0, "first_week_sale_days": 0, "first_6_week_sale_days": 0, "sale_name_overlap": ""}

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
    return clip_factor((uvp / avg_uvp) ** PRICE_ELASTICITY, 0.75, 1.25)


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


def calculate_demand_factor(metric, ctx, behavioral_segment_multiplier=1.0):
    seasonality_factor = SEASONALITY_INDEX.get(int(ctx["launch_month"]), 1.0)
    strategy_factor = STRATEGY_FACTORS.get(ctx["strategy"], {}).get(metric, 1.0)
    flavour_factor = get_factor_from_dict(FLAVOUR_FACTORS, ctx["flavour_group"], metric, 1.0)
    product_form_factor = get_factor_from_dict(PRODUCT_FORM_FACTORS, ctx["product_form"], metric, 1.0)
    price_factor = calculate_price_factor(ctx["uvp"])
    sale_factor, sale_overlap = calculate_sale_factor(metric, ctx["launch_date"])

    total_factor = seasonality_factor * strategy_factor * flavour_factor * product_form_factor * price_factor * sale_factor * behavioral_segment_multiplier
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


# ============================================================
# BEHAVIORAL SEGMENT IMPACT
# ============================================================

def calculate_behavioral_segment_multiplier(similar_launches, strategy, use_behavioral_segmentation=True):
    if not use_behavioral_segmentation:
        return 1.0, {"enabled": False, "reason": "Disabled in cutoff-aware historical test to avoid customer-behavior leakage"}

    if not BEHAVIORAL_SEGMENTATION.get("enabled", False):
        return 1.0, {"enabled": False, "reason": BEHAVIORAL_SEGMENTATION.get("reason", "Behavioral segmentation not available")}

    profile = BEHAVIORAL_SEGMENTATION.get("launch_segment_profile")
    seg_summary = BEHAVIORAL_SEGMENTATION.get("segment_summary")
    if profile is None or seg_summary is None or profile.empty:
        return 1.0, {"enabled": False, "reason": "Segment profile table is empty"}

    ref = similar_launches[["sku", "similarity_score"]].copy()
    ref["sku"] = ref["sku"].astype(str)
    ref["similarity_score"] = pd.to_numeric(ref["similarity_score"], errors="coerce").fillna(0.0)
    ref = ref[ref["similarity_score"] > 0]

    profile = profile.copy()
    profile["sku"] = profile["sku"].astype(str)
    merged = ref.merge(profile, on="sku", how="left").dropna(subset=["segment_key"])
    if merged.empty:
        return 1.0, {"enabled": False, "reason": "No segment affinity from similar launches"}

    affinity = (
        merged.assign(weighted_share=merged["similarity_score"] * merged["launch_segment_share"])
        .groupby("segment_key", as_index=False)
        .agg(weighted_share=("weighted_share", "sum"))
        .sort_values("weighted_share", ascending=False)
    )
    affinity["affinity"] = affinity["weighted_share"] / max(affinity["weighted_share"].sum(), 1e-9)

    if strategy == "co_creation":
        propensity_col = "avg_co_creation_purchase_count_24m"
    elif strategy == "limited_edition":
        propensity_col = "avg_limited_edition_purchase_count_24m"
    else:
        propensity_col = "avg_launch_purchase_count_24m"

    if propensity_col not in seg_summary.columns:
        return 1.0, {"enabled": False, "reason": f"Segment summary missing {propensity_col}"}

    global_propensity = float(seg_summary[propensity_col].mean())
    if not np.isfinite(global_propensity) or global_propensity <= 0:
        global_propensity = 1.0

    affinity = affinity.merge(seg_summary[["segment_key", propensity_col]], on="segment_key", how="left")
    affinity[propensity_col] = pd.to_numeric(affinity[propensity_col], errors="coerce").fillna(global_propensity)
    affinity["propensity_ratio"] = affinity[propensity_col] / global_propensity

    multiplier = clip_factor(float((affinity["affinity"] * affinity["propensity_ratio"]).sum()), 0.90, 1.10)
    return multiplier, {
        "enabled": True,
        "strategy": strategy,
        "propensity_col": propensity_col,
        "global_propensity": global_propensity,
        "multiplier": multiplier,
        "top_segments": affinity[["segment_key", "affinity", "propensity_ratio"]].head(5).to_dict("records"),
    }


def build_behavioral_segment_table(similar_launches, metric_values, enabled=True):
    empty = pd.DataFrame([
        {
            "Segment": "N/A",
            "Segment label": "N/A",
            "Affinity score": np.nan,
            "Global customer share": np.nan,
            "First week units": np.nan,
            "First 6 week units": np.nan,
            "First week NC": np.nan,
            "First 6 week NC": np.nan,
            "Interpretation": "Disabled in cutoff-aware historical test to avoid customer-behavior leakage" if not enabled else BEHAVIORAL_SEGMENTATION.get("reason", "Behavioral segmentation not available"),
        }
    ])

    if not enabled or not BEHAVIORAL_SEGMENTATION.get("enabled", False):
        return empty

    profile = BEHAVIORAL_SEGMENTATION.get("launch_segment_profile")
    seg_summary = BEHAVIORAL_SEGMENTATION.get("segment_summary")
    if profile is None or seg_summary is None or profile.empty:
        return empty

    ref = similar_launches[["sku", "similarity_score"]].copy()
    ref["sku"] = ref["sku"].astype(str)
    ref["similarity_score"] = pd.to_numeric(ref["similarity_score"], errors="coerce").fillna(0.0)
    ref = ref[ref["similarity_score"] > 0]

    profile = profile.copy()
    profile["sku"] = profile["sku"].astype(str)
    merged = ref.merge(profile, on="sku", how="left").dropna(subset=["segment_key"])
    if merged.empty:
        return empty

    agg = (
        merged.assign(weighted_share=merged["similarity_score"] * merged["launch_segment_share"])
        .groupby("segment_key", as_index=False)
        .agg(weighted_share=("weighted_share", "sum"))
        .sort_values("weighted_share", ascending=False)
    )
    agg["affinity"] = agg["weighted_share"] / max(agg["weighted_share"].sum(), 1e-9)

    agg["First week units"] = (agg["affinity"] * metric_values.get("first_week_quantity", 0)).round().astype(int)
    agg["First 6 week units"] = (agg["affinity"] * metric_values.get("first_6_week_quantity", 0)).round().astype(int)
    agg["First week NC"] = (agg["affinity"] * metric_values.get("first_week_nc", 0)).round().astype(int)
    agg["First 6 week NC"] = (agg["affinity"] * metric_values.get("first_6_week_nc", 0)).round().astype(int)

    summary_cols = ["segment_key", "segment_label", "segment_description", "global_share"]
    summary_cols = [c for c in summary_cols if c in seg_summary.columns]
    agg = agg.merge(seg_summary[summary_cols], on="segment_key", how="left")

    agg["Segment"] = agg["segment_key"]
    agg["Segment label"] = agg.get("segment_label", "")
    agg["Affinity score"] = agg["affinity"].round(3)
    agg["Global customer share"] = agg.get("global_share", 0).fillna(0).round(3)
    agg["Interpretation"] = np.where(
        agg["affinity"] >= agg.get("global_share", 0).fillna(0),
        "Over-indexed vs global mix",
        "Under-indexed vs global mix",
    )

    return agg[
        [
            "Segment",
            "Segment label",
            "Affinity score",
            "Global customer share",
            "First week units",
            "First 6 week units",
            "First week NC",
            "First 6 week NC",
            "Interpretation",
        ]
    ]


# ============================================================
# RATIO FORECAST
# ============================================================

def weighted_ratio(similar_launches, clipped_col, raw_col, fallback):
    value = np.nan
    if clipped_col in similar_launches.columns:
        value = weighted_average(similar_launches[clipped_col], similar_launches["similarity_score"])
    if pd.isna(value) and raw_col in similar_launches.columns:
        value = weighted_average(similar_launches[raw_col], similar_launches["similarity_score"])
    return float(fallback if pd.isna(value) else value)


def fallback_median(ctx, clipped_col, raw_col, default):
    ratio_table = ctx.get("launch_ratio_table", LAUNCH_RATIO_TABLE)
    if ratio_table is None or ratio_table.empty:
        return default
    col = clipped_col if clipped_col in ratio_table.columns else raw_col
    if col not in ratio_table.columns:
        return default
    value = ratio_table[col].median()
    return float(default if pd.isna(value) else value)


def calculate_ratio_based_forecast(ctx, similar_launches):
    use_behavioral = not ctx.get("is_backtest", False)
    behavioral_multiplier, behavioral_details = calculate_behavioral_segment_multiplier(
        similar_launches,
        ctx["strategy"],
        use_behavioral_segmentation=use_behavioral,
    )

    ratio_context = ctx.get("ratio_context", RATIO_CONTEXT)
    active_customer_base = float(ratio_context.get("active_customer_count_12m", ratio_context.get("known_customer_count", 1)))
    active_customer_base = max(active_customer_base, 1.0)
    recent_monthly_nc_base = float(ratio_context.get("recent_3m_avg_new_customers", 1.0))
    recent_monthly_nc_base = max(recent_monthly_nc_base, 1.0)

    buyer_ratio_1w = weighted_ratio(similar_launches, "buyer_ratio_1w_existing_clipped", "buyer_ratio_1w_existing", fallback_median(ctx, "buyer_ratio_1w_existing_clipped", "buyer_ratio_1w_existing", 0.003))
    buyer_ratio_6w = weighted_ratio(similar_launches, "buyer_ratio_6w_existing_clipped", "buyer_ratio_6w_existing", fallback_median(ctx, "buyer_ratio_6w_existing_clipped", "buyer_ratio_6w_existing", 0.007))
    nc_ratio_1w = weighted_ratio(similar_launches, "nc_ratio_1w_vs_monthly_nc_clipped", "nc_ratio_1w_vs_monthly_nc", fallback_median(ctx, "nc_ratio_1w_vs_monthly_nc_clipped", "nc_ratio_1w_vs_monthly_nc", 0.06))
    nc_ratio_6w = weighted_ratio(similar_launches, "nc_ratio_6w_vs_monthly_nc_clipped", "nc_ratio_6w_vs_monthly_nc", fallback_median(ctx, "nc_ratio_6w_vs_monthly_nc_clipped", "nc_ratio_6w_vs_monthly_nc", 0.11))
    upc_1w = weighted_ratio(similar_launches, "units_per_customer_1w_clipped", "units_per_customer_1w", fallback_median(ctx, "units_per_customer_1w_clipped", "units_per_customer_1w", 1.30))
    upc_6w = weighted_ratio(similar_launches, "units_per_customer_6w_clipped", "units_per_customer_6w", fallback_median(ctx, "units_per_customer_6w_clipped", "units_per_customer_6w", 1.30))

    factor_existing_1w = calculate_demand_factor("first_week_total_c", ctx, behavioral_multiplier)
    factor_existing_6w = calculate_demand_factor("first_6_week_total_c", ctx, behavioral_multiplier)
    factor_nc_1w = calculate_demand_factor("first_week_nc", ctx, behavioral_multiplier)
    factor_nc_6w = calculate_demand_factor("first_6_week_nc", ctx, behavioral_multiplier)

    existing_1w_base = active_customer_base * buyer_ratio_1w
    existing_6w_base = active_customer_base * buyer_ratio_6w
    nc_1w_base = recent_monthly_nc_base * nc_ratio_1w
    nc_6w_base = recent_monthly_nc_base * nc_ratio_6w

    existing_1w = existing_1w_base * factor_existing_1w["total_factor"]
    existing_6w = existing_6w_base * factor_existing_6w["total_factor"]
    nc_1w = nc_1w_base * factor_nc_1w["total_factor"]
    nc_6w = nc_6w_base * factor_nc_6w["total_factor"]

    total_c_1w = max(0, existing_1w + nc_1w)
    total_c_6w = max(0, existing_6w + nc_6w)
    nc_1w = min(max(0, nc_1w), total_c_1w)
    nc_6w = min(max(0, nc_6w), total_c_6w)

    forecasts = {
        "first_week_quantity": total_c_1w * upc_1w,
        "first_6_week_quantity": total_c_6w * upc_6w,
        "first_week_nc": nc_1w,
        "first_6_week_nc": nc_6w,
        "first_week_total_c": total_c_1w,
        "first_6_week_total_c": total_c_6w,
        "first_week_existing_c": max(0, existing_1w),
        "first_6_week_existing_c": max(0, existing_6w),
    }

    factor_details = {}
    details_map = {
        "first_week_quantity": (forecasts["first_week_quantity"], buyer_ratio_1w, nc_ratio_1w, upc_1w, factor_existing_1w, "total customers × weighted units/customer"),
        "first_6_week_quantity": (forecasts["first_6_week_quantity"], buyer_ratio_6w, nc_ratio_6w, upc_6w, factor_existing_6w, "total customers × weighted units/customer"),
        "first_week_nc": (nc_1w_base, buyer_ratio_1w, nc_ratio_1w, upc_1w, factor_nc_1w, "recent monthly NC base × weighted NC ratio"),
        "first_6_week_nc": (nc_6w_base, buyer_ratio_6w, nc_ratio_6w, upc_6w, factor_nc_6w, "recent monthly NC base × weighted NC ratio"),
        "first_week_total_c": (existing_1w_base + nc_1w_base, buyer_ratio_1w, nc_ratio_1w, upc_1w, factor_existing_1w, "existing customers + new customers"),
        "first_6_week_total_c": (existing_6w_base + nc_6w_base, buyer_ratio_6w, nc_ratio_6w, upc_6w, factor_existing_6w, "existing customers + new customers"),
    }

    for metric, (base, buyer_ratio, nc_ratio, upc, factors, logic) in details_map.items():
        factor_details[metric] = {
            "forecast_logic": logic,
            "base": base,
            "active_customer_base": active_customer_base,
            "recent_monthly_nc_base": recent_monthly_nc_base,
            "buyer_ratio": buyer_ratio,
            "nc_ratio": nc_ratio,
            "units_per_customer": upc,
            "behavioral_segment_details": behavioral_details,
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
        ratio_bonus = min(0.08, int(similar_launches["buyer_ratio_6w_existing_clipped"].notna().sum()) * 0.01)
    return float(np.clip(0.35 + 0.35 * top_score + 0.15 * avg_top_3 + ref_bonus + ratio_bonus, 0.30, 0.92))


def scenario_bounds(base_forecast, confidence):
    uncertainty = 0.35 - (confidence - 0.30) * 0.25
    uncertainty = float(np.clip(uncertainty, 0.12, 0.35))
    return int(round(base_forecast * (1 - uncertainty))), int(round(base_forecast)), int(round(base_forecast * (1 + uncertainty)))


# ============================================================
# SUPERVISED ML FORECAST
# ============================================================

def build_ml_input_row(ctx, similar_launches):
    sale_overlap = calculate_sale_overlap_for_new_launch(ctx["launch_date"])
    row = {
        "product_need_area_norm": normalize_text(ctx["product_need_area"]),
        "flavour_group_norm": normalize_text(ctx["flavour_group"]),
        "product_form_norm": normalize_text(ctx["product_form"]),
        "launch_strategy_type": ctx["strategy"],
        "uvp": ctx["uvp"],
        "first_order_quantity": np.nan,
        "launch_month": ctx["launch_month"],
        "launch_during_sale": sale_overlap["launch_during_sale"],
        "first_week_sale_days": sale_overlap["first_week_sale_days"],
        "first_6_week_sale_days": sale_overlap["first_6_week_sale_days"],
        "launch_text": (
            normalize_text(ctx["product_name"])
            + " "
            + normalize_text(ctx["product_need_area"])
            + " "
            + " ".join(normalize_keyword_list(ctx["benefit_keywords"]))
            + " "
            + normalize_text(ctx["flavour"])
            + " "
            + normalize_text(ctx["flavour_group"])
            + " "
            + normalize_text(ctx["product_form"])
            + " "
            + ctx["strategy"]
        ).strip(),
    }

    ratio_feature_pairs = {
        "eligible_customers_before_launch": "eligible_customers_before_launch",
        "buyer_ratio_1w_existing_clipped": "buyer_ratio_1w_existing_clipped",
        "buyer_ratio_6w_existing_clipped": "buyer_ratio_6w_existing_clipped",
        "nc_ratio_1w_vs_monthly_nc_clipped": "nc_ratio_1w_vs_monthly_nc_clipped",
        "nc_ratio_6w_vs_monthly_nc_clipped": "nc_ratio_6w_vs_monthly_nc_clipped",
        "units_per_customer_1w_clipped": "units_per_customer_1w_clipped",
        "units_per_customer_6w_clipped": "units_per_customer_6w_clipped",
    }

    for out_col, source_col in ratio_feature_pairs.items():
        row[out_col] = weighted_average(similar_launches[source_col], similar_launches["similarity_score"]) if source_col in similar_launches.columns else np.nan

    if BEHAVIORAL_SEGMENTATION.get("enabled", False) and not ctx.get("is_backtest", False):
        profile = BEHAVIORAL_SEGMENTATION.get("launch_segment_profile")
        if profile is not None and not profile.empty:
            ref = similar_launches[["sku", "similarity_score"]].copy()
            ref["sku"] = ref["sku"].astype(str)
            profile = profile.copy()
            profile["sku"] = profile["sku"].astype(str)
            merged = ref.merge(profile, on="sku", how="left").dropna(subset=["segment_key"])
            if not merged.empty:
                merged["weighted_share"] = merged["similarity_score"] * merged["launch_segment_share"]
                seg = merged.groupby("segment_key")["weighted_share"].sum()
                total = max(seg.sum(), 1e-9)
                for seg_key, value in seg.items():
                    row[f"segment_share_{seg_key}"] = float(value / total)

    return pd.DataFrame([row])


def transform_ml_features(input_df, preprocessors):
    numeric_cols = preprocessors.get("numeric_columns", [])
    categorical_cols = preprocessors.get("categorical_columns", [])
    numeric_defaults = preprocessors.get("numeric_defaults", {})
    categorical_defaults = preprocessors.get("categorical_defaults", {})
    scaler = preprocessors["scaler"]
    encoder = preprocessors["encoder"]
    text_vectorizer = preprocessors["text_vectorizer"]

    df = input_df.copy()

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = numeric_defaults.get(col, 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(numeric_defaults.get(col, 0.0))

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = categorical_defaults.get(col, "unknown")
        df[col] = df[col].fillna(categorical_defaults.get(col, "unknown")).astype(str)

    x_num = sparse.csr_matrix(scaler.transform(df[numeric_cols])) if numeric_cols else sparse.csr_matrix((len(df), 0))
    x_cat = encoder.transform(df[categorical_cols]) if categorical_cols else sparse.csr_matrix((len(df), 0))
    x_text = text_vectorizer.transform(df["launch_text"].fillna(""))

    return sparse.hstack([x_num, x_cat, x_text], format="csr")


def predict_supervised_ml(ctx, similar_launches):
    if not SUPERVISED_ML.get("enabled", False):
        return {
            "enabled": False,
            "reason": SUPERVISED_ML.get("reason", "Supervised ML not available"),
            "point": {},
            "quantiles": {},
        }

    best_model_name = SUPERVISED_ML.get("best_model_name")
    models = SUPERVISED_ML.get("models", {})
    model = models.get(best_model_name)
    preprocessors = SUPERVISED_ML.get("preprocessors")

    if model is None or preprocessors is None:
        return {"enabled": False, "reason": "Best ML model or preprocessors missing", "point": {}, "quantiles": {}}

    input_df = build_ml_input_row(ctx, similar_launches)
    x = transform_ml_features(input_df, preprocessors)

    pred = model.predict(x)[0]
    point = {metric: max(0.0, float(value)) for metric, value in zip(TARGET_METRICS, pred)}

    quantiles = {}
    quantile_models = SUPERVISED_ML.get("quantile_models", {})
    if quantile_models:
        x_dense = x.toarray() if sparse.issparse(x) else x
        for metric in TARGET_METRICS:
            metric_models = quantile_models.get(metric, {})
            if metric_models:
                q10 = max(0.0, float(metric_models["q10"].predict(x_dense)[0])) if "q10" in metric_models else np.nan
                q50 = max(0.0, float(metric_models["q50"].predict(x_dense)[0])) if "q50" in metric_models else np.nan
                q90 = max(0.0, float(metric_models["q90"].predict(x_dense)[0])) if "q90" in metric_models else np.nan
                quantiles[metric] = {"q10": q10, "q50": q50, "q90": q90}

    return {
        "enabled": True,
        "reason": "ok",
        "best_model_name": best_model_name,
        "point": point,
        "quantiles": quantiles,
    }


# ============================================================
# OUTPUT TABLES
# ============================================================

def build_forecast_table(ratio_forecast, ml_forecast, confidence):
    rows = []

    for metric in TARGET_METRICS:
        low, base, high = scenario_bounds(ratio_forecast[metric], confidence)
        row = {
            "Metric": METRIC_LABELS[metric],
            "Ratio worst": low,
            "Ratio base": base,
            "Ratio best": high,
        }

        if ml_forecast.get("enabled"):
            point = int(round(ml_forecast["point"].get(metric, np.nan)))
            q = ml_forecast.get("quantiles", {}).get(metric, {})
            row.update(
                {
                    "ML point": point,
                    "ML q10": "" if pd.isna(q.get("q10", np.nan)) else int(round(q.get("q10"))),
                    "ML q50": "" if pd.isna(q.get("q50", np.nan)) else int(round(q.get("q50"))),
                    "ML q90": "" if pd.isna(q.get("q90", np.nan)) else int(round(q.get("q90"))),
                }
            )
        else:
            row.update({"ML point": "", "ML q10": "", "ML q50": "", "ML q90": ""})

        rows.append(row)

    first_order_base = int(round(ratio_forecast["first_6_week_quantity"] * 1.10))
    rows.append(
        {
            "Metric": "Recommended first-order quantity",
            "Ratio worst": "",
            "Ratio base": first_order_base,
            "Ratio best": "Base 6-week quantity\n+ 10% safety buffer",
            "ML point": "",
            "ML q10": "",
            "ML q50": "",
            "ML q90": "",
        }
    )

    return pd.DataFrame(rows)


def build_ml_model_summary_table():
    if not SUPERVISED_ML.get("enabled", False):
        return pd.DataFrame([{"Status": "Disabled", "Reason": SUPERVISED_ML.get("reason", "Not available")}])

    summary = SUPERVISED_ML.get("model_summary", pd.DataFrame())
    if summary is None or summary.empty:
        return pd.DataFrame([{"Status": "Enabled", "Best model": SUPERVISED_ML.get("best_model_name", "N/A")}])

    out = summary.copy()
    rename_map = {
        "model": "Model",
        "avg_mae": "Average MAE",
        "avg_rmse": "Average RMSE",
        "avg_smape": "Average sMAPE",
        "avg_r2": "Average R²",
    }
    out = out.rename(columns=rename_map)
    for col in ["Average MAE", "Average RMSE", "Average sMAPE", "Average R²"]:
        if col in out.columns:
            out[col] = out[col].round(3)
    return out


def build_similar_table(similar_launches):
    cols = [
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
        "similarity_score",
    ]
    cols = [col for col in cols if col in similar_launches.columns]
    out = similar_launches[cols].copy()

    for col in [c for c in out.columns if c.endswith("_score")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    for col in ["buyer_ratio_6w_existing_clipped", "nc_ratio_6w_vs_monthly_nc_clipped", "units_per_customer_6w_clipped"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    if "launch_date" in out.columns:
        out["launch_date"] = pd.to_datetime(out["launch_date"], errors="coerce").dt.date.astype(str)

    return out


def build_factor_table(factor_details):
    rows = []
    for metric in TARGET_METRICS:
        details = factor_details[metric]
        rows.append(
            {
                "Metric": METRIC_LABELS[metric],
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
    return pd.DataFrame(rows)


def build_chart(ratio_forecast, ml_forecast, confidence):
    labels = ["1W qty", "6W qty", "1W total C", "6W total C", "1W NC", "6W NC"]
    metrics = [
        "first_week_quantity",
        "first_6_week_quantity",
        "first_week_total_c",
        "first_6_week_total_c",
        "first_week_nc",
        "first_6_week_nc",
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[ratio_forecast[m] for m in metrics], name="Ratio base"))

    if ml_forecast.get("enabled"):
        fig.add_trace(go.Bar(x=labels, y=[ml_forecast["point"].get(m, 0) for m in metrics], name=f"ML point ({ml_forecast.get('best_model_name')})"))

    fig.update_layout(
        title=dict(text=f"Launch Forecast | Confidence: {confidence:.2f}", font=dict(color="#4a392d", size=18)),
        xaxis_title="Metric",
        yaxis_title="Forecast",
        height=420,
        plot_bgcolor="#fffaf4",
        paper_bgcolor="#fffaf4",
        bargap=0.35,
        barmode="group",
        margin=dict(l=40, r=30, t=70, b=50),
        font=dict(color="#4a392d"),
    )
    return fig


def build_explanation(ctx, similar_launches, factor_details, confidence, ml_forecast):
    top = similar_launches.head(3)
    keyword_text = ", ".join(ctx["benefit_keywords"]) if isinstance(ctx["benefit_keywords"], list) else str(ctx["benefit_keywords"])
    fwq = factor_details["first_week_quantity"]
    sixq = factor_details["first_6_week_quantity"]
    sale_overlap = calculate_sale_overlap_for_new_launch(ctx["launch_date"])

    lines = [
        f"Forecast generated for: {ctx['product_name']}",
        f"Product need area: {ctx['product_need_area']}",
        f"Benefit keywords: {keyword_text}",
        f"Flavour: {ctx['flavour']}",
        f"Flavour group: {ctx['flavour_group']}",
        f"Product form: {ctx['product_form_ui']}",
        f"Launch strategy: {ctx['strategy']}",
    ]

    if ctx.get("is_backtest", False):
        cutoff_context = ctx.get("ratio_context", {})
        lines.extend(
            [
                "",
                "Historical backtest mode:",
                f"- Cutoff date: {ctx['backtest_cutoff_date'].date()}",
                "- Similar launches after the cutoff date are excluded.",
                "- Customer base and recent new-customer scale are recalculated using only orders before the cutoff date.",
                "- Behavioral segment multiplier is disabled to avoid future customer-behavior leakage.",
                f"- Cutoff active customer base: {cutoff_context.get('active_customer_count_12m', 'N/A')}",
                f"- Cutoff recent monthly NC base: {round(cutoff_context.get('recent_3m_avg_new_customers', 0), 1)}",
            ]
        )

    lines.extend(["", "Top similar historical launches:"])

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i}. {row.get('product', 'N/A')} | need area: {row.get('product_need_area', 'N/A')} | "
            f"flavour: {row.get('flavour', 'N/A')} | score: {row.get('similarity_score', np.nan):.2f}"
        )

    lines.extend(
        [
            "",
            "Forecast logic:",
            "1) Similar historical launches are selected using structured product attributes.",
            "2) Ratio forecast uses historical buyer penetration, new-customer ratios, and units/customer.",
            "3) Adjustment factors are applied for seasonality, strategy, flavour group, product form, price, sale overlap, and behavioral segments.",
            "4) Supervised ML uses the best benchmark model trained during artifact build.",
            "",
            "Main ratio drivers:",
            f"- Active customer base used: {fwq['active_customer_base']:.0f}",
            f"- Recent monthly NC base used: {fwq['recent_monthly_nc_base']:.0f}",
            f"- Weighted first-week buyer ratio: {fwq['buyer_ratio']:.4f}",
            f"- Weighted first-6-week buyer ratio: {sixq['buyer_ratio']:.4f}",
            f"- Weighted first-week NC ratio vs monthly NC: {fwq['nc_ratio']:.4f}",
            f"- Weighted first-6-week NC ratio vs monthly NC: {sixq['nc_ratio']:.4f}",
            f"- Weighted units/customer first week: {fwq['units_per_customer']:.2f}",
            f"- Weighted units/customer first 6 weeks: {sixq['units_per_customer']:.2f}",
            "",
            "Main adjustment factors:",
            f"- Seasonality factor: {fwq['seasonality_factor']:.2f}",
            f"- Launch strategy factor: {fwq['strategy_factor']:.2f}",
            f"- Flavour group factor: {fwq['flavour_group_factor']:.2f}",
            f"- Product form factor: {fwq['product_form_factor']:.2f}",
            f"- Price factor: {fwq['price_factor']:.2f}",
            f"- Sale factor: {fwq['sale_factor']:.2f}",
            f"- Behavioral segment factor: {fwq.get('behavioral_segment_factor', 1.0):.2f}",
        ]
    )

    if sale_overlap["sale_name_overlap"]:
        lines.extend(
            [
                "",
                f"Sale overlap detected: {sale_overlap['sale_name_overlap']}",
                f"- First week sale days: {sale_overlap['first_week_sale_days']}",
                f"- First 6 week sale days: {sale_overlap['first_6_week_sale_days']}",
            ]
        )

    lines.append("")
    if ml_forecast.get("enabled"):
        lines.append(f"ML benchmark model used: {ml_forecast.get('best_model_name')}")
        if ctx.get("is_backtest", False):
            lines.append("ML note: This ML model was trained on the full artifact dataset, so historical-test ML outputs are diagnostic only unless a cutoff-specific ML artifact is built.")
    else:
        lines.append(f"ML forecast unavailable: {ml_forecast.get('reason')}")

    lines.extend(["", f"Similarity confidence score: {confidence:.2f}"])
    if confidence < 0.50:
        lines.append("Confidence note: Low. Similar historical references are weak or limited.")
    elif confidence < 0.70:
        lines.append("Confidence note: Medium. Forecast is usable but should be reviewed with business context.")
    else:
        lines.append("Confidence note: Good. Similar historical references are relatively strong.")

    return "\n".join(lines)


def build_summary_text(ctx, ratio_forecast, ml_forecast, confidence):
    lines = [
        f"Ratio base first week quantity: {int(round(ratio_forecast['first_week_quantity']))}",
        f"Ratio base first 6 week quantity: {int(round(ratio_forecast['first_6_week_quantity']))}",
        f"Ratio base first week total customers: {int(round(ratio_forecast['first_week_total_c']))}",
        f"Ratio base first 6 week total customers: {int(round(ratio_forecast['first_6_week_total_c']))}",
        f"Ratio base first week new customers: {int(round(ratio_forecast['first_week_nc']))}",
        f"Ratio base first 6 week new customers: {int(round(ratio_forecast['first_6_week_nc']))}",
    ]

    if ctx.get("is_backtest", False):
        ratio_context = ctx.get("ratio_context", {})
        lines.extend(
            [
                "",
                "Backtest mode: cutoff-aware ratio forecast",
                f"Cutoff date: {ctx['backtest_cutoff_date'].date()}",
                f"Cutoff active customer base: {ratio_context.get('active_customer_count_12m', 'N/A')}",
                f"Cutoff recent monthly NC base: {round(ratio_context.get('recent_3m_avg_new_customers', 0), 1)}",
            ]
        )

    if ml_forecast.get("enabled"):
        lines.extend(
            [
                "",
                f"ML model: {ml_forecast.get('best_model_name')}",
                f"ML first week quantity: {int(round(ml_forecast['point']['first_week_quantity']))}",
                f"ML first 6 week quantity: {int(round(ml_forecast['point']['first_6_week_quantity']))}",
                f"ML first week total customers: {int(round(ml_forecast['point']['first_week_total_c']))}",
                f"ML first 6 week total customers: {int(round(ml_forecast['point']['first_6_week_total_c']))}",
            ]
        )

    lines.extend(
        [
            "",
            f"Recommended first-order quantity: {int(round(ratio_forecast['first_6_week_quantity'] * 1.10))}",
            "Rationale: Ratio base 6-week quantity + 10% safety buffer",
            f"Product need area: {ctx['product_need_area']}",
            f"Product form: {ctx['product_form_ui']}",
            f"Flavour: {ctx['flavour']}",
            f"Flavour group: {ctx['flavour_group']}",
            f"Confidence: {confidence:.2f}",
        ]
    )
    return "\n".join(lines)


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
    launch_date_override=None,
    historical_reference_sku=None,
    backtest_cutoff_date=None,
):
    ctx = build_input_context(
        product_name=product_name,
        product_need_area=product_need_area,
        benefit_keywords=benefit_keywords,
        launch_month_year=launch_month_year,
        product_form_ui=product_form_ui,
        launch_strategy_type=launch_strategy_type,
        uvp=uvp,
        flavour=flavour,
        flavour_group=flavour_group,
        launch_date_override=launch_date_override,
        backtest_cutoff_date=backtest_cutoff_date,
    )

    similar_launches = find_similar_launches(
        ctx,
        top_n=3,
        exclude_sku=historical_reference_sku,
        reference_launch_date=ctx["launch_date"] if historical_reference_sku else None,
    )

    ratio_forecast, factor_details = calculate_ratio_based_forecast(ctx, similar_launches)
    confidence = calculate_confidence(similar_launches)
    ml_forecast = predict_supervised_ml(ctx, similar_launches)

    forecast_table = build_forecast_table(ratio_forecast, ml_forecast, confidence)
    behavioral_segment_table = build_behavioral_segment_table(similar_launches, ratio_forecast, enabled=not ctx.get("is_backtest", False))
    ml_model_summary_table = build_ml_model_summary_table()
    similar_table = build_similar_table(similar_launches)
    factor_table = build_factor_table(factor_details)
    fig = build_chart(ratio_forecast, ml_forecast, confidence)
    explanation = build_explanation(ctx, similar_launches, factor_details, confidence, ml_forecast)
    summary_text = build_summary_text(ctx, ratio_forecast, ml_forecast, confidence)

    run_id = f"FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    reference_skus = "|".join(similar_table["sku"].astype(str).head(5).tolist()) if "sku" in similar_table.columns else ""
    ratio_context = ctx.get("ratio_context", RATIO_CONTEXT)

    append_forecast_run_log(
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_mode": "historical_backtest_cutoff" if ctx.get("is_backtest", False) else "new_launch_forecast",
            "backtest_cutoff_date": str(ctx["backtest_cutoff_date"].date()) if ctx.get("is_backtest", False) else "",
            "product_name": ctx["product_name"],
            "product_need_area": ctx["product_need_area"],
            "benefit_keywords": ", ".join(ctx["benefit_keywords"]) if isinstance(ctx["benefit_keywords"], list) else str(ctx["benefit_keywords"]),
            "flavour": ctx["flavour"],
            "flavour_group": ctx["flavour_group"],
            "product_form_input": ctx["product_form_ui"],
            "product_form_model": ctx["product_form"],
            "uvp": ctx["uvp"],
            "launch_strategy_type": ctx["strategy"],
            "launch_date": str(ctx["launch_date"].date()),
            "launch_month_year": ctx["launch_month_year"],
            "launch_month": ctx["launch_month"],
            "ratio_first_week_qty": int(round(ratio_forecast["first_week_quantity"])),
            "ratio_first_6w_qty": int(round(ratio_forecast["first_6_week_quantity"])),
            "ratio_first_week_total_c": int(round(ratio_forecast["first_week_total_c"])),
            "ratio_first_6w_total_c": int(round(ratio_forecast["first_6_week_total_c"])),
            "ratio_first_week_nc": int(round(ratio_forecast["first_week_nc"])),
            "ratio_first_6w_nc": int(round(ratio_forecast["first_6_week_nc"])),
            "ml_enabled": ml_forecast.get("enabled", False),
            "ml_model": ml_forecast.get("best_model_name", ""),
            "ml_first_week_qty": int(round(ml_forecast["point"].get("first_week_quantity", np.nan))) if ml_forecast.get("enabled") else "",
            "ml_first_6w_qty": int(round(ml_forecast["point"].get("first_6_week_quantity", np.nan))) if ml_forecast.get("enabled") else "",
            "confidence": float(round(confidence, 4)),
            "recommended_first_order_qty": int(round(ratio_forecast["first_6_week_quantity"] * 1.10)),
            "reference_skus": reference_skus,
            "active_customer_base": ratio_context.get("active_customer_count_12m", ""),
            "recent_monthly_nc_base": ratio_context.get("recent_3m_avg_new_customers", ""),
        }
    )

    run_status = f"Run ID: {run_id}\nSaved to: {FORECAST_RUN_LOG_PATH}"

    return (
        forecast_table,
        behavioral_segment_table,
        ml_model_summary_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_status,
    )


# ============================================================
# HISTORICAL TEST
# ============================================================

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
        actual = to_float(actual_launch.get(actual_col), np.nan)
        match = forecast_table[forecast_table["Metric"] == forecast_label]
        ratio_forecast = to_float(match.iloc[0]["Ratio base"], np.nan) if not match.empty else np.nan
        ml_forecast = to_float(match.iloc[0]["ML point"], np.nan) if not match.empty and "ML point" in match.columns else np.nan

        rows.append(
            {
                "Metric": forecast_label,
                "Actual": actual,
                "Ratio forecast": ratio_forecast,
                "Ratio error %": safe_divide(ratio_forecast - actual, actual, np.nan) * 100 if not pd.isna(ratio_forecast) and not pd.isna(actual) else np.nan,
                "ML forecast": ml_forecast,
                "ML error %": safe_divide(ml_forecast - actual, actual, np.nan) * 100 if not pd.isna(ml_forecast) and not pd.isna(actual) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    for col in ["Actual", "Ratio forecast", "Ratio error %", "ML forecast", "ML error %"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out


def run_historical_launch_test(historical_launch_choice):
    launch = get_historical_launch_row(historical_launch_choice)
    launch_date = pd.to_datetime(launch.get("launch_date"), errors="coerce")

    result = run_forecast(
        product_name=launch.get("product", ""),
        product_need_area=launch.get("product_need_area", ""),
        benefit_keywords=split_keywords(launch.get("benefit_keywords", "")),
        launch_month_year=launch_date.strftime("%m-%Y"),
        product_form_ui=launch.get("product_form", ""),
        launch_strategy_type=launch.get("launch_strategy_type", "standard"),
        uvp=launch.get("uvp", np.nan),
        flavour=launch.get("flavour", ""),
        flavour_group=launch.get("flavour_group", ""),
        launch_date_override=launch_date,
        historical_reference_sku=launch.get("sku", ""),
        backtest_cutoff_date=launch_date,
    )

    (
        forecast_table,
        behavioral_segment_table,
        ml_model_summary_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_status,
    ) = result

    comparison_table = build_historical_comparison_table(launch, forecast_table)

    ratio_mape = comparison_table["Ratio error %"].dropna().abs().mean()
    ml_mape = comparison_table["ML error %"].dropna().abs().mean()

    historical_summary = "\n".join(
        [
            f"Historical launch SKU: {launch.get('sku', '')}",
            f"Historical launch date: {launch_date.date()}",
            f"Actual first week quantity: {to_float(launch.get('first_week_quantity'), np.nan)}",
            f"Actual first 6 week quantity: {to_float(launch.get('first_6_week_quantity'), np.nan)}",
            f"Cutoff-aware Ratio MAPE across available metrics: {ratio_mape:.2f}%" if not pd.isna(ratio_mape) else "Ratio MAPE: N/A",
            f"ML MAPE across available metrics: {ml_mape:.2f}%" if not pd.isna(ml_mape) else "ML MAPE: N/A",
            "Note: Ratio forecast is cutoff-aware. ML forecast is diagnostic unless cutoff-specific ML artifacts are built.",
        ]
    )

    return (
        forecast_table,
        behavioral_segment_table,
        ml_model_summary_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        f"{summary_text}\n\n{historical_summary}",
        f"Historical launch test completed for {launch.get('sku', '')}.\n{run_status}",
        comparison_table,
        historical_summary,
    )


# ============================================================
# GRADIO UI
# ============================================================

CSS = """
body, .gradio-container {
    font-family: Inter, Arial, sans-serif !important;
    background: linear-gradient(135deg, #fbf7f1 0%, #f3eadf 45%, #eadccb 100%) !important;
    color: #3f342c !important;
}

/* Wider app layout */
.gradio-container {
    max-width: 99vw !important;
    width: 99vw !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
}

/* Make Gradio main area wider too */
main {
    max-width: 99vw !important;
}

/* Cards */
.header-box, .input-card, .output-card {
    background: rgba(255, 252, 247, 0.94);
    border: 1px solid #e2d1bd;
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 14px 35px rgba(121, 92, 63, 0.11);
    width: 100% !important;
}

.header-box {
    padding: 22px 26px;
    margin-bottom: 18px;
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

/* General table width */
.dataframe, table {
    width: 100% !important;
}

.wrap {
    width: 100% !important;
}

/* Applied Ratio Factors: full-width readable table */
.factor-table-wide table {
    width: 100% !important;
    table-layout: auto !important;
    font-size: 11px !important;
}

.factor-table-wide th,
.factor-table-wide td {
    padding: 5px 7px !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    line-height: 1.2 !important;
}

/* Optional: make wide factor table less cramped */
.factor-table-wide {
    width: 100% !important;
}

/* Hide Gradio footer */
footer {
    visibility: hidden;
}
"""

theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="amber",
    neutral_hue="stone"
)

with gr.Blocks(
    title="Demand Forecasting Agent V2",
    theme=theme,
    css=CSS
) as demo:

    gr.HTML(
        f"""
        <div class="header-box">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:18px;">
                <div>
                    <div class="hero-title">Demand Forecasting Agent V2</div>
                    <div class="hero-subtitle">
                        Hybrid launch forecasting using ratio-based comparable launches, behavioral segmentation,
                        supervised ML benchmarking, and quantile intervals.
                    </div>
                </div>
                <div class="small-muted" style="text-align:right; min-width:250px;">
                    Artifact version: {METADATA.get("version", "N/A")}<br>
                    Created: {METADATA.get("created_at", "N/A")}<br>
                    Historical launches: {METADATA.get("launch_count", "N/A")}<br>
                    ML best model: {SUPERVISED_ML.get("best_model_name", "N/A")}
                </div>
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # Main input + output area
    # --------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='input-card'>")

            gr.HTML("<div class='section-title'>Historical Launch Test</div>")
            inp_historical_launch = gr.Dropdown(
                label="Historical launch",
                choices=HISTORICAL_LAUNCH_CHOICES,
                value=None
            )

            btn_test_historical = gr.Button(
                "🧪 Run Historical Test",
                variant="secondary"
            )

            gr.HTML("<div class='section-title'>New Launch Input</div>")

            inp_product_name = gr.Textbox(
                label="Product name",
                value="",
                placeholder="Example: Daily Carnitin"
            )

            inp_product_need_area = gr.Dropdown(
                label="Product need area",
                choices=PRODUCT_NEED_AREA_CHOICES,
                value=None
            )

            inp_benefit_keywords = gr.Dropdown(
                label="Benefit keywords",
                choices=BENEFIT_KEYWORD_CHOICES,
                value=[],
                multiselect=True
            )

            inp_month_year = gr.Dropdown(
                label="Launch month",
                choices=MONTH_YEAR_CHOICES,
                value=None
            )

            inp_product_form = gr.Dropdown(
                label="Product form",
                choices=PRODUCT_FORM_CHOICES,
                value=None
            )

            inp_strategy = gr.Dropdown(
                label="Launch strategy type",
                choices=LAUNCH_STRATEGY_CHOICES,
                value=None
            )

            inp_uvp = gr.Number(
                label="UVP in EUR",
                value=None
            )

            inp_flavour = gr.Dropdown(
                label="Flavour",
                choices=FLAVOUR_CHOICES,
                value=None
            )

            inp_flavour_group = gr.Dropdown(
                label="Flavour group",
                choices=FLAVOUR_GROUP_CHOICES,
                value=None
            )

            btn = gr.Button(
                "✨ Generate Forecast",
                variant="primary",
                size="lg"
            )

            gr.HTML("<div class='section-title'>Summary</div>")

            out_summary = gr.Textbox(
                label="",
                lines=16,
                interactive=False
            )

            out_run_record_status = gr.Textbox(
                label="Run record",
                lines=3,
                interactive=False
            )

            gr.HTML("</div>")

        with gr.Column(scale=2):
            gr.HTML("<div class='output-card'>")

            gr.HTML("<div class='section-title'>Forecast Output</div>")

            out_forecast_table = gr.DataFrame(
                label="Forecast table",
                interactive=False,
                wrap=True
            )

            out_chart = gr.Plot(
                label="Forecast chart"
            )

            gr.HTML("<div class='section-title'>Behavioral Segment Impact</div>")

            out_behavioral_table = gr.DataFrame(
                label="Behavioral segmentation contribution",
                interactive=False,
                wrap=True
            )

            gr.HTML("<div class='section-title'>ML Model Benchmark</div>")

            out_ml_model_summary = gr.DataFrame(
                label="Model comparison summary",
                interactive=False,
                wrap=True
            )

            gr.HTML("<div class='section-title'>Forecast Explanation</div>")

            out_explanation = gr.Textbox(
                label="",
                lines=18,
                interactive=False
            )

            gr.HTML("</div>")

    # --------------------------------------------------------
    # Similar historical launches
    # --------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")

            gr.HTML("<div class='section-title'>Similar Historical Launches</div>")

            out_similar_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True
            )

            gr.HTML("</div>")

    # --------------------------------------------------------
    # Historical launch comparison
    # --------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")

            gr.HTML("<div class='section-title'>Historical Launch Comparison</div>")

            out_historical_comparison = gr.DataFrame(
                label="Historical vs forecast",
                interactive=False,
                wrap=True
            )

            out_historical_summary = gr.Textbox(
                label="",
                lines=8,
                interactive=False
            )

            gr.HTML("</div>")

    # --------------------------------------------------------
    # Applied ratio factors - now full width
    # --------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")

            gr.HTML("<div class='section-title'>Applied Ratio Factors</div>")

            out_factor_table = gr.DataFrame(
                label="",
                interactive=False,
                wrap=True,
                elem_classes=["factor-table-wide"]
            )

            gr.HTML("</div>")

    # --------------------------------------------------------
    # Button actions
    # --------------------------------------------------------
    btn.click(
        fn=run_forecast,
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
            out_ml_model_summary,
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
        inputs=[
            inp_historical_launch
        ],
        outputs=[
            out_forecast_table,
            out_behavioral_table,
            out_ml_model_summary,
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
    demo.launch(debug=True)