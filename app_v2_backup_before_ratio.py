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
from sklearn.metrics.pairwise import cosine_similarity


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

FLAVOUR_CHOICES = [
    "Lemon Ice Tea",
    "Mango Passionfruit",
    "Berrymix",
    "Orange",
    "Coconut-Pineapple",
    "Cherry",
    "Ginger Lemon",
    "Lemon Coconut",
    "2 Flavours",
    "Blueberry",
    "Red Fruits",
    "Coffee",
    "Creamy Matcha",
    "Creamy Pistachio",
    "Vanilla Cinnamon",
    "Pink Grapefruit",
    "Pomegranate Hibiscus",
    "Mango Maracuja",
    "Cookie",
    "Pomegranate Ice Tea",
    "Lavender",
    "Caramel",
    "Chocolate Caramel",
    "Cucumber",
    "Creamy Vanilla",
    "Peach Ice Tea",
    "Raspberry",
    "Strawberry",
    "3 Flavours",
    "Apple Strudel",
    "null",
    "Blueberry Lemon",
    "Apple Kiwi",
    "Neutral",
    "Mocha",
    "Choco",
    "Almond Vanilla",
    "Cinnamon",
    "No Flavour",
    "New Flavour",
]

FLAVOUR_TYPE_CHOICES = [
    "sweet",
    "sour",
    "bitter",
    "neutral",
    "fruity",
    "creamy",
    "coffee_chocolate",
    "spiced",
    "herbal_floral",
    "no_flavour",
    "new_flavour",
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


def build_future_month_year_choices(n_months=12):
    """
    Build future launch month choices in MM-YYYY format.
    Current and past months are not included.
    """
    today = pd.Timestamp.today().normalize()
    start = (today + pd.offsets.MonthBegin(1)).replace(day=1)

    return [
        (start + pd.DateOffset(months=i)).strftime("%m-%Y")
        for i in range(n_months)
    ]


def parse_launch_month_year(launch_month_year):
    """
    Parse MM-YYYY launch month input.
    If invalid or not future, fallback to next month.
    """
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


def safe_choice(value, choices, fallback):
    if value in choices:
        return value
    return fallback


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
SEMANTIC_SIMILARITY = ARTIFACTS.get("semantic_similarity", {})
BEHAVIORAL_SEGMENTATION = ARTIFACTS.get("behavioral_segmentation", {})

MONTH_YEAR_CHOICES = build_future_month_year_choices(12)


# ============================================================
# SIMILARITY ENGINE
# ============================================================

def score_launch_similarity(
    row,
    semantic_score,
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

    rule_score = (
        0.20 * product_score
        + 0.20 * use_case_score
        + 0.14 * target_group_score
        + 0.14 * flavour_score
        + 0.12 * product_form_score
        + 0.10 * strategy_score
        + 0.05 * price_score
        + 0.05 * month_score
    )

    total_score = 0.55 * semantic_score + 0.45 * rule_score

    return {
        "similarity_score": float(total_score),
        "semantic_score": float(semantic_score),
        "rule_score": float(rule_score),
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
    semantic_scores = np.zeros(len(LAUNCHES), dtype=float)

    if SEMANTIC_SIMILARITY.get("enabled", False):
        vectorizer = SEMANTIC_SIMILARITY.get("vectorizer")
        launch_matrix = SEMANTIC_SIMILARITY.get("launch_matrix")

        if vectorizer is not None and launch_matrix is not None:
            query_text = " ".join(
                [
                    normalize_text(product_name),
                    normalize_text(use_case),
                    normalize_text(target_group),
                    normalize_text(flavour),
                    normalize_text(product_form),
                    normalize_strategy(launch_strategy_type),
                ]
            ).strip()
            q_vec = vectorizer.transform([query_text])
            semantic_scores = cosine_similarity(q_vec, launch_matrix).flatten()

    scored_rows = []

    for idx, (_, row) in enumerate(LAUNCHES.iterrows()):
        scores = score_launch_similarity(
            row=row,
            semantic_score=float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0,
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


def build_behavioral_segment_table(similar_launches, fw_qty_base, six_qty_base, fw_nc_base, six_nc_base):
    if not BEHAVIORAL_SEGMENTATION.get("enabled", False):
        return pd.DataFrame(
            [
                {
                    "Segment": "N/A",
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
    ref["similarity_score"] = pd.to_numeric(ref["similarity_score"], errors="coerce").fillna(0.0)
    ref = ref[ref["similarity_score"] > 0]

    merged = ref.merge(profile, on="sku", how="left")
    merged = merged.dropna(subset=["segment_key"]).copy()

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "Segment": "N/A",
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

    seg_summary_small = seg_summary[["segment_key", "global_share", "avg_frequency", "avg_monetary"]].copy()
    agg = agg.merge(seg_summary_small, on="segment_key", how="left")

    agg["Interpretation"] = np.where(
        agg["affinity"] >= agg["global_share"].fillna(0),
        "Over-indexed vs global mix",
        "Under-indexed vs global mix",
    )

    agg.rename(
        columns={
            "segment_key": "Segment",
            "affinity": "Affinity score",
        },
        inplace=True,
    )

    agg["Affinity score"] = agg["Affinity score"].round(3)
    agg["global_share"] = agg["global_share"].fillna(0).round(3)
    agg["avg_frequency"] = agg["avg_frequency"].fillna(0).round(2)
    agg["avg_monetary"] = agg["avg_monetary"].fillna(0).round(2)

    return agg[
        [
            "Segment",
            "Affinity score",
            "First week units (base)",
            "First 6 week units (base)",
            "First week NC (base)",
            "First 6 week NC (base)",
            "global_share",
            "avg_frequency",
            "avg_monetary",
            "Interpretation",
        ]
    ]


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

    base = weighted_average(
        similar_launches[metric],
        similar_launches["similarity_score"],
    )

    if pd.isna(base):
        base = LAUNCHES[metric].mean()

    seasonality_factor = SEASONALITY_INDEX.get(int(launch_month), 1.0)

    strategy = normalize_strategy(launch_strategy_type)
    strategy_factor = STRATEGY_FACTORS.get(strategy, {}).get(metric, 1.0)

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
            f"({row.get('launch_strategy_type', '')}) | total: {row['similarity_score']:.2f} "
            f"(semantic: {row.get('semantic_score', np.nan):.2f}, rule: {row.get('rule_score', np.nan):.2f})"
        )

    lines.append("")
    lines.append("How this output is calculated:")
    lines.append("1) Semantic + rule-based similarity scoring against historical launches")
    lines.append("2) Similar-launch weighted baseline for each KPI")
    lines.append("3) Multiplicative calibration factors: seasonality, strategy, flavour, product form, price elasticity, company growth, sale overlap")
    lines.append("4) Confidence-derived worst/base/best scenario ranges")
    lines.append("5) Behavioral segment contribution estimated from launch-window purchase patterns")

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
    launch_month_year,
    product_form_ui,
    launch_strategy_type,
    uvp,
    flavour,
    flavour_type,
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
):
    launch_date_parsed = parse_launch_month_year(launch_month_year)
    launch_month = int(launch_date_parsed.month)
    strategy = normalize_strategy(launch_strategy_type)

    target_group = ""
    product_form = canonical_product_form(product_form_ui)

    if not product_name:
        product_name = "New Product"

    uvp = to_float(uvp, default=np.nan)
    if pd.isna(uvp) or uvp <= 0:
        uvp = LAUNCHES["uvp"].dropna().mean()

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
                "Metric": "First week total customers",
                "Worst case": "",
                "Base case": int(round(forecasts["first_week_total_c"])),
                "Best case": "",
            },
            {
                "Metric": "First 6 week total customers",
                "Worst case": "",
                "Base case": int(round(forecasts["first_6_week_total_c"])),
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
        "flavour",
        "product_form",
        "launch_strategy_type",
        "launch_date",
        "uvp",
        "first_week_quantity",
        "first_6_week_quantity",
        "first_week_nc",
        "first_6_week_nc",
        "semantic_score",
        "rule_score",
        "similarity_score",
    ]

    existing_similar_cols = [col for col in similar_table_cols if col in similar_launches.columns]
    similar_table = similar_launches[existing_similar_cols].copy()

    if "similarity_score" in similar_table.columns:
        similar_table["similarity_score"] = similar_table["similarity_score"].round(3)

    if "launch_date" in similar_table.columns:
        similar_table["launch_date"] = pd.to_datetime(similar_table["launch_date"]).dt.date.astype(str)

    factor_rows = []

    for metric in ["first_week_quantity", "first_6_week_quantity", "first_week_nc", "first_6_week_nc"]:
        details = factor_details_by_metric[metric]
        factor_rows.append(
            {
                "Metric": metric,
                "Similar launch baseline": round(details["base"], 1),
                "Seasonality": round(details["seasonality_factor"], 2),
                "Strategy": round(details["strategy_factor"], 2),
                "Flavour": round(details["flavour_factor"], 2),
                "Product form": round(details["product_form_factor"], 2),
                "Price": round(details["price_factor"], 2),
                "Growth": round(details["growth_factor"], 2),
                "Sale": round(details["sale_factor"], 2),
            }
        )

    factor_table = pd.DataFrame(factor_rows)

    target_group_table = build_target_group_table(
        product_name=product_name,
        use_case=use_case,
        flavour=flavour,
        launch_strategy_type=strategy,
        uvp=uvp,
    )

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
        x=["First week qty", "First 6 week qty", "First week NC", "First 6 week NC"],
        y=[fw_qty_base, six_qty_base, fw_nc_base, six_nc_base],
        name="Base forecast",
        marker_color=["#b89572", "#a98467", "#d8bfa5", "#c9aa8c"],
    )
)

    fig.update_layout(
    title=dict(
        text=f"Launch Forecast | Confidence: {confidence:.2f}",
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
    xaxis=dict(
        gridcolor="#eadccb",
        zerolinecolor="#d8c4ad",
    ),
    yaxis=dict(
        gridcolor="#eadccb",
        zerolinecolor="#d8c4ad",
    ),
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
        f"Recommended first-order quantity: {proposed_first_order_qty}\n"
        f"Rationale: {proposed_rationale}\n"
        f"Base first week new customers: {fw_nc_base}\n"
        f"Base first 6 week new customers: {six_nc_base}\n"
        f"Product form: {product_form_ui}\n"
        f"Flavour: {flavour}\n"
        f"Flavour type: {flavour_type}\n"
        f"Confidence: {confidence:.2f}"
    )

    run_id = f"FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    top_target_group = ""
    if not target_group_table.empty:
        top_target_group = str(target_group_table.iloc[0].get("Predicted Target Group", ""))

    reference_skus = "|".join(
        similar_table["sku"].astype(str).head(5).tolist()
    ) if "sku" in similar_table.columns else ""

    append_forecast_run_log(
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "product_name": product_name,
            "use_case": use_case,
            "target_group_input": target_group,
            "flavour": flavour,
            "flavour_type": flavour_type,
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
            "top_predicted_target_group": top_target_group,
            "reference_skus": reference_skus,
        }
    )

    run_record_status = (
        f"Run ID: {run_id}\n"
        f"Review decision: {review_decision}\n"
        f"Saved to: {FORECAST_RUN_LOG_PATH}"
    )

    return (
        forecast_table,
        target_group_table,
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
# GRADIO UI
# ============================================================

def run_forecast_ui(
    product_name,
    use_case,
    launch_month_year,
    product_form_ui,
    launch_strategy_type,
    uvp,
    flavour,
    flavour_type,
):
    """
    Simplified UI wrapper.
    The main run_forecast function still has optional backend/review parameters,
    but the visible UI only sends the required launch inputs.
    """
    result = run_forecast(
        product_name=product_name,
        use_case=use_case,
        launch_month_year=launch_month_year,
        product_form_ui=product_form_ui,
        launch_strategy_type=launch_strategy_type,
        uvp=uvp,
        flavour=flavour,
        flavour_type=flavour_type,
    )

    (
        forecast_table,
        target_group_table,
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
        target_group_table,
        behavioral_segment_table,
        fig,
        similar_table,
        factor_table,
        explanation,
        summary_text,
        run_record_status,
    )

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

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 30px rgba(141, 111, 85, 0.30) !important;
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

.dataframe, .table-wrap {
    border-radius: 18px !important;
    overflow: hidden !important;
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
    css=CSS,
    theme=theme,
) as demo:

    gr.HTML(
        f"""
        <div class="header-box">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:18px;">
                <div>
                    <div class="hero-title">
                        🌸 Demand Forecasting Agent V2
                    </div>
                    <div class="hero-subtitle">
                        Forecast first-week and first-six-week launch demand using similar launches,
                        seasonality, flavour, strategy, price and growth signals.
                    </div>
                </div>
                <div class="small-muted" style="text-align:right; min-width:230px;">
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
            gr.HTML("<div class='input-card'>")
            gr.HTML("<div class='section-title'>New Launch Input</div>")

            inp_product_name = gr.Textbox(
                label="Product name",
                value="Daily Carnitin",
                placeholder="Example: Daily Carnitin",
            )

            inp_use_case = gr.Textbox(
                label="Use case",
                value="For individuals who train a lot and sporty people",
                placeholder="Describe what the product is for",
                lines=4,
            )

            inp_month_year = gr.Dropdown(
                label="Launch month",
                choices=MONTH_YEAR_CHOICES,
                value=MONTH_YEAR_CHOICES[0] if MONTH_YEAR_CHOICES else None,
            )

            inp_product_form = gr.Dropdown(
                label="Product form",
                choices=PRODUCT_FORM_CHOICES,
                value=safe_choice("Drinking powder", PRODUCT_FORM_CHOICES, PRODUCT_FORM_CHOICES[0]),
            )

            inp_strategy = gr.Dropdown(
                label="Launch strategy type",
                choices=["standard", "co_creation", "limited_edition"],
                value="standard",
            )

            inp_uvp = gr.Number(
                label="UVP in EUR",
                value=29.90,
                minimum=0.1,
            )

            inp_flavour = gr.Dropdown(
                label="Flavour",
                choices=FLAVOUR_CHOICES,
                value=safe_choice("Orange", FLAVOUR_CHOICES, FLAVOUR_CHOICES[0]),
            )

            inp_flavour_type = gr.Dropdown(
                label="Flavour type",
                choices=FLAVOUR_TYPE_CHOICES,
                value="sweet",
            )

            btn = gr.Button("✨ Generate Forecast", variant="primary", size="lg")

            gr.HTML("<div class='section-title'>Summary</div>")
            out_summary = gr.Textbox(
                label="",
                lines=9,
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

            gr.HTML("<div class='section-title'>Target Group Recommendation</div>")
            out_target_group_table = gr.DataFrame(
                label="Target-group inference",
                interactive=False,
                wrap=True,
            )

            gr.HTML("<div class='section-title'>Behavioral Segment Impact</div>")
            out_behavioral_table = gr.DataFrame(
                label="Behavioral segmentation contribution",
                interactive=False,
                wrap=True,
            )

            gr.HTML("<div class='section-title'>Forecast Explanation</div>")
            out_explanation = gr.Textbox(
                label="",
                lines=14,
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

        with gr.Column(scale=1):
            gr.HTML("<div class='output-card'>")
            gr.HTML("<div class='section-title'>Applied Factors</div>")
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
            inp_use_case,
            inp_month_year,
            inp_product_form,
            inp_strategy,
            inp_uvp,
            inp_flavour,
            inp_flavour_type,
        ],
        outputs=[
            out_forecast_table,
            out_target_group_table,
            out_behavioral_table,
            out_chart,
            out_similar_table,
            out_factor_table,
            out_explanation,
            out_summary,
            out_run_record_status,
        ],
    )



if __name__ == "__main__":
    demo.launch(debug=True)
