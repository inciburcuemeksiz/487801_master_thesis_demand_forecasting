import os
import re
import pickle
import unicodedata
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = os.getenv("DATA_DIR", "data/raw")

ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
SALE_TIMES_PATH = os.path.join(DATA_DIR, "sale_times.csv")
LAUNCHES_PATH = os.path.join(DATA_DIR, "launched_product_details.csv")
TARGET_GROUP_MAPPING_PATH = os.path.join(DATA_DIR, "target_group_mapping.csv")

ARTIFACT_DIR = "artifacts"
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "model_artifacts_v2.pkl")

TARGET_GROUP_TOP_N = 8
RANDOM_STATE = 42

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]

ORDER_REQUIRED_COLS = [
    "order_id",
    "customer_nr",
    "sku",
    "price",
    "date",
    "artikel_name",
    "net_revenue",
    "quantity",
    "flavour",
    "product_category",
    "product",
    "customer_status",
]

SALE_REQUIRED_COLS = ["name", "start_d", "end_d"]

LAUNCH_REQUIRED_COLS = [
    "sku",
    "artikel_name",
    "product",
    "product_need_area",
    "benefit_keywords",
    "flavour",
    "flavour_group",
    "product_form",
    "launch_date",
    "first_order_quantity",
    "uvp",
    "launch_strategy_type",
    "Product Use Case / What it is about",
    "Target Group",
    *TARGET_METRICS,
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(message):
    print(message, flush=True)


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"[^a-z0-9äöüß\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def clean_numeric(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if x in ["", "-", "-%", "nan", "None"]:
        return np.nan

    x = x.replace("€", "").replace("%", "").strip()

    if re.match(r"^\d{1,3}(,\d{3})+$", x):
        x = x.replace(",", "")
    else:
        x = x.replace(",", ".")

    x = re.sub(r"[^0-9.\-]", "", x)
    if x in ["", "-", "."]:
        return np.nan
    return float(x)


def normalize_strategy(x):
    if pd.isna(x):
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


def normalize_keyword_field(value):
    if pd.isna(value):
        return ""
    tokens = [normalize_text(token) for token in str(value).split(",") if str(token).strip()]
    return ", ".join(token for token in tokens if token)


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


def require_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_raw_data():
    log("Loading raw data...")
    orders = pd.read_csv(ORDERS_PATH, low_memory=False)
    sale_times = pd.read_csv(SALE_TIMES_PATH, sep=";", low_memory=False)
    launches = pd.read_csv(LAUNCHES_PATH, low_memory=False)

    for df in [orders, sale_times, launches]:
        df.columns = df.columns.str.strip()

    log(f"orders: {orders.shape}")
    log(f"sale_times: {sale_times.shape}")
    log(f"launches: {launches.shape}")
    return orders, sale_times, launches


def clean_orders(orders):
    df = orders.copy()
    require_columns(df, ORDER_REQUIRED_COLS, "orders.csv")

    date_cols = ["date", "first_order_date", "last_order_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["quantity", "price", "net_revenue", "months_since_first_order", "nr_of_purchase"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["date"].notna()
        & df["customer_nr"].notna()
        & df["sku"].notna()
        & (df["quantity"].fillna(0) > 0)
    ].copy()

    remove_pattern = "shipping|versand|discount|rabatt|gutschein|gift card|free article|gratis"
    combined_text = ""
    for col in ["artikel_name", "product", "product_category"]:
        combined_text += " " + df[col].astype(str)

    df = df[~combined_text.str.lower().str.contains(remove_pattern, regex=True, na=False)].copy()

    df["sku"] = df["sku"].astype(str)
    df["product_norm"] = df["product"].apply(normalize_text)
    df["flavour_norm"] = df["flavour"].apply(normalize_text)
    df["category_norm"] = df["product_category"].apply(normalize_text)
    df["artikel_name_norm"] = df["artikel_name"].apply(normalize_text)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    log(f"orders cleaned: {df.shape}")
    return df


def clean_sale_times(sale_times):
    df = sale_times.copy()
    require_columns(df, SALE_REQUIRED_COLS, "sale_times.csv")

    df["start_d"] = pd.to_datetime(df["start_d"], errors="coerce")
    df["end_d"] = pd.to_datetime(df["end_d"], errors="coerce")
    df = df[df["start_d"].notna() & df["end_d"].notna()].copy()
    df["name_norm"] = df["name"].apply(normalize_text)

    log(f"sale_times cleaned: {df.shape}")
    return df


def clean_launches(launches):
    df = launches.copy()
    require_columns(df, LAUNCH_REQUIRED_COLS, "launched_product_details.csv")

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["uvp"] = df["uvp"].apply(clean_numeric)
    df["first_order_quantity"] = df["first_order_quantity"].apply(clean_numeric)

    for col in TARGET_METRICS:
        df[col] = df[col].apply(clean_numeric)

    df = df[df["launch_date"].notna()].copy()
    df["sku"] = df["sku"].astype(str)
    df["launch_strategy_type"] = df["launch_strategy_type"].apply(normalize_strategy)

    norm_map = {
        "product_norm": "product",
        "product_need_area_norm": "product_need_area",
        "benefit_keywords_norm": "benefit_keywords",
        "flavour_norm": "flavour",
        "flavour_group_norm": "flavour_group",
        "product_form_norm": "product_form",
        "use_case_norm": "Product Use Case / What it is about",
        "target_group_norm": "Target Group",
        "artikel_name_norm": "artikel_name",
    }

    for new_col, src_col in norm_map.items():
        if new_col == "benefit_keywords_norm":
            df[new_col] = df[src_col].apply(normalize_keyword_field)
        else:
            df[new_col] = df[src_col].apply(normalize_text)

    df["launch_month"] = df["launch_date"].dt.month
    df["launch_year"] = df["launch_date"].dt.year
    df["launch_year_month"] = df["launch_date"].dt.to_period("M").astype(str)

    df["units_per_customer_1w"] = df["first_week_quantity"] / df["first_week_total_c"].replace(0, np.nan)
    df["units_per_customer_6w"] = df["first_6_week_quantity"] / df["first_6_week_total_c"].replace(0, np.nan)
    df["first_week_share_of_6w"] = df["first_week_quantity"] / df["first_6_week_quantity"].replace(0, np.nan)
    df["new_customer_share_1w"] = df["first_week_nc"] / df["first_week_total_c"].replace(0, np.nan)
    df["new_customer_share_6w"] = df["first_6_week_nc"] / df["first_6_week_total_c"].replace(0, np.nan)

    log(f"launches cleaned: {df.shape}")
    return df


# ============================================================
# SALE FEATURES
# ============================================================

def add_order_sale_flags(orders, sale_times):
    df = orders.copy()
    df["sale_name"] = ""
    df["is_sale_period"] = 0

    for _, sale in sale_times.iterrows():
        mask = (df["date"] >= sale["start_d"]) & (df["date"] <= sale["end_d"])
        df.loc[mask, "sale_name"] = sale["name"]
        df.loc[mask, "is_sale_period"] = 1

    return df


def add_launch_sale_features(launches, sale_times):
    df = launches.copy()
    df["launch_during_sale"] = 0
    df["first_week_sale_days"] = 0
    df["first_6_week_sale_days"] = 0
    df["sale_name_overlap"] = ""

    for idx, row in df.iterrows():
        launch_date = row["launch_date"]
        first_week_end = launch_date + pd.Timedelta(days=6)
        first_6w_end = launch_date + pd.Timedelta(days=41)

        overlap_names = []
        first_week_days = 0
        first_6w_days = 0
        launch_during_sale = 0

        for _, sale in sale_times.iterrows():
            sale_start, sale_end = sale["start_d"], sale["end_d"]

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

        df.loc[idx, "launch_during_sale"] = launch_during_sale
        df.loc[idx, "first_week_sale_days"] = first_week_days
        df.loc[idx, "first_6_week_sale_days"] = first_6w_days
        df.loc[idx, "sale_name_overlap"] = ", ".join(sorted(set(overlap_names)))

    return df


# ============================================================
# CALIBRATION + RATIO ASSETS
# ============================================================

def build_monthly_company_scale(orders):
    monthly = (
        orders.groupby("year_month")
        .agg(
            monthly_quantity=("quantity", "sum"),
            monthly_revenue=("net_revenue", "sum"),
            monthly_orders=("order_id", "nunique"),
            monthly_customers=("customer_nr", "nunique"),
        )
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly["year_month"] + "-01")
    return monthly


def build_seasonality_index(orders):
    month_qty = orders.groupby("month")["quantity"].sum().reindex(range(1, 13))
    avg_month = month_qty.mean()
    return {
        int(month): clip_factor(safe_divide(qty, avg_month, 1.0), 0.6, 1.5)
        for month, qty in month_qty.items()
    }


def build_group_factors(launches, group_col, low=0.6, high=1.7):
    factors = {}
    if group_col not in launches.columns:
        return factors

    for group_value, group in launches.groupby(group_col):
        if not group_value:
            continue
        factors[group_value] = {}
        for metric in TARGET_METRICS:
            factors[group_value][metric] = clip_factor(
                safe_divide(group[metric].mean(), launches[metric].mean(), 1.0),
                low,
                high,
            )
    return factors


def build_strategy_factors(launches):
    factors = {}
    standard = launches[launches["launch_strategy_type"] == "standard"]

    for strategy, group in launches.groupby("launch_strategy_type"):
        factors[strategy] = {}
        for metric in TARGET_METRICS:
            factors[strategy][metric] = clip_factor(
                safe_divide(group[metric].mean(), standard[metric].mean(), 1.0),
                0.5,
                2.2,
            )
    return factors


def build_sale_factors(launches):
    factors = {}
    for metric in TARGET_METRICS:
        baseline = launches[metric].mean()
        factors[metric] = {
            "launch_during_sale": clip_factor(
                safe_divide(launches.loc[launches["launch_during_sale"] == 1, metric].mean(), baseline, 1.0),
                0.6,
                1.8,
            ),
            "first_week_sale_overlap": clip_factor(
                safe_divide(launches.loc[launches["first_week_sale_days"] > 0, metric].mean(), baseline, 1.0),
                0.6,
                1.8,
            ),
            "first_6_week_sale_overlap": clip_factor(
                safe_divide(launches.loc[launches["first_6_week_sale_days"] > 0, metric].mean(), baseline, 1.0),
                0.6,
                1.8,
            ),
        }
    return factors


def build_price_elasticity(launches):
    df = launches[["uvp", "first_6_week_quantity"]].dropna()
    df = df[(df["uvp"] > 0) & (df["first_6_week_quantity"] > 0)]

    if len(df) < 5:
        return -0.5

    coef = np.polyfit(np.log(df["uvp"]), np.log(df["first_6_week_quantity"]), 1)
    return float(np.clip(coef[0], -1.5, -0.1))


def build_growth_context(orders):
    monthly = build_monthly_company_scale(orders).sort_values("date")
    if monthly.empty:
        return {
            "recent_3m_avg_quantity": 1.0,
            "historical_avg_quantity": 1.0,
            "company_growth_factor": 1.0,
        }

    recent_avg = monthly.tail(3)["monthly_quantity"].mean()
    hist_avg = monthly["monthly_quantity"].mean()
    return {
        "recent_3m_avg_quantity": float(recent_avg),
        "historical_avg_quantity": float(hist_avg),
        "company_growth_factor": clip_factor(safe_divide(recent_avg, hist_avg, 1.0), 0.7, 1.8),
    }


def build_monthly_new_customers(orders):
    if "first_order_date" not in orders.columns:
        return pd.DataFrame(columns=["first_order_month", "new_customers", "date"])

    df = (
        orders[["customer_nr", "first_order_date"]]
        .dropna()
        .drop_duplicates("customer_nr")
        .copy()
    )

    if df.empty:
        return pd.DataFrame(columns=["first_order_month", "new_customers", "date"])

    df["first_order_month"] = df["first_order_date"].dt.to_period("M").astype(str)
    monthly = df.groupby("first_order_month")["customer_nr"].nunique().reset_index(name="new_customers")
    monthly["date"] = pd.to_datetime(monthly["first_order_month"] + "-01")
    return monthly.sort_values("date").reset_index(drop=True)


def build_ratio_context(orders, monthly_new_customers):
    max_date = orders["date"].max()
    if pd.isna(max_date):
        return {
            "latest_order_date": None,
            "known_customer_count": 0,
            "active_customer_count_12m": 0,
            "active_customer_count_6m": 0,
            "recent_3m_avg_new_customers": 1.0,
            "historical_avg_new_customers": 1.0,
            "new_customer_growth_factor": 1.0,
        }

    recent_3m = 1.0 if monthly_new_customers.empty else float(monthly_new_customers.tail(3)["new_customers"].mean())
    historical = 1.0 if monthly_new_customers.empty else float(monthly_new_customers["new_customers"].mean())

    return {
        "latest_order_date": str(max_date.date()),
        "known_customer_count": int(orders["customer_nr"].nunique()),
        "active_customer_count_12m": int(orders.loc[orders["date"] >= max_date - pd.DateOffset(months=12), "customer_nr"].nunique()),
        "active_customer_count_6m": int(orders.loc[orders["date"] >= max_date - pd.DateOffset(months=6), "customer_nr"].nunique()),
        "recent_3m_avg_new_customers": recent_3m,
        "historical_avg_new_customers": historical,
        "new_customer_growth_factor": clip_factor(safe_divide(recent_3m, historical, 1.0), 0.5, 2.5),
    }


def build_historical_buyer_ratio_table(orders, launches, monthly_new_customers):
    order_df = orders.copy()
    order_df["sku_str"] = order_df["sku"].astype(str)
    rows = []

    for _, launch in launches.iterrows():
        sku = str(launch["sku"])
        launch_date = launch["launch_date"]
        first_week_end = launch_date + pd.Timedelta(days=6)
        first_6w_end = launch_date + pd.Timedelta(days=41)

        eligible_customers = set(order_df.loc[order_df["date"] < launch_date, "customer_nr"].dropna().astype(str))
        eligible_count = len(eligible_customers)

        buyers_1w = set(
            order_df.loc[
                (order_df["date"] >= launch_date)
                & (order_df["date"] <= first_week_end)
                & (order_df["sku_str"] == sku),
                "customer_nr",
            ].dropna().astype(str)
        )

        buyers_6w = set(
            order_df.loc[
                (order_df["date"] >= launch_date)
                & (order_df["date"] <= first_6w_end)
                & (order_df["sku_str"] == sku),
                "customer_nr",
            ].dropna().astype(str)
        )

        launch_month = launch_date.to_period("M").strftime("%Y-%m")
        monthly_match = monthly_new_customers[monthly_new_customers["first_order_month"] == launch_month]
        monthly_nc = np.nan if monthly_match.empty else float(monthly_match["new_customers"].iloc[0])

        rows.append(
            {
                "sku": sku,
                "product": launch.get("product", ""),
                "flavour": launch.get("flavour", ""),
                "product_form": launch.get("product_form", ""),
                "launch_strategy_type": launch.get("launch_strategy_type", ""),
                "launch_date": launch_date,
                "launch_month": launch.get("launch_month", np.nan),
                "launch_year_month": launch_month,
                "uvp": launch.get("uvp", np.nan),
                "eligible_customers_before_launch": eligible_count,
                "buyers_1w_existing": len(buyers_1w),
                "buyers_6w_existing": len(buyers_6w),
                "buyer_ratio_1w_existing": len(buyers_1w) / eligible_count if eligible_count else np.nan,
                "buyer_ratio_6w_existing": len(buyers_6w) / eligible_count if eligible_count else np.nan,
                "monthly_new_customers_at_launch": monthly_nc,
                "nc_ratio_1w_vs_monthly_nc": safe_divide(launch.get("first_week_nc", np.nan), monthly_nc, np.nan),
                "nc_ratio_6w_vs_monthly_nc": safe_divide(launch.get("first_6_week_nc", np.nan), monthly_nc, np.nan),
                "units_per_customer_1w": safe_divide(launch.get("first_week_quantity", np.nan), launch.get("first_week_total_c", np.nan), np.nan),
                "units_per_customer_6w": safe_divide(launch.get("first_6_week_quantity", np.nan), launch.get("first_6_week_total_c", np.nan), np.nan),
                **{metric: launch.get(metric, np.nan) for metric in TARGET_METRICS},
            }
        )

    ratio_df = pd.DataFrame(rows)
    if ratio_df.empty:
        return ratio_df

    for col in [
        "buyer_ratio_1w_existing",
        "buyer_ratio_6w_existing",
        "nc_ratio_1w_vs_monthly_nc",
        "nc_ratio_6w_vs_monthly_nc",
        "units_per_customer_1w",
        "units_per_customer_6w",
    ]:
        series = pd.to_numeric(ratio_df[col], errors="coerce")
        if series.notna().sum() >= 5:
            ratio_df[f"{col}_clipped"] = series.clip(series.quantile(0.05), series.quantile(0.95))
        else:
            ratio_df[f"{col}_clipped"] = series

    return ratio_df


# ============================================================
# TARGET GROUP INFERENCE
# ============================================================

def ensure_target_group_mapping_file(launches):
    if os.path.exists(TARGET_GROUP_MAPPING_PATH):
        mapping_df = pd.read_csv(TARGET_GROUP_MAPPING_PATH)
        require_columns(mapping_df, ["raw_target_group", "canonical_target_group"], "target_group_mapping.csv")
        mapping_df = mapping_df.dropna().drop_duplicates()
        return mapping_df

    raw_values = sorted({str(x).strip() for x in launches["Target Group"].dropna() if str(x).strip()})
    mapping_df = pd.DataFrame({"raw_target_group": raw_values, "canonical_target_group": raw_values})
    os.makedirs(os.path.dirname(TARGET_GROUP_MAPPING_PATH), exist_ok=True)
    mapping_df.to_csv(TARGET_GROUP_MAPPING_PATH, index=False)
    log(f"Created starter mapping file: {TARGET_GROUP_MAPPING_PATH}")
    return mapping_df


def apply_target_group_mapping(launches, mapping_df):
    df = launches.copy()
    mapping = {
        normalize_text(raw): str(canonical).strip()
        for raw, canonical in mapping_df[["raw_target_group", "canonical_target_group"]].values
        if str(raw).strip() and str(canonical).strip()
    }

    df["raw_target_group"] = df["Target Group"].fillna("").astype(str).str.strip()
    df["raw_target_group_norm"] = df["raw_target_group"].apply(normalize_text)
    df["canonical_target_group"] = df.apply(
        lambda row: mapping.get(row["raw_target_group_norm"], row["raw_target_group"] or "OTHER"),
        axis=1,
    )
    return df


def build_target_group_model(launches, orders, top_n=8):
    log("Building target-group inference model...")
    order_df = orders.copy()
    order_df["is_new_customer"] = order_df["customer_status"].astype(str).str.upper().str.contains("NEW").astype(int)

    order_agg = (
        order_df.groupby("sku")
        .agg(
            order_count=("order_id", "nunique"),
            order_quantity_sum=("quantity", "sum"),
            order_revenue_sum=("net_revenue", "sum"),
            order_avg_price=("price", "mean"),
            order_avg_quantity=("quantity", "mean"),
            order_customer_count=("customer_nr", "nunique"),
            order_new_customer_share=("is_new_customer", "mean"),
            order_sale_share=("is_sale_period", "mean"),
        )
        .reset_index()
    )

    numeric_cols = [
        "uvp",
        "order_count",
        "order_quantity_sum",
        "order_revenue_sum",
        "order_avg_price",
        "order_avg_quantity",
        "order_customer_count",
        "order_new_customer_share",
        "order_sale_share",
    ]

    train_df = launches[
        ["sku", "product_norm", "use_case_norm", "flavour_norm", "launch_strategy_type", "uvp", "canonical_target_group"]
    ].merge(order_agg, on="sku", how="left")

    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    train_df["feature_text"] = (
        train_df["product_norm"].fillna("")
        + " "
        + train_df["use_case_norm"].fillna("")
        + " "
        + train_df["flavour_norm"].fillna("")
        + " "
        + train_df["launch_strategy_type"].fillna("standard")
    ).str.strip()

    class_counts = train_df["canonical_target_group"].value_counts()
    top_classes = class_counts.head(top_n).index.tolist()
    y = np.where(train_df["canonical_target_group"].isin(top_classes), train_df["canonical_target_group"], "OTHER")

    if len(train_df) < 10 or pd.Series(y).nunique() < 2:
        return {
            "enabled": False,
            "reason": "Insufficient labeled data",
            "training_rows": int(len(train_df)),
            "top_classes": top_classes,
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "mapping_path": TARGET_GROUP_MAPPING_PATH,
        }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_text = vectorizer.fit_transform(train_df["feature_text"].fillna(""))
    x_num = sparse.csr_matrix(train_df[numeric_cols].astype(float).values)
    x_train = sparse.hstack([x_text, x_num], format="csr")

    model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    model.fit(x_train, y)

    return {
        "enabled": True,
        "reason": "ok",
        "model": model,
        "vectorizer": vectorizer,
        "numeric_feature_columns": numeric_cols,
        "numeric_feature_defaults": {col: float(train_df[col].mean()) for col in numeric_cols},
        "training_rows": int(len(train_df)),
        "classes": model.classes_.tolist(),
        "top_classes": top_classes,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "mapping_path": TARGET_GROUP_MAPPING_PATH,
    }


# ============================================================
# SIMILARITY
# ============================================================

def build_semantic_similarity_assets(launches):
    log("Building semantic similarity assets...")
    df = launches.copy()
    df["semantic_text"] = (
        df["product_norm"].fillna("")
        + " "
        + df["product_need_area_norm"].fillna("")
        + " "
        + df["benefit_keywords_norm"].fillna("")
        + " "
        + df["use_case_norm"].fillna("")
        + " "
        + df["target_group_norm"].fillna("")
        + " "
        + df["flavour_norm"].fillna("")
        + " "
        + df["flavour_group_norm"].fillna("")
        + " "
        + df["product_form_norm"].fillna("")
        + " "
        + df["launch_strategy_type"].fillna("standard")
    ).str.strip()

    if len(df) < 3:
        return {"enabled": False, "reason": "Insufficient launch rows", "row_count": int(len(df))}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(df["semantic_text"].fillna(""))
    return {
        "enabled": True,
        "reason": "ok",
        "vectorizer": vectorizer,
        "launch_matrix": matrix,
        "row_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
    }


# ============================================================
# BEHAVIORAL SEGMENTATION
# ============================================================

def label_behavioral_segments(seg_summary):
    df = seg_summary.copy()
    if df.empty:
        df["segment_label"] = ""
        df["segment_description"] = ""
        return df

    labels = {}
    descriptions = {}
    assigned = set()

    def assign(seg_key, label, description):
        if seg_key not in assigned:
            labels[seg_key] = label
            descriptions[seg_key] = description
            assigned.add(seg_key)

    candidates = [
        (
            df.sort_values(["avg_monetary", "avg_frequency"], ascending=[False, False]),
            "Loyal high-value buyers",
            "High order frequency and high total spend.",
        ),
        (
            df.sort_values(["avg_launch_share_24m", "avg_unique_launch_skus_24m"], ascending=[False, False]),
            "Launch adopters",
            "Above-average tendency to buy newly launched SKUs.",
        ),
        (
            df.sort_values("avg_sale_share", ascending=False),
            "Sale-sensitive buyers",
            "Purchases are concentrated in sale or campaign periods.",
        ),
        (
            df.sort_values(["avg_product_diversity_ratio_24m", "avg_unique_product_count_24m"], ascending=[False, False]),
            "Variety seekers",
            "Customers who try a wider mix of products, categories, or flavours.",
        ),
        (
            df.sort_values(["avg_recency_days", "avg_monetary", "avg_frequency"], ascending=[False, True, True]),
            "Dormant low-value customers",
            "Old last purchase dates, low frequency, and low total spend.",
        ),
    ]

    for candidate_df, label, desc in candidates:
        candidate_df = candidate_df[~candidate_df["segment_key"].isin(assigned)]
        if not candidate_df.empty:
            assign(candidate_df.iloc[0]["segment_key"], label, desc)

    for _, row in df.iterrows():
        assign(row["segment_key"], "Broad occasional customers", "General segment with occasional purchasing behavior.")

    df["segment_label"] = df["segment_key"].map(labels)
    df["segment_description"] = df["segment_key"].map(descriptions)
    return df


def build_behavioral_segmentation(orders, launches, n_clusters=5, fit_sample_size=50000):
    log("Building behavioral segmentation...")

    required_cols = {
        "customer_nr",
        "date",
        "order_id",
        "net_revenue",
        "price",
        "quantity",
        "is_sale_period",
        "sku",
        "product",
        "product_category",
        "flavour",
    }

    if not required_cols.issubset(orders.columns):
        return {"enabled": False, "reason": f"Missing columns: {sorted(required_cols - set(orders.columns))}"}

    df = orders[orders["customer_nr"].notna()].copy()
    if df.empty:
        return {"enabled": False, "reason": "No customer records"}

    df["sku_str"] = df["sku"].astype(str)
    launch_skus = set(launches["sku"].astype(str))
    limited_skus = set(launches.loc[launches["launch_strategy_type"] == "limited_edition", "sku"].astype(str))
    co_creation_skus = set(launches.loc[launches["launch_strategy_type"] == "co_creation", "sku"].astype(str))

    max_date = df["date"].max()
    df["is_launch_sku"] = df["sku_str"].isin(launch_skus).astype(int)
    df["is_limited_edition_sku"] = df["sku_str"].isin(limited_skus).astype(int)
    df["is_co_creation_sku"] = df["sku_str"].isin(co_creation_skus).astype(int)

    customer_features = (
        df.groupby("customer_nr")
        .agg(
            frequency=("order_id", "nunique"),
            monetary=("net_revenue", "sum"),
            avg_price=("price", "mean"),
            avg_quantity=("quantity", "mean"),
            sale_share=("is_sale_period", "mean"),
            last_order_date=("date", "max"),
        )
        .reset_index()
    )
    customer_features["recency_days"] = (max_date - customer_features["last_order_date"]).dt.days

    recent = df[df["date"] >= max_date - pd.DateOffset(months=24)].copy()
    recent_features = (
        recent.groupby("customer_nr")
        .agg(
            order_lines_24m=("sku_str", "count"),
            unique_product_count_24m=("product", "nunique"),
            unique_category_count_24m=("product_category", "nunique"),
            unique_flavour_count_24m=("flavour", "nunique"),
            launch_purchase_count_24m=("is_launch_sku", "sum"),
            limited_edition_purchase_count_24m=("is_limited_edition_sku", "sum"),
            co_creation_purchase_count_24m=("is_co_creation_sku", "sum"),
        )
        .reset_index()
    )

    unique_launch_skus = (
        recent[recent["is_launch_sku"] == 1]
        .groupby("customer_nr")["sku_str"]
        .nunique()
        .reset_index(name="unique_launch_skus_24m")
    )

    customer_features = customer_features.merge(recent_features, on="customer_nr", how="left")
    customer_features = customer_features.merge(unique_launch_skus, on="customer_nr", how="left")

    fill_zero_cols = [
        "order_lines_24m",
        "unique_product_count_24m",
        "unique_category_count_24m",
        "unique_flavour_count_24m",
        "launch_purchase_count_24m",
        "unique_launch_skus_24m",
        "limited_edition_purchase_count_24m",
        "co_creation_purchase_count_24m",
    ]
    customer_features[fill_zero_cols] = customer_features[fill_zero_cols].fillna(0)

    customer_features["launch_share_24m"] = (
        customer_features["launch_purchase_count_24m"] / customer_features["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)
    customer_features["product_diversity_ratio_24m"] = (
        customer_features["unique_product_count_24m"] / customer_features["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)
    customer_features["flavour_diversity_ratio_24m"] = (
        customer_features["unique_flavour_count_24m"] / customer_features["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)

    feature_cols = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_price",
        "avg_quantity",
        "sale_share",
        "launch_purchase_count_24m",
        "unique_launch_skus_24m",
        "launch_share_24m",
        "unique_product_count_24m",
        "unique_category_count_24m",
        "unique_flavour_count_24m",
        "product_diversity_ratio_24m",
        "flavour_diversity_ratio_24m",
        "limited_edition_purchase_count_24m",
        "co_creation_purchase_count_24m",
    ]

    for col in feature_cols:
        customer_features[col] = pd.to_numeric(customer_features[col], errors="coerce")
        customer_features[col] = customer_features[col].replace([np.inf, -np.inf], np.nan)
        customer_features[col] = customer_features[col].fillna(customer_features[col].median())

    if len(customer_features) < 20:
        return {"enabled": False, "reason": "Not enough customers", "customer_count": int(len(customer_features))}

    cluster_count = min(n_clusters, max(2, len(customer_features) // 25))
    fit_df = customer_features.sample(n=min(fit_sample_size, len(customer_features)), random_state=RANDOM_STATE)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_df[feature_cols])

    model = MiniBatchKMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=20, batch_size=2048)
    model.fit(x_fit)

    customer_features["segment_id"] = model.predict(scaler.transform(customer_features[feature_cols])).astype(int)
    customer_features["segment_key"] = customer_features["segment_id"].apply(lambda x: f"SEG_{int(x)}")

    seg_summary = (
        customer_features.groupby("segment_key")
        .agg(
            customer_count=("customer_nr", "nunique"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_sale_share=("sale_share", "mean"),
            avg_launch_purchase_count_24m=("launch_purchase_count_24m", "mean"),
            avg_unique_launch_skus_24m=("unique_launch_skus_24m", "mean"),
            avg_launch_share_24m=("launch_share_24m", "mean"),
            avg_unique_product_count_24m=("unique_product_count_24m", "mean"),
            avg_unique_category_count_24m=("unique_category_count_24m", "mean"),
            avg_unique_flavour_count_24m=("unique_flavour_count_24m", "mean"),
            avg_product_diversity_ratio_24m=("product_diversity_ratio_24m", "mean"),
            avg_flavour_diversity_ratio_24m=("flavour_diversity_ratio_24m", "mean"),
            avg_limited_edition_purchase_count_24m=("limited_edition_purchase_count_24m", "mean"),
            avg_co_creation_purchase_count_24m=("co_creation_purchase_count_24m", "mean"),
        )
        .reset_index()
    )
    seg_summary["global_share"] = seg_summary["customer_count"] / max(seg_summary["customer_count"].sum(), 1)
    seg_summary = label_behavioral_segments(seg_summary)

    launch_segment_profile = build_launch_segment_profile(df, launches, customer_features)

    return {
        "enabled": True,
        "reason": "ok",
        "feature_columns": feature_cols,
        "scaler": scaler,
        "model": model,
        "customer_count": int(len(customer_features)),
        "cluster_count": int(cluster_count),
        "segment_summary": seg_summary,
        "launch_segment_profile": launch_segment_profile,
    }


def build_launch_segment_profile(order_df, launches, customer_features):
    customer_segment = customer_features[["customer_nr", "segment_key"]].copy()
    rows = []

    for _, launch in launches[["sku", "launch_date"]].dropna().drop_duplicates().iterrows():
        sku = str(launch["sku"])
        launch_date = launch["launch_date"]
        window_end = launch_date + pd.Timedelta(days=41)

        buyers = order_df.loc[
            (order_df["sku_str"] == sku)
            & (order_df["date"] >= launch_date)
            & (order_df["date"] <= window_end),
            ["customer_nr"],
        ].drop_duplicates()

        if buyers.empty:
            continue

        buyer_segments = buyers.merge(customer_segment, on="customer_nr", how="left")
        shares = buyer_segments["segment_key"].value_counts(normalize=True)

        for seg_key, share in shares.items():
            rows.append({"sku": sku, "segment_key": seg_key, "launch_segment_share": float(share)})

    return pd.DataFrame(rows)


# ============================================================
# SUPERVISED ML MODEL COMPARISON
# ============================================================

def build_launch_ml_table(launches, launch_ratio_table, behavioral_segmentation):
    log("Building supervised ML training table...")
    df = launches.copy()

    ratio_cols = [
        "sku",
        "eligible_customers_before_launch",
        "buyer_ratio_1w_existing_clipped",
        "buyer_ratio_6w_existing_clipped",
        "nc_ratio_1w_vs_monthly_nc_clipped",
        "nc_ratio_6w_vs_monthly_nc_clipped",
        "units_per_customer_1w_clipped",
        "units_per_customer_6w_clipped",
    ]
    available_ratio_cols = [col for col in ratio_cols if col in launch_ratio_table.columns]
    if available_ratio_cols:
        df = df.merge(launch_ratio_table[available_ratio_cols], on="sku", how="left")

    seg = behavioral_segmentation or {}
    launch_segment_profile = seg.get("launch_segment_profile", pd.DataFrame())
    if seg.get("enabled") and not launch_segment_profile.empty:
        seg_wide = (
            launch_segment_profile.pivot_table(
                index="sku",
                columns="segment_key",
                values="launch_segment_share",
                aggfunc="mean",
                fill_value=0,
            )
            .add_prefix("segment_share_")
            .reset_index()
        )
        df = df.merge(seg_wide, on="sku", how="left")

    df["launch_text"] = (
        df["product_norm"].fillna("")
        + " "
        + df["product_need_area_norm"].fillna("")
        + " "
        + df["benefit_keywords_norm"].fillna("")
        + " "
        + df["flavour_norm"].fillna("")
        + " "
        + df["flavour_group_norm"].fillna("")
        + " "
        + df["product_form_norm"].fillna("")
        + " "
        + df["launch_strategy_type"].fillna("standard")
    ).str.strip()

    return df


def prepare_ml_features(train_df):
    categorical_cols = [
        "product_need_area_norm",
        "flavour_group_norm",
        "product_form_norm",
        "launch_strategy_type",
    ]

    numeric_cols = [
        "uvp",
        "first_order_quantity",
        "launch_month",
        "launch_during_sale",
        "first_week_sale_days",
        "first_6_week_sale_days",
        "eligible_customers_before_launch",
        "buyer_ratio_1w_existing_clipped",
        "buyer_ratio_6w_existing_clipped",
        "nc_ratio_1w_vs_monthly_nc_clipped",
        "nc_ratio_6w_vs_monthly_nc_clipped",
        "units_per_customer_1w_clipped",
        "units_per_customer_6w_clipped",
    ]

    segment_cols = [col for col in train_df.columns if col.startswith("segment_share_")]
    numeric_cols = [col for col in numeric_cols if col in train_df.columns] + segment_cols
    categorical_cols = [col for col in categorical_cols if col in train_df.columns]

    feature_df = train_df[categorical_cols + numeric_cols + ["launch_text"] + TARGET_METRICS].copy()
    feature_df = feature_df.dropna(subset=TARGET_METRICS).reset_index(drop=True)

    if feature_df.empty:
        return None

    for col in numeric_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    for col in categorical_cols:
        feature_df[col] = feature_df[col].fillna("unknown").astype(str)

    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=100)
    x_text = text_vectorizer.fit_transform(feature_df["launch_text"].fillna(""))

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_cat = encoder.fit_transform(feature_df[categorical_cols]) if categorical_cols else sparse.csr_matrix((len(feature_df), 0))

    scaler = StandardScaler()
    x_num = scaler.fit_transform(feature_df[numeric_cols]) if numeric_cols else np.zeros((len(feature_df), 0))
    x_num = sparse.csr_matrix(x_num)

    x = sparse.hstack([x_num, x_cat, x_text], format="csr")
    y = feature_df[TARGET_METRICS].astype(float)

    preprocessors = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_column": "launch_text",
        "numeric_defaults": {col: float(feature_df[col].median()) for col in numeric_cols},
        "categorical_defaults": {col: "unknown" for col in categorical_cols},
        "scaler": scaler,
        "encoder": encoder,
        "text_vectorizer": text_vectorizer,
    }

    return x, y, feature_df, preprocessors


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true, y_pred, model_name):
    rows = []
    y_pred = pd.DataFrame(y_pred, columns=TARGET_METRICS, index=y_true.index)

    for metric in TARGET_METRICS:
        actual = y_true[metric].values
        pred = y_pred[metric].values
        rows.append(
            {
                "model": model_name,
                "target_metric": metric,
                "mae": float(mean_absolute_error(actual, pred)),
                "rmse": rmse(actual, pred),
                "r2": float(r2_score(actual, pred)) if len(actual) > 1 else np.nan,
                "smape": float(np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-9))),
            }
        )
    return rows


def candidate_models():
    models = {
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
            )
        )

    if LGBMRegressor is not None:
        models["lightgbm"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
        )

    if CatBoostRegressor is not None:
        models["catboost"] = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=4,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
            )
        )

    return models


def train_quantile_models(x_train, y_train):
    quantiles = [0.1, 0.5, 0.9]
    quantile_models = {}

    for metric in TARGET_METRICS:
        quantile_models[metric] = {}
        for q in quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
            model.fit(x_train.toarray() if sparse.issparse(x_train) else x_train, y_train[metric])
            quantile_models[metric][f"q{int(q * 100)}"] = model

    return quantile_models


def evaluate_quantile_coverage(quantile_models, x_test, y_test):
    if y_test.empty:
        return pd.DataFrame()

    x_eval = x_test.toarray() if sparse.issparse(x_test) else x_test
    rows = []

    for metric in TARGET_METRICS:
        lower = quantile_models[metric]["q10"].predict(x_eval)
        median = quantile_models[metric]["q50"].predict(x_eval)
        upper = quantile_models[metric]["q90"].predict(x_eval)
        actual = y_test[metric].values

        rows.append(
            {
                "target_metric": metric,
                "interval": "q10_q90",
                "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                "avg_interval_width": float(np.mean(upper - lower)),
                "median_mae": float(mean_absolute_error(actual, median)),
            }
        )

    return pd.DataFrame(rows)


def train_supervised_ml_models(launches, launch_ratio_table, behavioral_segmentation):
    log("Training supervised ML model comparison layer...")
    ml_table = build_launch_ml_table(launches, launch_ratio_table, behavioral_segmentation)
    prepared = prepare_ml_features(ml_table)

    if prepared is None:
        return {
            "enabled": False,
            "reason": "No valid ML training rows after cleaning target metrics",
        }

    x, y, feature_df, preprocessors = prepared

    if len(feature_df) < 12:
        return {
            "enabled": False,
            "reason": "Insufficient launch rows for stable supervised ML training",
            "training_rows": int(len(feature_df)),
            "ml_training_table": ml_table,
        }

    test_size = 0.25 if len(feature_df) >= 20 else 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=RANDOM_STATE)

    fitted_models = {}
    metric_rows = []

    for name, model in candidate_models().items():
        log(f"Training model: {name}")
        try:
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            fitted_models[name] = model
            metric_rows.extend(evaluate_predictions(y_test, preds, name))
        except Exception as exc:
            log(f"Skipped {name}: {exc}")

    metrics_df = pd.DataFrame(metric_rows)

    if metrics_df.empty:
        return {
            "enabled": False,
            "reason": "All supervised ML models failed",
            "training_rows": int(len(feature_df)),
            "ml_training_table": ml_table,
        }

    model_summary = (
        metrics_df.groupby("model")
        .agg(
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_smape=("smape", "mean"),
            avg_r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(["avg_smape", "avg_mae"], ascending=[True, True])
    )

    best_model_name = model_summary.iloc[0]["model"]

    log("Training quantile interval models...")
    quantile_models = train_quantile_models(x_train, y_train)
    quantile_metrics = evaluate_quantile_coverage(quantile_models, x_test, y_test)

    return {
        "enabled": True,
        "reason": "ok",
        "training_rows": int(len(feature_df)),
        "target_metrics": TARGET_METRICS,
        "preprocessors": preprocessors,
        "models": fitted_models,
        "best_model_name": str(best_model_name),
        "model_metrics": metrics_df,
        "model_summary": model_summary,
        "quantile_models": quantile_models,
        "quantile_metrics": quantile_metrics,
        "ml_training_table": ml_table,
    }


# ============================================================
# EXPORT + MAIN
# ============================================================

def save_csv_exports(launch_ratio_table, monthly_new_customers, supervised_ml):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    launch_ratio_path = os.path.join(ARTIFACT_DIR, "launch_ratio_table_v2.csv")
    monthly_nc_path = os.path.join(ARTIFACT_DIR, "monthly_new_customers_v2.csv")
    ml_metrics_path = os.path.join(ARTIFACT_DIR, "ml_model_metrics_v2.csv")
    ml_summary_path = os.path.join(ARTIFACT_DIR, "ml_model_summary_v2.csv")
    quantile_metrics_path = os.path.join(ARTIFACT_DIR, "quantile_metrics_v2.csv")

    launch_ratio_table.to_csv(launch_ratio_path, index=False)
    monthly_new_customers.to_csv(monthly_nc_path, index=False)

    if supervised_ml.get("enabled"):
        supervised_ml["model_metrics"].to_csv(ml_metrics_path, index=False)
        supervised_ml["model_summary"].to_csv(ml_summary_path, index=False)
        supervised_ml["quantile_metrics"].to_csv(quantile_metrics_path, index=False)

    return {
        "launch_ratio_table": launch_ratio_path,
        "monthly_new_customers": monthly_nc_path,
        "ml_model_metrics": ml_metrics_path if supervised_ml.get("enabled") else None,
        "ml_model_summary": ml_summary_path if supervised_ml.get("enabled") else None,
        "quantile_metrics": quantile_metrics_path if supervised_ml.get("enabled") else None,
    }


def build_artifacts():
    orders_raw, sale_times_raw, launches_raw = load_raw_data()

    orders = clean_orders(orders_raw)
    sale_times = clean_sale_times(sale_times_raw)
    launches = clean_launches(launches_raw)

    orders = add_order_sale_flags(orders, sale_times)
    launches = add_launch_sale_features(launches, sale_times)

    mapping_df = ensure_target_group_mapping_file(launches)
    launches = apply_target_group_mapping(launches, mapping_df)

    monthly_company_scale = build_monthly_company_scale(orders)
    monthly_new_customers = build_monthly_new_customers(orders)
    launch_ratio_table = build_historical_buyer_ratio_table(orders, launches, monthly_new_customers)

    behavioral_segmentation = build_behavioral_segmentation(orders, launches, n_clusters=5)
    target_group_inference = build_target_group_model(launches, orders, top_n=TARGET_GROUP_TOP_N)
    semantic_similarity = build_semantic_similarity_assets(launches)
    supervised_ml = train_supervised_ml_models(launches, launch_ratio_table, behavioral_segmentation)

    artifacts = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "version": "v2_refactored_ml",
            "orders_rows": int(len(orders)),
            "launch_count": int(len(launches)),
            "launch_ratio_rows": int(len(launch_ratio_table)),
            "sale_period_count": int(len(sale_times)),
            "training_start_date": str(orders["date"].min().date()),
            "training_end_date": str(orders["date"].max().date()),
        },
        "data": {
            "orders": orders,
            "sale_times": sale_times,
            "launches": launches,
            "monthly_company_scale": monthly_company_scale,
            "monthly_new_customers": monthly_new_customers,
            "launch_ratio_table": launch_ratio_table,
        },
        "calibration": {
            "seasonality_index": build_seasonality_index(orders),
            "strategy_factors": build_strategy_factors(launches),
            "flavour_factors": build_group_factors(launches, "flavour_group_norm"),
            "product_form_factors": build_group_factors(launches, "product_form_norm"),
            "sale_factors": build_sale_factors(launches),
            "price_elasticity": build_price_elasticity(launches),
            "growth_context": build_growth_context(orders),
            "ratio_context": build_ratio_context(orders, monthly_new_customers),
            "target_metrics": TARGET_METRICS,
        },
        "target_group_inference": target_group_inference,
        "semantic_similarity": semantic_similarity,
        "behavioral_segmentation": behavioral_segmentation,
        "supervised_ml": supervised_ml,
    }

    export_paths = save_csv_exports(launch_ratio_table, monthly_new_customers, supervised_ml)
    artifacts["metadata"]["export_paths"] = export_paths

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    return artifacts


def print_summary(artifacts):
    supervised_ml = artifacts["supervised_ml"]
    segmentation = artifacts["behavioral_segmentation"]

    print("\n==========================================")
    print("Artifact saved successfully")
    print(f"Path: {ARTIFACT_PATH}")
    print("==========================================")

    print("\nMetadata:")
    print(artifacts["metadata"])

    print("\nBehavioral segmentation:")
    print(
        {
            "enabled": segmentation.get("enabled", False),
            "customers": segmentation.get("customer_count", 0),
            "clusters": segmentation.get("cluster_count", 0),
        }
    )

    print("\nSupervised ML:")
    print(
        {
            "enabled": supervised_ml.get("enabled", False),
            "reason": supervised_ml.get("reason"),
            "training_rows": supervised_ml.get("training_rows", 0),
            "best_model_name": supervised_ml.get("best_model_name"),
        }
    )

    if supervised_ml.get("enabled"):
        print("\nModel summary:")
        print(supervised_ml["model_summary"])

        print("\nQuantile interval metrics:")
        print(supervised_ml["quantile_metrics"])

    print("\nCSV exports:")
    for name, path in artifacts["metadata"].get("export_paths", {}).items():
        if path:
            print(f"{name}: {path}")


def main():
    artifacts = build_artifacts()
    print_summary(artifacts)


if __name__ == "__main__":
    main()
