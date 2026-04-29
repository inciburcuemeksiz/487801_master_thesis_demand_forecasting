import os
import re
import pickle
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

ORDERS_PATH = "data/raw/orders.csv"
SALE_TIMES_PATH = "data/raw/sale_times.csv"
LAUNCHES_PATH = "data/raw/launched_product_details.csv"
TARGET_GROUP_MAPPING_PATH = "data/raw/target_group_mapping.csv"

ARTIFACT_DIR = "artifacts"
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "model_artifacts_v2.pkl")

TARGET_GROUP_TOP_N = 8

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(x):
    """
    Normalize text for matching and grouping.
    """
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join([c for c in x if not unicodedata.combining(c)])
    x = re.sub(r"[^a-z0-9äöüß\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_numeric(x):
    """
    Converts values like:
    '39.90 €', '39,90 €', '8,000', '1 %', '-', '-%'
    into numeric float values.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x in ["", "-", "-%", "nan", "None"]:
        return np.nan

    x = x.replace("€", "")
    x = x.replace("%", "")
    x = x.strip()

    # Handle thousands format like 8,000 or 1,500
    if re.match(r"^\d{1,3}(,\d{3})+$", x):
        x = x.replace(",", "")
    else:
        # Handle decimal comma like 39,90
        x = x.replace(",", ".")

    x = re.sub(r"[^0-9.\-]", "", x)

    if x in ["", "-", "."]:
        return np.nan

    return float(x)


def normalize_strategy(x):
    """
    Normalize launch strategy labels.
    """
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


def normalize_keyword_field(value):
    if pd.isna(value):
        return ""

    tokens = [
        normalize_text(token)
        for token in str(value).split(",")
        if str(token).strip()
    ]

    tokens = [token for token in tokens if token]
    return ", ".join(tokens)


def safe_divide(a, b, default=1.0):
    """
    Safe division helper.
    """
    if b is None or pd.isna(b) or b == 0:
        return default
    if a is None or pd.isna(a):
        return default
    return a / b


def clip_factor(x, low=0.5, high=1.8):
    """
    Prevent unstable multipliers.
    """
    if pd.isna(x):
        return 1.0
    return float(np.clip(x, low, high))


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading data...")

    orders = pd.read_csv(ORDERS_PATH, low_memory=False)

    # sale_times.csv is semicolon-separated.
    sale_times = pd.read_csv(SALE_TIMES_PATH, sep=";", low_memory=False)

    # launched_product_details.csv is currently comma-separated.
    launches = pd.read_csv(LAUNCHES_PATH, low_memory=False)

    orders.columns = orders.columns.str.strip()
    sale_times.columns = sale_times.columns.str.strip()
    launches.columns = launches.columns.str.strip()

    print(f"orders: {orders.shape}")
    print(f"sale_times: {sale_times.shape}")
    print(f"launches: {launches.shape}")

    print("\nLaunch columns:")
    print(list(launches.columns))

    return orders, sale_times, launches


# ============================================================
# CLEAN DATA
# ============================================================

def clean_orders(orders):
    df = orders.copy()

    required_cols = [
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

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"orders.csv missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "first_order_date" in df.columns:
        df["first_order_date"] = pd.to_datetime(df["first_order_date"], errors="coerce")

    if "last_order_date" in df.columns:
        df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors="coerce")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["net_revenue"] = pd.to_numeric(df["net_revenue"], errors="coerce")

    if "months_since_first_order" in df.columns:
        df["months_since_first_order"] = pd.to_numeric(
            df["months_since_first_order"],
            errors="coerce",
        )

    if "nr_of_purchase" in df.columns:
        df["nr_of_purchase"] = pd.to_numeric(
            df["nr_of_purchase"],
            errors="coerce",
        )

    # Basic cleaning
    df = df[df["date"].notna()]
    df = df[df["customer_nr"].notna()]
    df = df[df["sku"].notna()]
    df = df[df["quantity"].fillna(0) > 0]

    # Remove likely non-product rows
    remove_pattern = "shipping|versand|discount|rabatt|gutschein|gift card|free article|gratis"
    text_cols = ["artikel_name", "product", "product_category"]

    combined_text = ""
    for col in text_cols:
        if col in df.columns:
            combined_text = combined_text + " " + df[col].astype(str)

    mask_remove = combined_text.str.lower().str.contains(
        remove_pattern,
        regex=True,
        na=False,
    )
    df = df[~mask_remove].copy()

    df["product_norm"] = df["product"].apply(normalize_text)
    df["flavour_norm"] = df["flavour"].apply(normalize_text)
    df["category_norm"] = df["product_category"].apply(normalize_text)
    df["artikel_name_norm"] = df["artikel_name"].apply(normalize_text)

    product_need_area_series = df["product_need_area"] if "product_need_area" in df.columns else pd.Series("", index=df.index)
    benefit_keywords_series = df["benefit_keywords"] if "benefit_keywords" in df.columns else pd.Series("", index=df.index)
    flavour_group_series = df["flavour_group"] if "flavour_group" in df.columns else pd.Series("", index=df.index)

    df["product_need_area_norm"] = product_need_area_series.apply(normalize_text)
    df["benefit_keywords_norm"] = benefit_keywords_series.apply(normalize_keyword_field)
    df["flavour_group_norm"] = flavour_group_series.apply(normalize_text)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    print(f"orders cleaned: {df.shape}")

    return df


def clean_sale_times(sale_times):
    df = sale_times.copy()

    required_cols = ["name", "start_d", "end_d"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"sale_times.csv missing columns: {missing}")

    df["start_d"] = pd.to_datetime(df["start_d"], errors="coerce")
    df["end_d"] = pd.to_datetime(df["end_d"], errors="coerce")

    df = df[df["start_d"].notna() & df["end_d"].notna()].copy()
    df["name_norm"] = df["name"].apply(normalize_text)

    print(f"sale_times cleaned: {df.shape}")

    return df


def clean_launches(launches):
    df = launches.copy()

    required_cols = [
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
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"launched_product_details.csv missing columns: {missing}")

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["uvp"] = df["uvp"].apply(clean_numeric)
    df["first_order_quantity"] = df["first_order_quantity"].apply(clean_numeric)

    for col in TARGET_METRICS:
        df[col] = df[col].apply(clean_numeric)

    if "first_week_quantity_target" in df.columns:
        df["first_week_quantity_target"] = df["first_week_quantity_target"].apply(clean_numeric)

    if "first_week_quantity_target_accuracy" in df.columns:
        df["first_week_quantity_target_accuracy"] = (
            df["first_week_quantity_target_accuracy"].apply(clean_numeric)
        )

    df = df[df["launch_date"].notna()].copy()

    df["sku"] = df["sku"].astype(str)
    df["launch_strategy_type"] = df["launch_strategy_type"].apply(normalize_strategy)

    df["product_norm"] = df["product"].apply(normalize_text)
    df["flavour_norm"] = df["flavour"].apply(normalize_text)
    df["product_form_norm"] = df["product_form"].apply(normalize_text)
    df["use_case_norm"] = df["Product Use Case / What it is about"].apply(normalize_text)
    df["target_group_norm"] = df["Target Group"].apply(normalize_text)
    df["artikel_name_norm"] = df["artikel_name"].apply(normalize_text)

    df["launch_month"] = df["launch_date"].dt.month
    df["launch_year"] = df["launch_date"].dt.year
    df["launch_year_month"] = df["launch_date"].dt.to_period("M").astype(str)

    df["units_per_customer_1w"] = (
        df["first_week_quantity"] / df["first_week_total_c"].replace(0, np.nan)
    )
    df["units_per_customer_6w"] = (
        df["first_6_week_quantity"] / df["first_6_week_total_c"].replace(0, np.nan)
    )
    df["first_week_share_of_6w"] = (
        df["first_week_quantity"] / df["first_6_week_quantity"].replace(0, np.nan)
    )
    df["new_customer_share_1w"] = (
        df["first_week_nc"] / df["first_week_total_c"].replace(0, np.nan)
    )
    df["new_customer_share_6w"] = (
        df["first_6_week_nc"] / df["first_6_week_total_c"].replace(0, np.nan)
    )

    print(f"launches cleaned: {df.shape}")
    print("launch strategy counts:")
    print(df["launch_strategy_type"].value_counts(dropna=False))

    return df


# ============================================================
# SALE FEATURES
# ============================================================

def add_order_sale_flags(orders, sale_times):
    df = orders.copy()
    df["sale_name"] = None
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
            sale_start = sale["start_d"]
            sale_end = sale["end_d"]

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
# CALIBRATION TABLES
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
    month_qty = (
        orders.groupby("month")["quantity"]
        .sum()
        .reindex(range(1, 13))
    )

    avg_month = month_qty.mean()

    seasonality = {}
    for month, qty in month_qty.items():
        seasonality[int(month)] = clip_factor(
            safe_divide(qty, avg_month, default=1.0),
            0.6,
            1.5,
        )

    return seasonality


def build_strategy_factors(launches):
    factors = {}

    standard = launches[launches["launch_strategy_type"] == "standard"]

    for strategy in ["standard", "co_creation", "limited_edition"]:
        strategy_df = launches[launches["launch_strategy_type"] == strategy]

        factors[strategy] = {}

        for metric in TARGET_METRICS:
            standard_mean = standard[metric].mean()
            strategy_mean = strategy_df[metric].mean()

            raw_factor = safe_divide(strategy_mean, standard_mean, default=1.0)
            factors[strategy][metric] = clip_factor(raw_factor, 0.5, 2.2)

    return factors


def build_flavour_factors(launches):
    """
    Flavour-group factor by metric:
    average metric for flavour group / overall average metric.
    """
    factors = {}

    group_col = "flavour_group_norm"

    if group_col not in launches.columns:
        group_col = "flavour_norm"

    for flavour_group, group in launches.groupby(group_col):
        if not flavour_group:
            continue

        factors[flavour_group] = {}

        for metric in TARGET_METRICS:
            overall_mean = launches[metric].mean()
            group_mean = group[metric].mean()
            raw_factor = safe_divide(group_mean, overall_mean, default=1.0)
            factors[flavour_group][metric] = clip_factor(raw_factor, 0.6, 1.7)

    return factors


def build_product_form_factors(launches):
    factors = {}

    for form, group in launches.groupby("product_form_norm"):
        if not form:
            continue

        factors[form] = {}

        for metric in TARGET_METRICS:
            overall_mean = launches[metric].mean()
            form_mean = group[metric].mean()
            raw_factor = safe_divide(form_mean, overall_mean, default=1.0)
            factors[form][metric] = clip_factor(raw_factor, 0.6, 1.7)

    return factors


def build_sale_factors(launches):
    """
    Simple launch sale overlap factor.
    Separate first-week and 6-week sale factors.
    """
    sale_factors = {}

    for metric in TARGET_METRICS:
        baseline_mean = launches[metric].mean()

        during_sale_mean = launches.loc[
            launches["launch_during_sale"] == 1,
            metric,
        ].mean()

        first_week_sale_mean = launches.loc[
            launches["first_week_sale_days"] > 0,
            metric,
        ].mean()

        first_6w_sale_mean = launches.loc[
            launches["first_6_week_sale_days"] > 0,
            metric,
        ].mean()

        sale_factors[metric] = {
            "launch_during_sale": clip_factor(
                safe_divide(during_sale_mean, baseline_mean, default=1.0),
                0.6,
                1.8,
            ),
            "first_week_sale_overlap": clip_factor(
                safe_divide(first_week_sale_mean, baseline_mean, default=1.0),
                0.6,
                1.8,
            ),
            "first_6_week_sale_overlap": clip_factor(
                safe_divide(first_6w_sale_mean, baseline_mean, default=1.0),
                0.6,
                1.8,
            ),
        }

    return sale_factors


def build_price_elasticity(launches):
    df = launches[["uvp", "first_6_week_quantity"]].dropna()
    df = df[(df["uvp"] > 0) & (df["first_6_week_quantity"] > 0)]

    if len(df) < 5:
        return -0.5

    df["log_price"] = np.log(df["uvp"])
    df["log_qty"] = np.log(df["first_6_week_quantity"])

    coef = np.polyfit(df["log_price"], df["log_qty"], 1)
    elasticity = float(coef[0])

    # Keep it stable and conservative.
    elasticity = float(np.clip(elasticity, -1.5, -0.1))
    return elasticity


def build_growth_context(orders):
    """
    Recent company scale vs all-history monthly average.
    """
    monthly = build_monthly_company_scale(orders).sort_values("date")

    if monthly.empty:
        return {
            "recent_3m_avg_quantity": 1.0,
            "historical_avg_quantity": 1.0,
            "company_growth_factor": 1.0,
        }

    recent = monthly.tail(3)
    recent_avg = recent["monthly_quantity"].mean()
    hist_avg = monthly["monthly_quantity"].mean()

    growth_factor = clip_factor(
        safe_divide(recent_avg, hist_avg, default=1.0),
        0.7,
        1.8,
    )

    return {
        "recent_3m_avg_quantity": float(recent_avg),
        "historical_avg_quantity": float(hist_avg),
        "company_growth_factor": float(growth_factor),
    }


# ============================================================
# RATIO-BASED FORECAST TABLES
# ============================================================

def build_monthly_new_customers(orders):
    """
    Builds monthly new-customer scale from first_order_date.
    One customer is counted once in the month of their first order.
    """
    if "first_order_date" not in orders.columns:
        return pd.DataFrame(columns=["first_order_month", "new_customers", "date"])

    customer_first_orders = (
        orders[["customer_nr", "first_order_date"]]
        .dropna(subset=["customer_nr", "first_order_date"])
        .drop_duplicates(subset=["customer_nr"])
        .copy()
    )

    if customer_first_orders.empty:
        return pd.DataFrame(columns=["first_order_month", "new_customers", "date"])

    customer_first_orders["first_order_month"] = (
        customer_first_orders["first_order_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_new_customers = (
        customer_first_orders
        .groupby("first_order_month")["customer_nr"]
        .nunique()
        .reset_index(name="new_customers")
    )

    monthly_new_customers["date"] = pd.to_datetime(
        monthly_new_customers["first_order_month"] + "-01"
    )

    monthly_new_customers = (
        monthly_new_customers
        .sort_values("date")
        .reset_index(drop=True)
    )

    return monthly_new_customers


def build_historical_buyer_ratio_table(orders, launches, monthly_new_customers):
    """
    Builds normalized launch-level ratio table.

    Main outputs:
    - existing-customer penetration before launch
    - NC strength relative to monthly new-customer scale
    - units per customer
    """
    order_df = orders.copy()
    order_df["sku_str"] = order_df["sku"].astype(str)

    rows = []

    for _, launch in launches.iterrows():
        launch_sku = str(launch["sku"])
        launch_date = launch["launch_date"]

        if pd.isna(launch_date):
            continue

        first_week_end = launch_date + pd.Timedelta(days=6)
        first_6w_end = launch_date + pd.Timedelta(days=41)

        eligible_customers = set(
            order_df.loc[
                order_df["date"] < launch_date,
                "customer_nr",
            ]
            .dropna()
            .astype(str)
        )

        eligible_count = len(eligible_customers)

        buyers_1w = set(
            order_df.loc[
                (order_df["date"] >= launch_date)
                & (order_df["date"] <= first_week_end)
                & (order_df["sku_str"] == launch_sku),
                "customer_nr",
            ]
            .dropna()
            .astype(str)
        )

        buyers_6w = set(
            order_df.loc[
                (order_df["date"] >= launch_date)
                & (order_df["date"] <= first_6w_end)
                & (order_df["sku_str"] == launch_sku),
                "customer_nr",
            ]
            .dropna()
            .astype(str)
        )

        buyers_1w_count = len(buyers_1w)
        buyers_6w_count = len(buyers_6w)

        buyer_ratio_1w = (
            buyers_1w_count / eligible_count
            if eligible_count > 0
            else np.nan
        )

        buyer_ratio_6w = (
            buyers_6w_count / eligible_count
            if eligible_count > 0
            else np.nan
        )

        first_week_total_c = launch.get("first_week_total_c", np.nan)
        first_6_week_total_c = launch.get("first_6_week_total_c", np.nan)
        first_week_nc = launch.get("first_week_nc", np.nan)
        first_6_week_nc = launch.get("first_6_week_nc", np.nan)
        first_week_quantity = launch.get("first_week_quantity", np.nan)
        first_6_week_quantity = launch.get("first_6_week_quantity", np.nan)

        nc_share_1w = safe_divide(
            first_week_nc,
            first_week_total_c,
            default=np.nan,
        )

        nc_share_6w = safe_divide(
            first_6_week_nc,
            first_6_week_total_c,
            default=np.nan,
        )

        units_per_customer_1w = safe_divide(
            first_week_quantity,
            first_week_total_c,
            default=np.nan,
        )

        units_per_customer_6w = safe_divide(
            first_6_week_quantity,
            first_6_week_total_c,
            default=np.nan,
        )

        launch_year_month = launch_date.to_period("M").strftime("%Y-%m")

        monthly_match = monthly_new_customers[
            monthly_new_customers["first_order_month"] == launch_year_month
        ]

        if monthly_match.empty:
            monthly_new_customers_at_launch = np.nan
        else:
            monthly_new_customers_at_launch = float(
                monthly_match["new_customers"].iloc[0]
            )

        nc_ratio_1w_vs_monthly_nc = safe_divide(
            first_week_nc,
            monthly_new_customers_at_launch,
            default=np.nan,
        )

        nc_ratio_6w_vs_monthly_nc = safe_divide(
            first_6_week_nc,
            monthly_new_customers_at_launch,
            default=np.nan,
        )

        rows.append(
            {
                "sku": launch_sku,
                "product": launch.get("product", ""),
                "flavour": launch.get("flavour", ""),
                "product_form": launch.get("product_form", ""),
                "launch_strategy_type": launch.get("launch_strategy_type", ""),
                "launch_date": launch_date,
                "launch_month": launch.get("launch_month", np.nan),
                "launch_year_month": launch_year_month,
                "uvp": launch.get("uvp", np.nan),

                "eligible_customers_before_launch": eligible_count,

                "buyers_1w_existing": buyers_1w_count,
                "buyers_6w_existing": buyers_6w_count,

                "buyer_ratio_1w_existing": buyer_ratio_1w,
                "buyer_ratio_6w_existing": buyer_ratio_6w,

                "first_week_total_c": first_week_total_c,
                "first_6_week_total_c": first_6_week_total_c,
                "first_week_nc": first_week_nc,
                "first_6_week_nc": first_6_week_nc,

                "nc_share_1w": nc_share_1w,
                "nc_share_6w": nc_share_6w,

                "monthly_new_customers_at_launch": monthly_new_customers_at_launch,
                "nc_ratio_1w_vs_monthly_nc": nc_ratio_1w_vs_monthly_nc,
                "nc_ratio_6w_vs_monthly_nc": nc_ratio_6w_vs_monthly_nc,

                "units_per_customer_1w": units_per_customer_1w,
                "units_per_customer_6w": units_per_customer_6w,

                "first_week_quantity": first_week_quantity,
                "first_6_week_quantity": first_6_week_quantity,
            }
        )

    ratio_df = pd.DataFrame(rows)

    if ratio_df.empty:
        return ratio_df

    ratio_df["flag_nc_ratio_too_high"] = (
        ratio_df["nc_ratio_6w_vs_monthly_nc"] > 0.5
    )

    ratio_df["flag_nc_6w_gt_monthly_nc"] = (
        ratio_df["first_6_week_nc"] > ratio_df["monthly_new_customers_at_launch"]
    )

    ratio_df["flag_nc_gt_total_customers_1w"] = (
        ratio_df["first_week_nc"] > ratio_df["first_week_total_c"]
    )

    ratio_df["flag_nc_gt_total_customers_6w"] = (
        ratio_df["first_6_week_nc"] > ratio_df["first_6_week_total_c"]
    )

    return ratio_df


def add_clipped_ratio_columns(launch_ratio_table):
    """
    Winsorizes key ratio columns at 5th and 95th percentiles.
    Keeps all launch rows but prevents extreme launches from dominating forecasts.
    """
    df = launch_ratio_table.copy()

    clip_cols = [
        "buyer_ratio_1w_existing",
        "buyer_ratio_6w_existing",
        "nc_ratio_1w_vs_monthly_nc",
        "nc_ratio_6w_vs_monthly_nc",
        "units_per_customer_1w",
        "units_per_customer_6w",
    ]

    for col in clip_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        if series.notna().sum() < 5:
            df[f"{col}_clipped"] = series
            continue

        lower = series.quantile(0.05)
        upper = series.quantile(0.95)

        df[f"{col}_clipped"] = series.clip(lower=lower, upper=upper)

    return df


def build_ratio_context(orders, monthly_new_customers):
    """
    Builds current customer-base and recent NC scale context for the app.
    """
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

    known_customer_count = int(orders["customer_nr"].nunique())

    active_12m_start = max_date - pd.DateOffset(months=12)
    active_6m_start = max_date - pd.DateOffset(months=6)

    active_customer_count_12m = int(
        orders.loc[orders["date"] >= active_12m_start, "customer_nr"].nunique()
    )

    active_customer_count_6m = int(
        orders.loc[orders["date"] >= active_6m_start, "customer_nr"].nunique()
    )

    if monthly_new_customers.empty:
        recent_3m_avg_new_customers = 1.0
        historical_avg_new_customers = 1.0
    else:
        recent_3m_avg_new_customers = float(
            monthly_new_customers.tail(3)["new_customers"].mean()
        )
        historical_avg_new_customers = float(
            monthly_new_customers["new_customers"].mean()
        )

    new_customer_growth_factor = clip_factor(
        safe_divide(
            recent_3m_avg_new_customers,
            historical_avg_new_customers,
            default=1.0,
        ),
        0.5,
        2.5,
    )

    return {
        "latest_order_date": str(max_date.date()),
        "known_customer_count": known_customer_count,
        "active_customer_count_12m": active_customer_count_12m,
        "active_customer_count_6m": active_customer_count_6m,
        "recent_3m_avg_new_customers": recent_3m_avg_new_customers,
        "historical_avg_new_customers": historical_avg_new_customers,
        "new_customer_growth_factor": new_customer_growth_factor,
    }


# ============================================================
# TARGET GROUP INFERENCE
# ============================================================

def ensure_target_group_mapping_file(launches):
    """
    Creates a starter raw->canonical mapping file if it does not exist.
    """
    if os.path.exists(TARGET_GROUP_MAPPING_PATH):
        mapping_df = pd.read_csv(TARGET_GROUP_MAPPING_PATH)

        required_cols = ["raw_target_group", "canonical_target_group"]
        missing = [c for c in required_cols if c not in mapping_df.columns]
        if missing:
            raise ValueError(
                f"target_group_mapping.csv missing columns: {missing}"
            )

        mapping_df = mapping_df.copy()
        mapping_df["raw_target_group"] = mapping_df["raw_target_group"].astype(str).str.strip()
        mapping_df["canonical_target_group"] = (
            mapping_df["canonical_target_group"].astype(str).str.strip()
        )
        mapping_df = mapping_df[
            mapping_df["raw_target_group"].astype(bool)
            & mapping_df["canonical_target_group"].astype(bool)
        ].drop_duplicates()
        return mapping_df

    raw_values = sorted(
        {
            str(x).strip()
            for x in launches["Target Group"].dropna().tolist()
            if str(x).strip()
        }
    )

    mapping_df = pd.DataFrame(
        {
            "raw_target_group": raw_values,
            "canonical_target_group": raw_values,
        }
    )

    os.makedirs(os.path.dirname(TARGET_GROUP_MAPPING_PATH), exist_ok=True)
    mapping_df.to_csv(TARGET_GROUP_MAPPING_PATH, index=False)

    print(
        f"Created starter target-group mapping file: {TARGET_GROUP_MAPPING_PATH}"
    )
    return mapping_df


def apply_target_group_mapping(launches, mapping_df):
    df = launches.copy()

    mapping = {
        normalize_text(raw): canonical
        for raw, canonical in mapping_df[["raw_target_group", "canonical_target_group"]].values
        if str(raw).strip() and str(canonical).strip()
    }

    df["raw_target_group"] = df["Target Group"].fillna("").astype(str).str.strip()
    df["raw_target_group_norm"] = df["raw_target_group"].apply(normalize_text)

    def _map_target_group(raw, raw_norm):
        if raw_norm in mapping:
            return mapping[raw_norm]
        if raw:
            return raw
        return "OTHER"

    df["canonical_target_group"] = df.apply(
        lambda row: _map_target_group(
            row["raw_target_group"],
            row["raw_target_group_norm"],
        ),
        axis=1,
    )

    df["canonical_target_group"] = df["canonical_target_group"].astype(str).str.strip()
    df["canonical_target_group"] = df["canonical_target_group"].replace("", "OTHER")

    return df


def build_target_group_training_table(launches, orders, top_n=8):
    """
    Build product-level supervised training data by joining launch labels with
    aggregated order behavior on SKU.
    """
    order_df = orders.copy()

    order_df["is_new_customer"] = (
        order_df["customer_status"].astype(str).str.upper().str.contains("NEW")
    ).astype(int)

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

    train_df = launches[
        [
            "sku",
            "product_norm",
            "use_case_norm",
            "flavour_norm",
            "launch_strategy_type",
            "uvp",
            "canonical_target_group",
        ]
    ].copy()

    train_df = train_df.merge(order_agg, on="sku", how="left")

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

    train_df["target_group_label"] = np.where(
        train_df["canonical_target_group"].isin(top_classes),
        train_df["canonical_target_group"],
        "OTHER",
    )

    return train_df, numeric_cols, class_counts.to_dict(), top_classes


def train_target_group_inference_model(train_df, numeric_cols):
    y = train_df["target_group_label"].astype(str)
    class_count = y.nunique()

    if len(train_df) < 10 or class_count < 2:
        return {
            "enabled": False,
            "reason": "Insufficient labeled data to train target-group model",
            "training_rows": int(len(train_df)),
            "classes": sorted(y.unique().tolist()),
        }

    text_series = train_df["feature_text"].fillna("")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_text = vectorizer.fit_transform(text_series)

    x_num = sparse.csr_matrix(train_df[numeric_cols].astype(float).values)
    x_train = sparse.hstack([x_text, x_num], format="csr")

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
    )
    model.fit(x_train, y)

    coverage_counts = y.value_counts()
    total = max(int(len(y)), 1)

    coverage_by_group = {
        label: {
            "count": int(count),
            "share": float(count / total),
        }
        for label, count in coverage_counts.items()
    }

    numeric_defaults = {
        col: float(train_df[col].mean()) if col in train_df.columns else 0.0
        for col in numeric_cols
    }

    return {
        "enabled": True,
        "reason": "ok",
        "model": model,
        "vectorizer": vectorizer,
        "numeric_feature_columns": numeric_cols,
        "numeric_feature_defaults": numeric_defaults,
        "coverage_by_group": coverage_by_group,
        "training_rows": int(len(train_df)),
        "classes": model.classes_.tolist(),
    }


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def build_semantic_similarity_assets(launches):
    """
    Build TF-IDF assets for semantic launch similarity scoring.
    """
    df = launches.copy()

    df["semantic_text"] = (
        df["product_norm"].fillna("")
        + " "
        + df["use_case_norm"].fillna("")
        + " "
        + df["target_group_norm"].fillna("")
        + " "
        + df["flavour_norm"].fillna("")
        + " "
        + df["product_form_norm"].fillna("")
        + " "
        + df["launch_strategy_type"].fillna("standard")
    ).str.strip()

    if len(df) < 3:
        return {
            "enabled": False,
            "reason": "Insufficient launch rows for semantic similarity",
            "row_count": int(len(df)),
        }

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
    """
    Adds business-friendly labels to KMeans segments based on segment behavior.
    Labels are rule-based, so they remain meaningful even if SEG_ numbers change.
    """
    df = seg_summary.copy()

    if df.empty:
        df["segment_label"] = ""
        df["segment_description"] = ""
        return df

    # Rank helpers
    df["rank_monetary_desc"] = df["avg_monetary"].rank(ascending=False, method="first")
    df["rank_frequency_desc"] = df["avg_frequency"].rank(ascending=False, method="first")
    df["rank_sale_desc"] = df["avg_sale_share"].rank(ascending=False, method="first")
    df["rank_recency_desc"] = df["avg_recency_days"].rank(ascending=False, method="first")
    df["rank_launch_share_desc"] = df["avg_launch_share_24m"].rank(ascending=False, method="first")
    df["rank_diversity_desc"] = df["avg_product_diversity_ratio_24m"].rank(ascending=False, method="first")

    labels = {}
    descriptions = {}

    assigned = set()

    def assign(seg_key, label, description):
        if seg_key not in assigned:
            labels[seg_key] = label
            descriptions[seg_key] = description
            assigned.add(seg_key)

    # 1) Loyal high-value: high monetary + high frequency
    loyal_candidates = df.sort_values(
        ["avg_monetary", "avg_frequency"],
        ascending=[False, False],
    )
    if not loyal_candidates.empty:
        row = loyal_candidates.iloc[0]
        assign(
            row["segment_key"],
            "Loyal high-value buyers",
            "Small or mid-sized segment with high order frequency and high total spend.",
        )

    # 2) Launch adopters: high launch share / many launch purchases
    launch_candidates = df[~df["segment_key"].isin(assigned)].sort_values(
        ["avg_launch_share_24m", "avg_unique_launch_skus_24m"],
        ascending=[False, False],
    )
    if not launch_candidates.empty:
        row = launch_candidates.iloc[0]
        if row["avg_launch_share_24m"] > 0:
            assign(
                row["segment_key"],
                "Launch adopters",
                "Customers with above-average tendency to buy newly launched SKUs.",
            )

    # 3) Sale-sensitive: highest sale share
    sale_candidates = df[~df["segment_key"].isin(assigned)].sort_values(
        "avg_sale_share",
        ascending=False,
    )
    if not sale_candidates.empty:
        row = sale_candidates.iloc[0]
        if row["avg_sale_share"] >= 0.25:
            assign(
                row["segment_key"],
                "Sale-sensitive buyers",
                "Customers whose purchases are strongly concentrated in sale or campaign periods.",
            )

    # 4) Variety seekers: high product/flavour diversity
    variety_candidates = df[~df["segment_key"].isin(assigned)].sort_values(
        ["avg_product_diversity_ratio_24m", "avg_unique_product_count_24m", "avg_unique_flavour_count_24m"],
        ascending=[False, False, False],
    )
    if not variety_candidates.empty:
        row = variety_candidates.iloc[0]
        assign(
            row["segment_key"],
            "Variety seekers",
            "Customers who try a wider mix of products, categories, or flavours.",
        )

    # 5) Dormant low-value: old recency + low monetary/frequency
    dormant_candidates = df[~df["segment_key"].isin(assigned)].sort_values(
        ["avg_recency_days", "avg_monetary", "avg_frequency"],
        ascending=[False, True, True],
    )
    if not dormant_candidates.empty:
        row = dormant_candidates.iloc[0]
        assign(
            row["segment_key"],
            "Dormant low-value customers",
            "Customers with old last purchase dates, low frequency, and low total spend.",
        )

    # Fallback for any unassigned segment
    for _, row in df.iterrows():
        seg_key = row["segment_key"]
        if seg_key not in assigned:
            assign(
                seg_key,
                "Broad occasional customers",
                "Large general segment with occasional purchasing behavior.",
            )

    df["segment_label"] = df["segment_key"].map(labels)
    df["segment_description"] = df["segment_key"].map(descriptions)

    drop_cols = [
        "rank_monetary_desc",
        "rank_frequency_desc",
        "rank_sale_desc",
        "rank_recency_desc",
        "rank_launch_share_desc",
        "rank_diversity_desc",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def log_progress(message):
    print(message, flush=True)


def build_behavioral_segmentation(orders, launches, n_clusters=5, fit_sample_size=50000):
    """
    Build customer behavioral segments and segment affinity to launches.

    New version:
    - Uses classic RFM/sale behavior
    - Adds last-24-month launch adoption and product diversity features
    - Creates 5 behavior segments
    - Adds business-friendly segment labels
    """
    log_progress("[behavioral_segmentation] Building customer features...")
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

    if not required_cols.issubset(set(orders.columns)):
        missing = sorted(required_cols - set(orders.columns))
        return {
            "enabled": False,
            "reason": f"orders data does not contain required columns: {missing}",
        }

    customer_df = orders.copy()
    customer_df = customer_df[customer_df["customer_nr"].notna()].copy()

    if customer_df.empty:
        return {
            "enabled": False,
            "reason": "No customer records found",
        }

    customer_df["sku_str"] = customer_df["sku"].astype(str)

    launch_skus = set(launches["sku"].astype(str))

    limited_skus = set(
        launches.loc[
            launches["launch_strategy_type"] == "limited_edition",
            "sku",
        ].astype(str)
    )

    co_creation_skus = set(
        launches.loc[
            launches["launch_strategy_type"] == "co_creation",
            "sku",
        ].astype(str)
    )

    max_date = customer_df["date"].max()

    customer_df["is_launch_sku"] = customer_df["sku_str"].isin(launch_skus).astype(int)
    customer_df["is_limited_edition_sku"] = customer_df["sku_str"].isin(limited_skus).astype(int)
    customer_df["is_co_creation_sku"] = customer_df["sku_str"].isin(co_creation_skus).astype(int)

    # ------------------------------------------------------------
    # Classic RFM / sale features
    # ------------------------------------------------------------
    agg = (
        customer_df.groupby("customer_nr")
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

    agg["recency_days"] = (max_date - agg["last_order_date"]).dt.days
    agg["recency_days"] = agg["recency_days"].fillna(agg["recency_days"].median())

    # ------------------------------------------------------------
    # Last 24-month launch/diversity behavior
    # ------------------------------------------------------------
    cutoff_24m = max_date - pd.DateOffset(months=24)
    recent_orders = customer_df[customer_df["date"] >= cutoff_24m].copy()

    if recent_orders.empty:
        launch_behavior = pd.DataFrame({"customer_nr": agg["customer_nr"]})
    else:
        launch_behavior = (
            recent_orders.groupby("customer_nr")
            .agg(
                order_lines_24m=("sku_str", "count"),
                quantity_24m=("quantity", "sum"),
                unique_product_count_24m=("product", "nunique"),
                unique_category_count_24m=("product_category", "nunique"),
                unique_flavour_count_24m=("flavour", "nunique"),
                launch_purchase_count_24m=("is_launch_sku", "sum"),
                limited_edition_purchase_count_24m=("is_limited_edition_sku", "sum"),
                co_creation_purchase_count_24m=("is_co_creation_sku", "sum"),
            )
            .reset_index()
        )

        launch_unique = (
            recent_orders[recent_orders["is_launch_sku"] == 1]
            .groupby("customer_nr")["sku_str"]
            .nunique()
            .rename("unique_launch_skus_24m")
            .reset_index()
        )

        launch_behavior = launch_behavior.merge(
            launch_unique,
            on="customer_nr",
            how="left",
        )

    behavior_defaults = {
        "order_lines_24m": 0,
        "quantity_24m": 0,
        "unique_product_count_24m": 0,
        "unique_category_count_24m": 0,
        "unique_flavour_count_24m": 0,
        "launch_purchase_count_24m": 0,
        "unique_launch_skus_24m": 0,
        "limited_edition_purchase_count_24m": 0,
        "co_creation_purchase_count_24m": 0,
    }

    for col, default in behavior_defaults.items():
        if col not in launch_behavior.columns:
            launch_behavior[col] = default

    agg = agg.merge(launch_behavior, on="customer_nr", how="left")

    for col, default in behavior_defaults.items():
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(default)

    agg["launch_share_24m"] = (
        agg["launch_purchase_count_24m"]
        / agg["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)

    agg["product_diversity_ratio_24m"] = (
        agg["unique_product_count_24m"]
        / agg["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)

    agg["flavour_diversity_ratio_24m"] = (
        agg["unique_flavour_count_24m"]
        / agg["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)

    agg["category_diversity_ratio_24m"] = (
        agg["unique_category_count_24m"]
        / agg["order_lines_24m"].replace(0, np.nan)
    ).fillna(0)

    log_progress(f"[behavioral_segmentation] Customer features built for {len(agg):,} customers.")

    # ------------------------------------------------------------
    # Features used by clustering
    # ------------------------------------------------------------
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
        agg[col] = pd.to_numeric(agg[col], errors="coerce")
        agg[col] = agg[col].replace([np.inf, -np.inf], np.nan)
        agg[col] = agg[col].fillna(agg[col].median())

    if len(agg) < 20:
        return {
            "enabled": False,
            "reason": "Not enough customers for stable segmentation",
            "customer_count": int(len(agg)),
        }

    cluster_count = min(n_clusters, max(2, len(agg) // 25))

    sample_size = min(int(fit_sample_size), int(len(agg)))
    fit_df = agg.sample(n=sample_size, random_state=42) if sample_size < len(agg) else agg

    log_progress(
        f"[behavioral_segmentation] Fitting scaler on {len(fit_df):,} sampled customers..."
    )

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(fit_df[feature_cols])

    log_progress("[behavioral_segmentation] Scaler fitted.")

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=20,
        batch_size=2048,
    )
    log_progress("[behavioral_segmentation] Fitting MiniBatchKMeans...")
    model.fit(x_fit_scaled)
    log_progress("[behavioral_segmentation] MiniBatchKMeans fitted.")

    log_progress(f"[behavioral_segmentation] Assigning segments for {len(agg):,} customers...")
    x_all_scaled = scaler.transform(agg[feature_cols])
    agg["segment_id"] = model.predict(x_all_scaled).astype(int)
    agg["segment_key"] = agg["segment_id"].apply(lambda x: f"SEG_{int(x)}")
    log_progress("[behavioral_segmentation] Segments assigned.")

    # ------------------------------------------------------------
    # Segment summary
    # ------------------------------------------------------------
    log_progress("[behavioral_segmentation] Building segment summary...")
    seg_summary = (
        agg.groupby("segment_key")
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

    seg_summary["global_share"] = (
        seg_summary["customer_count"] / max(seg_summary["customer_count"].sum(), 1)
    )

    seg_summary = label_behavioral_segments(seg_summary)
    log_progress("[behavioral_segmentation] Segment summary built.")

    # ------------------------------------------------------------
    # Launch-window segment affinity
    # ------------------------------------------------------------
    log_progress("[behavioral_segmentation] Building launch segment profile...")
    customer_segment = agg[["customer_nr", "segment_key"]].copy()
    launch_rows = []

    launch_base = launches[["sku", "launch_date"]].dropna().drop_duplicates().copy()

    for _, launch in launch_base.iterrows():
        sku = str(launch["sku"])
        ldate = pd.to_datetime(launch["launch_date"], errors="coerce")

        if pd.isna(ldate):
            continue

        window_end = ldate + pd.Timedelta(days=41)

        sku_orders = customer_df[
            (customer_df["sku_str"] == sku)
            & (customer_df["date"] >= ldate)
            & (customer_df["date"] <= window_end)
        ][["customer_nr"]].drop_duplicates()

        if sku_orders.empty:
            continue

        sku_segments = sku_orders.merge(customer_segment, on="customer_nr", how="left")
        sku_counts = sku_segments["segment_key"].value_counts(normalize=True)

        for seg_key, share in sku_counts.items():
            launch_rows.append(
                {
                    "sku": sku,
                    "segment_key": seg_key,
                    "launch_segment_share": float(share),
                }
            )

    launch_segment_profile = pd.DataFrame(launch_rows)
    log_progress("[behavioral_segmentation] Launch segment profile built.")

    return {
        "enabled": True,
        "reason": "ok",
        "feature_columns": feature_cols,
        "scaler": scaler,
        "model": model,
        "customer_count": int(len(agg)),
        "cluster_count": int(cluster_count),
        "segment_summary": seg_summary,
        "launch_segment_profile": launch_segment_profile,
    }
# ============================================================
# MAIN
# ============================================================

def main():
    orders_raw, sale_times_raw, launches_raw = load_data()

    orders = clean_orders(orders_raw)
    sale_times = clean_sale_times(sale_times_raw)
    launches = clean_launches(launches_raw)

    orders = add_order_sale_flags(orders, sale_times)
    launches = add_launch_sale_features(launches, sale_times)

    # Core calibration
    seasonality_index = build_seasonality_index(orders)
    strategy_factors = build_strategy_factors(launches)
    flavour_factors = build_flavour_factors(launches)
    product_form_factors = build_product_form_factors(launches)
    sale_factors = build_sale_factors(launches)
    price_elasticity = build_price_elasticity(launches)
    growth_context = build_growth_context(orders)
    monthly_company_scale = build_monthly_company_scale(orders)

    # Ratio-based forecast assets
    monthly_new_customers = build_monthly_new_customers(orders)

    launch_ratio_table = build_historical_buyer_ratio_table(
        orders=orders,
        launches=launches,
        monthly_new_customers=monthly_new_customers,
    )

    launch_ratio_table = add_clipped_ratio_columns(launch_ratio_table)

    ratio_context = build_ratio_context(
        orders=orders,
        monthly_new_customers=monthly_new_customers,
    )

    # Target group inference
    mapping_df = ensure_target_group_mapping_file(launches)
    launches = apply_target_group_mapping(launches, mapping_df)

    tg_train_df, tg_numeric_cols, tg_class_counts, tg_top_classes = (
        build_target_group_training_table(
            launches=launches,
            orders=orders,
            top_n=TARGET_GROUP_TOP_N,
        )
    )

    target_group_inference = train_target_group_inference_model(
        train_df=tg_train_df,
        numeric_cols=tg_numeric_cols,
    )

    # Similarity and segmentation assets
    semantic_similarity = build_semantic_similarity_assets(launches)
    behavioral_segmentation = build_behavioral_segmentation(
        orders,
        launches,
        n_clusters=5,
    )
    artifacts = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "version": "v2",
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
            "seasonality_index": seasonality_index,
            "strategy_factors": strategy_factors,
            "flavour_factors": flavour_factors,
            "product_form_factors": product_form_factors,
            "sale_factors": sale_factors,
            "price_elasticity": price_elasticity,
            "growth_context": growth_context,
            "ratio_context": ratio_context,
            "target_metrics": TARGET_METRICS,
        },
        "target_group_inference": {
            **target_group_inference,
            "top_n": TARGET_GROUP_TOP_N,
            "top_classes": tg_top_classes,
            "class_counts": {
                str(k): int(v)
                for k, v in tg_class_counts.items()
            },
            "mapping_path": TARGET_GROUP_MAPPING_PATH,
        },
        "semantic_similarity": semantic_similarity,
        "behavioral_segmentation": behavioral_segmentation,
    }

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    launch_ratio_csv_path = os.path.join(ARTIFACT_DIR, "launch_ratio_table_v2.csv")
    monthly_new_customers_csv_path = os.path.join(ARTIFACT_DIR, "monthly_new_customers_v2.csv")

    launch_ratio_table.to_csv(launch_ratio_csv_path, index=False)
    monthly_new_customers.to_csv(monthly_new_customers_csv_path, index=False)

    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print("\n==========================================")
    print("Artifact saved successfully")
    print(f"Path: {ARTIFACT_PATH}")
    print("==========================================")

    print("\nMetadata:")
    print(artifacts["metadata"])

    print("\nSeasonality index:")
    print(seasonality_index)

    print("\nGrowth context:")
    print(growth_context)

    print("\nPrice elasticity:")
    print(price_elasticity)

    print("\nRatio context:")
    print(ratio_context)

    print("\nLaunch ratio table:")
    print(
        {
            "rows": int(len(launch_ratio_table)),
            "buyer_ratio_6w_median": float(
                launch_ratio_table["buyer_ratio_6w_existing"].median()
            ) if not launch_ratio_table.empty else None,
            "nc_ratio_6w_vs_monthly_nc_median": float(
                launch_ratio_table["nc_ratio_6w_vs_monthly_nc"].median()
            ) if not launch_ratio_table.empty else None,
            "units_per_customer_6w_median": float(
                launch_ratio_table["units_per_customer_6w"].median()
            ) if not launch_ratio_table.empty else None,
        }
    )

    print("\nSaved CSV exports:")
    print(launch_ratio_csv_path)
    print(monthly_new_customers_csv_path)

    print("\nTarget-group model:")
    print(
        {
            "enabled": target_group_inference.get("enabled", False),
            "training_rows": target_group_inference.get("training_rows", 0),
            "class_count": len(target_group_inference.get("classes", [])),
        }
    )

    print("\nSemantic similarity:")
    print(
        {
            "enabled": semantic_similarity.get("enabled", False),
            "rows": semantic_similarity.get("row_count", 0),
            "features": semantic_similarity.get("feature_count", 0),
        }
    )

    print("\nBehavioral segmentation:")
    print(
        {
            "enabled": behavioral_segmentation.get("enabled", False),
            "customers": behavioral_segmentation.get("customer_count", 0),
            "clusters": behavioral_segmentation.get("cluster_count", 0),
        }
    )


if __name__ == "__main__":
    main()