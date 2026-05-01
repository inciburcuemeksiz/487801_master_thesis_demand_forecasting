import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

ARTIFACT_PATH = "artifacts/model_artifacts_v2.pkl"

OUTPUT_DIR = "artifacts/backtesting"
DETAIL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "topn_backtest_details.csv")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "topn_backtest_summary.csv")

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]

TOP_N_VALUES = [3, 5, 7, 10, 15]

# Bu kolonlar similarity hesabında kullanılacak.
# Target kolonları dahil etmiyoruz, çünkü bu leakage olur.
SIMILARITY_FEATURE_COLS = [
    "uvp",
    "first_order_quantity",
    "launch_month",
    "launch_during_sale",
    "first_week_sale_days",
    "first_6_week_sale_days",
    "units_per_customer_1w",
    "units_per_customer_6w",
    "first_week_share_of_6w",
    "new_customer_share_1w",
    "new_customer_share_6w",
]


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


# ============================================================
# LOAD DATA
# ============================================================

def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f"Artifact file not found: {ARTIFACT_PATH}")

    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)

    return artifacts


def find_launch_dataframe(artifacts):
    """
    Finds the cleaned launch-level dataframe inside model_artifacts_v2.pkl.

    In your build_artifacts_v2.py, launches are stored here:

        artifacts["data"]["launches"]
    """

    if not isinstance(artifacts, dict):
        raise TypeError("Artifacts object is not a dictionary.")

    if "data" not in artifacts:
        raise KeyError("Could not find artifacts['data'].")

    if not isinstance(artifacts["data"], dict):
        raise TypeError("artifacts['data'] is not a dictionary.")

    if "launches" not in artifacts["data"]:
        raise KeyError("Could not find artifacts['data']['launches'].")

    launches = artifacts["data"]["launches"]

    if not isinstance(launches, pd.DataFrame):
        raise TypeError("artifacts['data']['launches'] is not a pandas DataFrame.")

    missing_targets = [col for col in TARGET_METRICS if col not in launches.columns]
    if missing_targets:
        raise ValueError(
            "Launch dataframe is missing target metric columns: "
            f"{missing_targets}"
        )

    print('Using launch dataframe from artifact key: artifacts["data"]["launches"]')

    return launches.copy()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def select_available_similarity_features(df):
    available_cols = [col for col in SIMILARITY_FEATURE_COLS if col in df.columns]

    missing_cols = [col for col in SIMILARITY_FEATURE_COLS if col not in df.columns]

    if missing_cols:
        print("\nWarning: These similarity feature columns are missing and will be skipped:")
        for col in missing_cols:
            print(f"- {col}")

    if not available_cols:
        raise ValueError("No similarity feature columns are available.")

    return available_cols


def clean_backtest_dataframe(df):
    """
    Keeps only rows with valid target metrics.
    """

    cleaned = df.copy()

    for metric in TARGET_METRICS:
        cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")

    cleaned = cleaned.dropna(subset=TARGET_METRICS).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid launch rows after dropping missing target metrics.")

    return cleaned


def prepare_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    # If a column is completely missing, median is also NaN.
    # Fill remaining NaNs with 0.
    X = X.fillna(0)

    # Standardization
    means = X.mean()
    stds = X.std().replace(0, 1)

    X_scaled = (X - means) / stds

    return X_scaled


# ============================================================
# SIMILARITY + FORECASTING
# ============================================================

def cosine_similarity_to_one(X_train, x_test):
    train_values = X_train.values.astype(float)
    test_values = x_test.values.astype(float).reshape(1, -1)

    train_norms = np.linalg.norm(train_values, axis=1)
    test_norm = np.linalg.norm(test_values)

    denominator = train_norms * test_norm
    denominator = np.where(denominator == 0, 1e-9, denominator)

    similarities = np.dot(train_values, test_values.T).flatten() / denominator

    return similarities


def get_top_comparables(train_df, train_X, test_x, top_n):
    similarities = cosine_similarity_to_one(train_X, test_x)

    comparable_df = train_df.copy()
    comparable_df["similarity_score"] = similarities

    comparable_df = comparable_df.sort_values(
        "similarity_score",
        ascending=False,
    ).head(top_n)

    return comparable_df


def predict_from_top_comparables(top_comparables, target_metric):
    """
    Similarity-weighted average.
    If all similarities are <= 0, fallback to simple mean.
    """

    values = pd.to_numeric(top_comparables[target_metric], errors="coerce")
    weights = pd.to_numeric(top_comparables["similarity_score"], errors="coerce").clip(lower=0)

    valid_mask = values.notna()

    values = values[valid_mask]
    weights = weights[valid_mask]

    if len(values) == 0:
        return np.nan

    if weights.sum() <= 0:
        return float(values.mean())

    return float(np.average(values, weights=weights))


# ============================================================
# BACKTEST
# ============================================================

def run_backtest_for_top_n(df, feature_cols, top_n):
    results = []

    X_all = prepare_feature_matrix(df, feature_cols)

    for test_idx in df.index:
        test_row = df.loc[test_idx]
        test_x = X_all.loc[test_idx]

        train_df = df.drop(index=test_idx).copy()
        train_X = X_all.drop(index=test_idx).copy()

        if len(train_df) < top_n:
            continue

        top_comparables = get_top_comparables(
            train_df=train_df,
            train_X=train_X,
            test_x=test_x,
            top_n=top_n,
        )

        row_result = {
            "test_index": int(test_idx),
            "top_n": int(top_n),
        }

        # Helpful identifiers
        for col in [
            "sku",
            "artikel_name",
            "product",
            "flavour",
            "flavour_group",
            "product_form",
            "launch_strategy_type",
            "launch_date",
        ]:
            if col in df.columns:
                row_result[col] = test_row[col]

        # Save comparable SKUs/products for explainability
        if "sku" in top_comparables.columns:
            row_result["comparable_skus"] = ", ".join(top_comparables["sku"].astype(str).tolist())

        if "artikel_name" in top_comparables.columns:
            row_result["comparable_names"] = " | ".join(
                top_comparables["artikel_name"].astype(str).tolist()
            )

        row_result["avg_similarity_score"] = float(top_comparables["similarity_score"].mean())

        for metric in TARGET_METRICS:
            actual = float(test_row[metric])
            pred = predict_from_top_comparables(top_comparables, metric)

            row_result[f"{metric}_actual"] = actual
            row_result[f"{metric}_pred"] = pred
            row_result[f"{metric}_abs_error"] = abs(actual - pred)

            if actual != 0:
                row_result[f"{metric}_ape"] = abs((actual - pred) / actual) * 100
            else:
                row_result[f"{metric}_ape"] = np.nan

        results.append(row_result)

    return pd.DataFrame(results)


def summarize_backtest(backtest_df):
    summary_rows = []

    for top_n, group in backtest_df.groupby("top_n"):
        summary = {
            "top_n": int(top_n),
            "n_tests": int(len(group)),
            "avg_similarity_score": float(group["avg_similarity_score"].mean()),
        }

        for metric in TARGET_METRICS:
            actual_col = f"{metric}_actual"
            pred_col = f"{metric}_pred"

            valid = group[[actual_col, pred_col]].dropna()

            if valid.empty:
                summary[f"{metric}_MAE"] = np.nan
                summary[f"{metric}_RMSE"] = np.nan
                summary[f"{metric}_MAPE"] = np.nan
                summary[f"{metric}_SMAPE"] = np.nan
                continue

            y_true = valid[actual_col]
            y_pred = valid[pred_col]

            summary[f"{metric}_MAE"] = float(mean_absolute_error(y_true, y_pred))
            summary[f"{metric}_RMSE"] = safe_rmse(y_true, y_pred)
            summary[f"{metric}_MAPE"] = safe_mape(y_true, y_pred)
            summary[f"{metric}_SMAPE"] = safe_smape(y_true, y_pred)

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    mape_cols = [f"{metric}_MAPE" for metric in TARGET_METRICS]
    smape_cols = [f"{metric}_SMAPE" for metric in TARGET_METRICS]
    mae_cols = [f"{metric}_MAE" for metric in TARGET_METRICS]

    summary_df["avg_MAPE_all_metrics"] = summary_df[mape_cols].mean(axis=1)
    summary_df["avg_SMAPE_all_metrics"] = summary_df[smape_cols].mean(axis=1)
    summary_df["avg_MAE_all_metrics"] = summary_df[mae_cols].mean(axis=1)

    summary_df = summary_df.sort_values(
        ["avg_MAPE_all_metrics", "avg_SMAPE_all_metrics", "avg_MAE_all_metrics"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading artifacts...")
    artifacts = load_artifacts()

    df = find_launch_dataframe(artifacts)
    df = clean_backtest_dataframe(df)

    print("\nHistorical launch dataframe shape:")
    print(df.shape)

    feature_cols = select_available_similarity_features(df)

    print("\nSimilarity feature columns used:")
    for col in feature_cols:
        print(f"- {col}")

    all_results = []

    for top_n in TOP_N_VALUES:
        print(f"\nRunning leave-one-launch-out backtest for top_n={top_n}...")
        result_df = run_backtest_for_top_n(
            df=df,
            feature_cols=feature_cols,
            top_n=top_n,
        )
        all_results.append(result_df)

    backtest_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_backtest(backtest_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    backtest_df.to_csv(DETAIL_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("\n==========================================")
    print("Backtesting summary")
    print("==========================================")
    print(summary_df)

    print("\n==========================================")
    print("Best top_n based on average MAPE")
    print("==========================================")

    best_row = summary_df.iloc[0]
    print(best_row[["top_n", "avg_MAPE_all_metrics", "avg_SMAPE_all_metrics"]])

    print("\nSaved files:")
    print(f"- Detail results: {DETAIL_OUTPUT_PATH}")
    print(f"- Summary results: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()