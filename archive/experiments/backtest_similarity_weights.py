import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model_artifacts_v2.pkl"

SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "weight_backtest_results.csv"
DETAIL_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "weight_backtest_predictions.csv"

RANDOM_SEARCH_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "weight_random_search_results.csv"
BEST_WEIGHTS_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "best_similarity_weights.csv"
BEST_WEIGHTS_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "best_similarity_weights_backtest_summary.csv"
BEST_WEIGHTS_DETAIL_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "best_similarity_weights_predictions.csv"


# ============================================================
# TARGET METRICS
# ============================================================

TARGET_METRICS = [
    "first_week_quantity",
    "first_6_week_quantity",
    "first_week_nc",
    "first_6_week_nc",
    "first_week_total_c",
    "first_6_week_total_c",
]


# ============================================================
# RANDOM SEARCH CONFIG
# ============================================================

RANDOM_SEARCH_ITERATIONS = 1000
RANDOM_SEED = 42
TOP_K = 3


# ============================================================
# BASELINE WEIGHT CONFIGURATIONS
# ============================================================

CURRENT_WEIGHTS = {
    "launch_month": 0.221323,
    "flavour": 0.183200,
    "product_need_area": 0.182740,
    "launch_strategy": 0.148905,
    "benefit_keywords": 0.123736,
    "price": 0.053567,
    "product_text": 0.046785,
    "flavour_group": 0.035550,
    "product_form": 0.004193,
}

WEIGHT_KEYS = list(CURRENT_WEIGHTS.keys())

EQUAL_WEIGHTS = {
    key: 1 / len(CURRENT_WEIGHTS)
    for key in CURRENT_WEIGHTS
}

CATEGORY_HEAVY_WEIGHTS = {
    "launch_month": 0.10,
    "flavour": 0.18,
    "product_need_area": 0.22,
    "launch_strategy": 0.10,
    "benefit_keywords": 0.12,
    "price": 0.06,
    "product_text": 0.08,
    "flavour_group": 0.08,
    "product_form": 0.06,
}

LAUNCH_HEAVY_WEIGHTS = {
    "launch_month": 0.28,
    "flavour": 0.12,
    "product_need_area": 0.12,
    "launch_strategy": 0.22,
    "benefit_keywords": 0.08,
    "price": 0.07,
    "product_text": 0.05,
    "flavour_group": 0.04,
    "product_form": 0.02,
}

WEIGHT_SETS = {
    "current_weights": CURRENT_WEIGHTS,
    "equal_weights": EQUAL_WEIGHTS,
    "category_heavy_weights": CATEGORY_HEAVY_WEIGHTS,
    "launch_heavy_weights": LAUNCH_HEAVY_WEIGHTS,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_weights(weights):
    total = sum(weights.values())

    if total == 0:
        raise ValueError("Weight total is zero. At least one weight must be positive.")

    return {k: float(v / total) for k, v in weights.items()}


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def wmape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / denominator


def token_jaccard_similarity(text_a, text_b):
    if pd.isna(text_a) or pd.isna(text_b):
        return 0.0

    text_a = str(text_a).lower().strip()
    text_b = str(text_b).lower().strip()

    if not text_a or not text_b or text_a == "nan" or text_b == "nan":
        return 0.0

    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    union = tokens_a | tokens_b

    if len(union) == 0:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def categorical_similarity(row_a, row_b, col):
    if col not in row_a.index or col not in row_b.index:
        return 0.0

    value_a = row_a[col]
    value_b = row_b[col]

    if pd.isna(value_a) or pd.isna(value_b):
        return 0.0

    return float(str(value_a).strip().lower() == str(value_b).strip().lower())


def price_similarity(row_a, row_b, col="price"):
    if col not in row_a.index or col not in row_b.index:
        return 0.0

    price_a = safe_float(row_a[col])
    price_b = safe_float(row_b[col])

    if pd.isna(price_a) or pd.isna(price_b):
        return 0.0

    max_price = max(abs(price_a), abs(price_b), 1.0)
    return 1 - min(abs(price_a - price_b) / max_price, 1)


# ============================================================
# SIMILARITY + FORECASTING
# ============================================================

def similarity_score(row_a, row_b, weights):
    """
    Computes weighted similarity between two historical launches.
    """

    score = 0.0

    categorical_cols = [
        "launch_month",
        "flavour",
        "product_need_area",
        "launch_strategy",
        "flavour_group",
        "product_form",
    ]

    for col in categorical_cols:
        score += weights.get(col, 0) * categorical_similarity(row_a, row_b, col)

    score += weights.get("price", 0) * price_similarity(row_a, row_b, "price")

    text_cols = [
        "product_text",
        "benefit_keywords",
    ]

    for col in text_cols:
        if col in row_a.index and col in row_b.index:
            text_sim = token_jaccard_similarity(row_a[col], row_b[col])
            score += weights.get(col, 0) * text_sim

    return score


def predict_from_similar_launches(target_row, candidate_rows, weights, metric, top_k=5):
    similarities = []

    for idx, candidate_row in candidate_rows.iterrows():
        sim = similarity_score(target_row, candidate_row, weights)
        similarities.append((idx, sim))

    if not similarities:
        return np.nan, []

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    top_pairs = similarities[:top_k]
    top_indices = [idx for idx, _ in top_pairs]
    top_sims = np.array([sim for _, sim in top_pairs], dtype=float)

    top_rows = candidate_rows.loc[top_indices].copy()

    if top_rows.empty or metric not in top_rows.columns:
        return np.nan, top_pairs

    values = pd.to_numeric(top_rows[metric], errors="coerce").values
    valid = ~np.isnan(values)

    values = values[valid]
    top_sims = top_sims[valid]

    if len(values) == 0:
        return np.nan, top_pairs

    if np.sum(top_sims) > 0:
        prediction = np.average(values, weights=top_sims)
    else:
        prediction = np.mean(values)

    return prediction, top_pairs


def run_backtest(launch_df, weights, weight_set_name, top_k=5):
    rows = []
    weights = normalize_weights(weights)

    available_metrics = [
        metric for metric in TARGET_METRICS
        if metric in launch_df.columns
    ]

    if not available_metrics:
        raise KeyError(
            "None of the TARGET_METRICS were found in launch_df. "
            f"Available columns include: {list(launch_df.columns)[:50]}"
        )

    for test_idx, target_row in launch_df.iterrows():
        train_df = launch_df.drop(index=test_idx)

        for metric in available_metrics:
            actual = safe_float(target_row[metric])

            if pd.isna(actual):
                continue

            prediction, top_pairs = predict_from_similar_launches(
                target_row=target_row,
                candidate_rows=train_df,
                weights=weights,
                metric=metric,
                top_k=top_k,
            )

            absolute_error = (
                abs(actual - prediction)
                if pd.notna(prediction)
                else np.nan
            )

            rows.append({
                "weight_set": weight_set_name,
                "test_launch_index": test_idx,
                "sku": target_row.get("sku", ""),
                "product": target_row.get("product", ""),
                "metric": metric,
                "actual": actual,
                "prediction": prediction,
                "absolute_error": absolute_error,
                "top_k": top_k,
                "top_match_indices": [idx for idx, _ in top_pairs],
                "top_match_scores": [score for _, score in top_pairs],
            })

    return pd.DataFrame(rows)


def evaluate_results(backtest_df):
    summary_rows = []

    for (weight_set, metric), group in backtest_df.groupby(["weight_set", "metric"]):
        y_true = pd.to_numeric(group["actual"], errors="coerce").values
        y_pred = pd.to_numeric(group["prediction"], errors="coerce").values

        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            continue

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        summary_rows.append({
            "weight_set": weight_set,
            "metric": metric,
            "n_launches": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": rmse,
            "MAPE": safe_mape(y_true, y_pred),
            "WMAPE": wmape(y_true, y_pred),
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# RANDOM SEARCH
# ============================================================

def generate_random_weights(weight_keys, rng):
    """
    Generates one random normalized weight set using a Dirichlet distribution.
    The generated weights always sum to 1.
    """
    values = rng.dirichlet(np.ones(len(weight_keys)))

    return {
        key: float(value)
        for key, value in zip(weight_keys, values)
    }


def score_weight_set(summary_df, score_metric="WMAPE"):
    """
    Computes one overall score for a candidate weight set.
    Lower is better.
    """
    if summary_df.empty:
        return np.nan

    return summary_df[score_metric].mean()


def run_random_weight_search(launch_df, n_iterations=5000, top_k=5, random_seed=42):
    rng = np.random.default_rng(random_seed)

    search_rows = []
    best_score = np.inf
    best_weights = None
    best_summary_df = None
    best_detail_df = None

    print("\nRunning random search for similarity weights...")
    print(f"Iterations: {n_iterations}")
    print("Optimization metric: average WMAPE across all target metrics")
    print("Lower is better.")

    for i in range(n_iterations):
        candidate_weights = generate_random_weights(WEIGHT_KEYS, rng)

        detail_df = run_backtest(
            launch_df=launch_df,
            weights=candidate_weights,
            weight_set_name=f"random_search_{i}",
            top_k=top_k,
        )

        summary_df = evaluate_results(detail_df)
        avg_wmape = score_weight_set(summary_df, score_metric="WMAPE")

        row = {
            "iteration": i,
            "avg_WMAPE": avg_wmape,
        }

        for key, value in candidate_weights.items():
            row[key] = value

        search_rows.append(row)

        if pd.notna(avg_wmape) and avg_wmape < best_score:
            best_score = avg_wmape
            best_weights = candidate_weights
            best_summary_df = summary_df.copy()
            best_detail_df = detail_df.copy()

        if (i + 1) % 500 == 0:
            print(
                f"Completed {i + 1}/{n_iterations} iterations "
                f"| best avg WMAPE: {best_score:.6f}"
            )

    search_results_df = pd.DataFrame(search_rows)
    search_results_df = search_results_df.sort_values("avg_WMAPE", ascending=True)

    return best_weights, best_score, best_summary_df, best_detail_df, search_results_df


# ============================================================
# ARTIFACT LOADING
# ============================================================

def load_launch_df():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Artifact file not found: {ARTIFACT_PATH}\n"
            "Run the artifact build first:\n"
            "python build_artifacts_v2.py"
        )

    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)

    print("\nAvailable artifact keys:")
    for key in artifacts.keys():
        print(f"- {key}")

    if (
        "supervised_ml" in artifacts
        and isinstance(artifacts["supervised_ml"], dict)
        and "ml_training_table" in artifacts["supervised_ml"]
        and isinstance(artifacts["supervised_ml"]["ml_training_table"], pd.DataFrame)
    ):
        launch_df = artifacts["supervised_ml"]["ml_training_table"].copy()
        print("\nUsing launch dataframe from:")
        print("artifacts['supervised_ml']['ml_training_table']")

    elif (
        "data" in artifacts
        and isinstance(artifacts["data"], dict)
        and "launches" in artifacts["data"]
        and isinstance(artifacts["data"]["launches"], pd.DataFrame)
    ):
        launch_df = artifacts["data"]["launches"].copy()
        print("\nUsing launch dataframe from:")
        print("artifacts['data']['launches']")

    else:
        raise KeyError(
            "Could not find a suitable launch-level dataframe. "
            "Expected artifacts['supervised_ml']['ml_training_table'] "
            "or artifacts['data']['launches']."
        )

    if "launch_month" not in launch_df.columns and "launch_date" in launch_df.columns:
        launch_df["launch_month"] = pd.to_datetime(
            launch_df["launch_date"],
            errors="coerce"
        ).dt.month

    if "price" not in launch_df.columns and "uvp" in launch_df.columns:
        launch_df["price"] = launch_df["uvp"]

    if "launch_strategy" not in launch_df.columns and "launch_strategy_type" in launch_df.columns:
        launch_df["launch_strategy"] = launch_df["launch_strategy_type"]

    column_aliases = {
        "product_need_area": "product_need_area_norm",
        "benefit_keywords": "benefit_keywords_norm",
        "flavour": "flavour_norm",
        "flavour_group": "flavour_group_norm",
        "product_form": "product_form_norm",
    }

    for target_col, source_col in column_aliases.items():
        if source_col in launch_df.columns:
            launch_df[target_col] = launch_df[source_col]

    if "product_text" not in launch_df.columns:
        text_columns = [
            "product",
            "artikel_name",
            "Product Use Case / What it is about",
            "Target Group",
            "use_case_norm",
            "target_group_norm",
        ]

        existing_text_columns = [
            col for col in text_columns
            if col in launch_df.columns
        ]

        if existing_text_columns:
            launch_df["product_text"] = ""

            for col in existing_text_columns:
                launch_df["product_text"] = (
                    launch_df["product_text"]
                    + " "
                    + launch_df[col].fillna("").astype(str)
                )

            launch_df["product_text"] = launch_df["product_text"].str.strip()
        else:
            launch_df["product_text"] = ""

    missing_metrics = [
        metric for metric in TARGET_METRICS
        if metric not in launch_df.columns
    ]

    if missing_metrics:
        raise KeyError(
            "The selected launch dataframe is missing target metrics:\n"
            f"{missing_metrics}\n"
            f"Available columns: {list(launch_df.columns)}"
        )

    similarity_columns = [
        "launch_month",
        "flavour",
        "product_need_area",
        "launch_strategy",
        "benefit_keywords",
        "price",
        "product_text",
        "flavour_group",
        "product_form",
    ]

    missing_similarity_cols = [
        col for col in similarity_columns
        if col not in launch_df.columns
    ]

    if missing_similarity_cols:
        raise KeyError(
            "The selected launch dataframe is missing similarity columns:\n"
            f"{missing_similarity_cols}\n"
            f"Available columns: {list(launch_df.columns)}"
        )

    print("\nLaunch dataframe shape:")
    print(launch_df.shape)

    print("\nAvailable target metrics:")
    for metric in TARGET_METRICS:
        print(f"- {metric}: OK")

    print("\nSimilarity columns used:")
    for col in similarity_columns:
        print(f"- {col}: OK")

    return launch_df.copy()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nArtifact path:")
    print(ARTIFACT_PATH)

    launch_df = load_launch_df()

    print("\nLaunch dataframe columns:")
    print(list(launch_df.columns))

    all_backtests = []

    for weight_set_name, weights in WEIGHT_SETS.items():
        print(f"\nRunning baseline backtest for: {weight_set_name}")

        result_df = run_backtest(
            launch_df=launch_df,
            weights=weights,
            weight_set_name=weight_set_name,
            top_k=TOP_K,
        )

        all_backtests.append(result_df)

    backtest_df = pd.concat(all_backtests, ignore_index=True)
    summary_df = evaluate_results(backtest_df)

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    backtest_df.to_csv(DETAIL_OUTPUT_PATH, index=False)

    sorted_summary_df = summary_df.sort_values(["metric", "WMAPE"])

    print("\nBaseline backtest summary sorted by metric and WMAPE:")
    print(sorted_summary_df.to_string(index=False))

    print(f"\nSaved baseline summary results to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved baseline detailed predictions to: {DETAIL_OUTPUT_PATH}")

    print("\nBest baseline weighting scheme by metric based on WMAPE:")
    best_by_metric = (
        summary_df.sort_values(["metric", "WMAPE"])
        .groupby("metric")
        .head(1)
        .reset_index(drop=True)
    )
    print(best_by_metric[["metric", "weight_set", "WMAPE", "MAE", "RMSE"]].to_string(index=False))

    # --------------------------------------------------------
    # Random search to empirically select similarity weights
    # --------------------------------------------------------

    (
        best_weights,
        best_score,
        best_summary_df,
        best_detail_df,
        search_results_df,
    ) = run_random_weight_search(
        launch_df=launch_df,
        n_iterations=RANDOM_SEARCH_ITERATIONS,
        top_k=TOP_K,
        random_seed=RANDOM_SEED,
    )

    search_results_df.to_csv(RANDOM_SEARCH_OUTPUT_PATH, index=False)

    best_weights_df = pd.DataFrame([
        {"attribute": key, "weight": value}
        for key, value in best_weights.items()
    ]).sort_values("weight", ascending=False)

    best_weights_df.to_csv(BEST_WEIGHTS_OUTPUT_PATH, index=False)
    best_summary_df.to_csv(BEST_WEIGHTS_SUMMARY_OUTPUT_PATH, index=False)
    best_detail_df.to_csv(BEST_WEIGHTS_DETAIL_OUTPUT_PATH, index=False)

    print("\nBest random-search weight set:")
    print(best_weights_df.to_string(index=False))

    print(f"\nBest average WMAPE: {best_score:.6f}")

    print("\nBacktest summary for best random-search weights:")
    print(best_summary_df.sort_values("WMAPE").to_string(index=False))

    print(f"\nSaved random search results to: {RANDOM_SEARCH_OUTPUT_PATH}")
    print(f"Saved best weights to: {BEST_WEIGHTS_OUTPUT_PATH}")
    print(f"Saved best weights backtest summary to: {BEST_WEIGHTS_SUMMARY_OUTPUT_PATH}")
    print(f"Saved best weights detailed predictions to: {BEST_WEIGHTS_DETAIL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()