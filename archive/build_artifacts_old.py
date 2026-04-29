"""
build_artifacts.py — Local ML pipeline
---------------------------------------
Runs the full data-prep + model training pipeline and saves model_artifacts.pkl
so that app.py can launch without needing the Colab notebook.

Usage:
    python build_artifacts.py --csv path/to/orders.csv
"""

import argparse
import hashlib
import pickle


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build model_artifacts.pkl from a CSV file.")
parser.add_argument("--csv", required=True, help="Path to the orders CSV file")
args = parser.parse_args()


# ── 1. LOAD & ANONYMISE ───────────────────────────────────────────────────────
print(f"[1/9] Loading CSV: {args.csv}")
df_raw = pd.read_csv(args.csv)

def anonymize_email(email):
    if pd.isna(email):
        return None
    return hashlib.sha256(email.encode()).hexdigest()[:16]

df_raw["customer_id"] = df_raw["customer_email"].apply(anonymize_email)
df_raw = df_raw.drop(columns=["customer_email"])


# ── 2. EXPLODE PRODUCT / QUANTITY LISTS ───────────────────────────────────────
print("[2/9] Exploding product & quantity lists...")
df_raw["product_list"]  = df_raw["product_list"].astype(str).str.split(",")
df_raw["quantity_list"] = df_raw["quantity_list"].astype(str).str.split(",")
df_raw["quantity_list"] = df_raw["quantity_list"].apply(
    lambda lst: [int(x.strip()) for x in lst]
)

rows = []
for _, row in df_raw.iterrows():
    products   = [p.strip() for p in row["product_list"]]
    quantities = row["quantity_list"]
    if len(products) != len(quantities):
        print(f"  ⚠️  Quantity mismatch at order_id: {row['order_id']}")
        continue
    for p, q in zip(products, quantities):
        rows.append({
            "customer_id":       row["customer_id"],
            "customer_status":   row["customer_status"],
            "order_date":        row["order_date"],
            "order_id":          row["order_id"],
            "product":           p,
            "quantity":          q,
            "net_revenue":       row["net_revenue"],
            "coupon_code":       row["coupon_code"],
            "coupon_influencer": row["coupon_influencer"],
        })

df_expanded = pd.DataFrame(rows)
df_expanded["order_date"] = pd.to_datetime(df_expanded["order_date"])


# ── 3. SALE PERIODS ───────────────────────────────────────────────────────────
print("[3/9] Tagging sale periods...")
sale_periods = [
    ("Winter_Sale_2024", "2024-01-23", "2024-01-28"),
    ("Easter_Sale_2024", "2024-03-28", "2024-04-03"),
    ("Summer_Sale_2024", "2024-08-26", "2024-09-03"),
    ("Early_Bird_2024",  "2024-10-28", "2024-11-04"),
    ("Black_Week_2024",  "2024-11-25", "2024-12-02"),
    ("Winter_Sale_2025", "2025-01-27", "2025-02-04"),
    ("Spring_Sale_2025", "2025-04-23", "2025-05-01"),
    ("Summer_Sale_2025", "2025-08-25", "2025-09-02"),
    ("Early_Bird_2025",  "2025-10-27", "2025-10-31"),
    ("Black_Week_2025",  "2025-11-24", "2025-12-01"),
]
sale_periods = [(n, pd.to_datetime(s), pd.to_datetime(e)) for n, s, e in sale_periods]

def get_sale_name(date):
    for name, start, end in sale_periods:
        if start <= date <= end:
            return name
    return None

df_expanded["sale_name"]      = df_expanded["order_date"].apply(get_sale_name)
df_expanded["is_sale_period"] = df_expanded["sale_name"].notna().astype(int)


# ── 4. PRODUCT CATEGORY FLAGS ─────────────────────────────────────────────────
print("[4/9] Computing product category affinities...")

health_products = [
    "Vitamin K2 Tropfen", "Vitamin D3+K2 Tropfen", "Vitamin D3 Omega Bundle",
    "Vitamin D3 + Magnesium Bundle", "Vitamin D + Test", "Vitamin C & Zink Kapseln",
    "Vitamin B12 Tropfen", "Vitamin B12 Tabletten", "Vitamin B Komplex",
    "Vegan Omega-3 Kapseln", "Vegan Omega 3", "Zink Kapseln", "Salbeiblatt Kapseln",
    "Salbei Extrakt", "Sägepalmen Kapseln", "Sägepalmen Extrakt", "Sägepalme Kapseln",
    "Reishi Premium Kapseln", "Reishi Forte Kapseln", "Reishi Forte",
    "Reishi Extrakt Kapseln", "Premium Greens", "Panax Ginseng Kapseln",
    "Magnesium Komplex Kapseln", "Panax Ginseng", "Omega-3 Kapseln", "L-Tyrosin",
    "L-Tryptophan Kapseln", "L-Lysin Kapseln", "L-Glutathion Kapseln",
    "Immune Protect Drink", "Immune Gummies", "Hydrate Drink", "Hyaluronsäure Kapseln",
    "Curcuma Kapseln", "Curcuma Extrakt", "Mood Kapseln", "Curcuma & Piperin Plus",
    "Vegan Basics", "Minerals Kapseln", "Colostrum Kapseln",
]
hydration_products = ["Hydrate Drink", "Recharge Drink", "Essential Aminos Drink"]
protein_products   = ["Whey Protein", "Vegan Protein Pulver"]

df_expanded["product_clean"] = df_expanded["product"].str.split("-").str[0].str.strip()
df_expanded["is_health_product"]    = df_expanded["product_clean"].isin(health_products).astype(int)
df_expanded["is_hydration_product"] = df_expanded["product_clean"].isin(hydration_products).astype(int)
df_expanded["is_protein_product"]   = df_expanded["product_clean"].isin(protein_products).astype(int)

df_expanded["is_prev_cocreation_product"] = (
    df_expanded["product"].str.contains("Vegan Protein Pulver", case=False) &
    df_expanded["product"].str.contains("Cookie", case=False)
).astype(int)

df_expanded["is_cocreation_influencer_customer"] = (
    df_expanded["coupon_influencer"].astype(str).str.lower() == "fit_laura"
).astype(int)


# ── 5. LAUNCH PERFORMANCE TABLE ───────────────────────────────────────────────
print("[5/9] Building launch performance table...")
launch_perf_df = pd.DataFrame([
    ("DAILY FIBER Lemon 330g Doypack DE/FR",                                    "Daily Fiber Drink",            "Lemon",            "2025-02-24", "mass market, health concious consumer",  "39,90 €", 15,  39,  0.2777777778, 54,   56),
    ("VEGAN PROTEIN Powder Neutral Doypack 600g DE/EN/FR",                      "Vegan Protein Pulver",         "Neutral",          "2025-09-29", "health conscious consumer",              "29,90 €", 37,  75,  0.3303571429, 112,  134),
    ("PREMIUM GREENS Apple-Kiwi 270g Doypack DE/FR",                            "Premium Greens",               "Apple Kiwi",       "2025-02-10", "health conscious consumer",              "69,90 €", 48,  229, 0.1732851986, 277,  296),
    ("VEGAN GLOW + CLEAR PROTEIN Mango Maracuja 300g Doypack DE Cocreation",    "Vegan Glow + Clear Protein Pulver", "Mango Maracuja","2025-03-10","Looking Good, Wellbeing",              "49,90 €", 95,  215, 0.3064516129, 310,  333),
    ("PREMIUM GREENS Mango-Maracuja CoCreation 270g Doypack DE/EN",             "Premium Greens",               "Mango Maracuja",   "2025-02-10", "health conscious consumer",              "69,90 €", 100, 467, 0.176366843,  567,  609),
    ("RECHARGE DRINK Tropical Fruits 360g Doypack DE/FR",                       "Recharge Drink",               "Tropical Fruits",  "2025-07-28", "health conscious consumer",              "39,90 €", 116, 406, 0.2222222222, 522,  578),
    ("VEGAN PROTEIN Powder Coffee Doypack 600g DE/EN/FR",                       "Vegan Protein Pulver",         "Coffee",           "2025-09-29", "health conscious consumer",              "32,90 €", 142, 244, 0.3678756477, 386,  423),
    ("MAGNESIUM DRINK Lavendel 120g Doypack DE",                                "Magnesium Drink",              "Lavender",         "2025-02-24", "trendy people, health concious consumer","32,90 €", 170, 367, 0.3165735568, 537,  552),
    ("DAILY GUT + IMMUNITY Ginger Lemon 240 g Doypack DE/FR Limited Edition",   "Daily Gut Pulver",             "Ginger Lemon",     "2025-10-06", "health conscious consumer",              "54,90 €", 189, 326, 0.3669902913, 515,  637),
    ("DAILY GUT CREAMY Pistachio 240 g Doypack DE/FR Limited Edition",          "Daily Gut Pulver",             "Creamy Pistachio", "2025-08-11", "mass market, health concious consumer",  "49,90 €", 191, 549, 0.2581081081, 740,  774),
    ("SUMMER COLLAGEN Mango Passionfruit 420g Doypack DE/FR Limited Edition",   "Summer Collagen",              "Mango Passionfruit","2025-06-24","health conscious consumer",              "49,90 €", 427, 820, 0.3424218123, 1247, 1357),
    ("VEGAN PROTEIN Pulver Cookie Dough Doypack 600g DE Cocreation",            "Vegan Protein Pulver",         "Cookie Dough",     "2025-04-22", "Looking Good, Wellbeing",                "32,90 €", 467, 1042,0.3094764745, 1509, 1594),
    ("DAILY GUT Powder Raspberry Hibiscus 240 g Doypack DE/FR Limited Edition", "Daily Gut Pulver",             "Raspberry Hibiscus","2025-08-11","mass market, health concious consumer",  "49,90 €", 473, 755, 0.3851791531, 1228, 1305),
    ("VEGAN PROTEIN Pulver Chocolate 600g Doypack DE/EN/FR",                    "Vegan Protein Pulver",         "Choco",            "2025-01-13", "Looking Good, Wellbeing",                "32,90 €", 479, 1011,0.3214765101, 1490, 1638),
    ("DAILY GUT CREAMY Matcha 240 g Doypack DE/FR Limited Edition",             "Daily Gut Pulver",             "Creamy Matcha",    "2025-07-07", "mass market, health concious consumer",  "49,90 €", 606, 938, 0.3924870466, 1544, 1672),
    ("MAGNESIUM DRINK Blueberry Lemon 120g Doypack DE",                         "Magnesium Drink",              "Blueberry Lemon",  "2025-02-24", "trendy people, health concious consumer","32,90 €", 624, 1055,0.3716497915, 1679, 1787),
    ("VEGAN PROTEIN Pulver Vanilla Cinnamon 600g Doypack DE/EN/FR",             "Vegan Protein Pulver",         "Vanilla Cinnamon", "2025-01-13", "Looking Good, Wellbeing",                "32,90 €", 796, 1527,0.342660353,  2323, 2795),
    ("DAILY GUT + COLLAGEN Powder Creamy Hazelnut 390 g Doypack DE CoCreation", "Daily Gut + Collagen Pulver",  "Hazelnut",         "2025-05-27", "mass market, health concious consumer",  "69,90 €", 849, 1735,0.3285603715, 2584, 2719),
    ("DAILY COLLAGEN Powder 450g Doypack DE/FR",                                "Daily Collagen Pulver",        "Neutral",          "2025-04-08", "mass market, health concious consumer",  "32,90 €", 932, 1342,0.4098504837, 2274, 2586),
    ("RECHARGE DRINK Lemon 360g Doypack DE/FR",                                 "Recharge Drink",               "Lemon",            "2025-07-28", "health conscious consumer",              "39,90 €", 15,  59,  0.1022405773, 74,   175),
    ("HYDRATE DRINK Powder 160g Doypack DE/FR",                                 "Hydrate Drink",                "Lemon",            "2025-05-19", "mass market, health concious consumer",  "29,90 €", 8,   34,  0.1707739667, 42,   45),
    ("DAILY GLOW COLLAGEN Raspberry Lemon 135g Doypack DE/FR",                  "Daily Glow Pulver",            "Raspberry Lemon",  "2024-12-10", "Looking Good, Wellbeing",                "34,90 €", 18,  36,  0.3181627509, 54,   55),
], columns=[
    "artikel_name", "product", "flavour", "launch_date", "target_group",
    "uvp", "nc_amount", "rc_amount", "nc_share", "total_customer", "total_quantity",
])

launch_perf_df["launch_date"] = pd.to_datetime(launch_perf_df["launch_date"])
launch_perf_df["uvp"] = (
    launch_perf_df["uvp"].astype(str)
    .str.replace("€", "").str.replace(",", ".").astype(float)
)
launch_perf_df["product_signature"] = (
    launch_perf_df["product"].astype(str).str.strip() + "–" +
    launch_perf_df["flavour"].astype(str).str.strip()
)


# ── 6. DATA-DRIVEN COEFFICIENTS ───────────────────────────────────────────────
print("[6/9] Computing data-driven coefficients...")

# Price elasticity (log-log regression)
df_price = launch_perf_df[["uvp", "total_quantity"]].dropna().copy()
df_price["log_price"] = np.log(df_price["uvp"])
df_price["log_qty"]   = np.log(df_price["total_quantity"])
coef = np.polyfit(df_price["log_price"], df_price["log_qty"], 1)
price_elasticity = coef[0]
print(f"  Price elasticity:      {price_elasticity:.4f}")

# Influencer / co-creation uplift
mask_co   = launch_perf_df["artikel_name"].str.contains("CoCreation", case=False)
co_avg    = launch_perf_df[mask_co]["total_quantity"].mean()
nonco_avg = launch_perf_df[~mask_co]["total_quantity"].mean()
influencer_uplift_factor = co_avg / nonco_avg if nonco_avg > 0 else 1.0
print(f"  Influencer uplift:     {influencer_uplift_factor:.4f}")


# ── 7. FULL FEATURE TABLE ─────────────────────────────────────────────────────
print("[7/9] Building customer feature table...")

df = df_expanded.copy()
df["customer_status"] = df["customer_status"].astype(str)
latest_date = df["order_date"].max()

def parse_product_flavour(x):
    if pd.isna(x):
        return None, None
    x = str(x).strip()
    if "–" in x:
        parts = x.split("–")
    elif "-" in x:
        parts = x.split("-")
    else:
        return x, None
    parts = [p.strip() for p in parts]
    return (parts[0], None) if len(parts) == 1 else (parts[0], parts[1])

df["product_clean"], df["flavour_clean"] = zip(*df["product"].apply(parse_product_flavour))
df["product_signature"] = (
    df["product_clean"].fillna("").astype(str) + "–" +
    df["flavour_clean"].fillna("").astype(str)
)
df["is_launch_product"] = df["product_signature"].isin(
    launch_perf_df["product_signature"].unique()
).astype(int)

# Sale-period features
sale_df = df.copy()
sale_df["sale_period_name"] = None
for name, start, end in sale_periods:
    mask = (sale_df["order_date"] >= start) & (sale_df["order_date"] <= end)
    sale_df.loc[mask, "sale_period_name"] = name

customer_last_sale_any = (
    sale_df[sale_df["sale_period_name"].notna()]
    .groupby("customer_id")["order_date"].max()
    .rename("last_sale_date_any_sale")
)
days_since_last_big_sale = (latest_date - customer_last_sale_any).dt.days
days_since_last_big_sale.name = "days_since_last_big_sale"

sale_periods_sorted = sorted(sale_periods, key=lambda x: x[2])
_, last_sale_start, last_sale_end = sale_periods_sorted[-1]
sale_df["bought_in_last_sale"] = (
    (sale_df["order_date"] >= last_sale_start) &
    (sale_df["order_date"] <= last_sale_end)
).astype(int)
bought_last_sale_flag = sale_df.groupby("customer_id")["bought_in_last_sale"].max()

# RFM + affinity features
recency       = df.groupby("customer_id")["order_date"].max().apply(lambda d: (latest_date - d).days).rename("recency_days")
frequency     = df.groupby("customer_id")["order_id"].nunique().rename("order_count")
total_qty     = df.groupby("customer_id")["quantity"].sum().rename("total_quantity")
health_aff    = df.groupby("customer_id")["is_health_product"].mean().rename("health_affinity")
hydration_aff = df.groupby("customer_id")["is_hydration_product"].mean().rename("hydration_affinity")
protein_aff   = df.groupby("customer_id")["is_protein_product"].mean().rename("protein_affinity")
prev_co_aff   = df.groupby("customer_id")["is_prev_cocreation_product"].max().rename("prev_cocreation_affinity")
inf_aff       = df.groupby("customer_id")["is_cocreation_influencer_customer"].max().rename("cocreation_influencer_affinity")
sale_freq     = df.groupby("customer_id")["is_sale_period"].mean().rename("sale_frequency")
last_sale_dt  = df[df["is_sale_period"] == 1].groupby("customer_id")["order_date"].max()
days_since_sale = (latest_date - last_sale_dt).dt.days.rename("days_since_last_sale")
df["days_from_latest"]  = (latest_date - df["order_date"]).dt.days
df["sale_last_60d_flag"] = ((df["is_sale_period"] == 1) & (df["days_from_latest"] <= 60)).astype(int)
sale_last_60d  = df.groupby("customer_id")["sale_last_60d_flag"].max().rename("sale_last_60d")
launch_aff     = df.groupby("customer_id")["is_launch_product"].mean().rename("launch_affinity")
cust_status    = df.groupby("customer_id")["customer_status"].first().rename("customer_status")

feature_table = pd.concat([
    recency, frequency, total_qty, health_aff, hydration_aff, protein_aff,
    prev_co_aff, inf_aff, sale_freq, days_since_sale, sale_last_60d,
    launch_aff, days_since_last_big_sale, bought_last_sale_flag, cust_status,
], axis=1).reset_index()

feature_table["days_since_last_sale"]     = feature_table["days_since_last_sale"].fillna(999)
feature_table["sale_frequency"]           = feature_table["sale_frequency"].fillna(0)
feature_table["sale_last_60d"]            = feature_table["sale_last_60d"].fillna(0)
feature_table["launch_affinity"]          = feature_table["launch_affinity"].fillna(0)
feature_table["days_since_last_big_sale"] = feature_table["days_since_last_big_sale"].fillna(999)
feature_table["bought_in_last_sale"]      = feature_table["bought_in_last_sale"].fillna(0)
feature_table["is_new_customer"]          = (feature_table["customer_status"].str.upper() == "NEW").astype(int)


# ── 8. PROPENSITY MODEL (MODEL 2) ───────────────────────────────────────────────
print("[8/9] Training propensity model (HistGradientBoosting)...")

feature_table["label_prev_cocreated"] = (feature_table["prev_cocreation_affinity"] > 0).astype(int)

feature_cols = [
    "recency_days", "order_count", "total_quantity", "health_affinity",
    "hydration_affinity", "protein_affinity", "prev_cocreation_affinity",
    "cocreation_influencer_affinity", "sale_frequency", "days_since_last_sale",
    "sale_last_60d", "launch_affinity", "days_since_last_big_sale",
    "bought_in_last_sale", "is_new_customer",
]

X = feature_table[feature_cols].fillna(0)
y = feature_table["label_prev_cocreated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=4, learning_rate=0.05,
    random_state=42,
)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"  Model AUC: {auc:.4f}")

feature_table["p_buy_model2"] = model.predict_proba(X)[:, 1]


# ── 9. HYDRATION MOMENTUM ─────────────────────────────────────────────────────
print("[9/9] Computing hydration momentum...")
hydration_df = df[df["is_hydration_product"] == 1].copy()

if hydration_df.empty:
    hydration_momentum = 1.0
else:
    hydration_df["week"] = hydration_df["order_date"].dt.isocalendar().week
    hydration_df["year"] = hydration_df["order_date"].dt.year
    cutoff12 = latest_date - pd.Timedelta(weeks=12)
    cutoff4  = latest_date - pd.Timedelta(weeks=4)
    w12 = hydration_df[hydration_df["order_date"] >= cutoff12].groupby(["year", "week"])["quantity"].sum().mean()
    w4  = hydration_df[hydration_df["order_date"] >= cutoff4].groupby(["year", "week"])["quantity"].sum().mean()
    hydration_momentum = (w4 / w12) if (w12 and w12 > 0) else 1.0

print(f"  Hydration momentum: {hydration_momentum:.4f}")


# ── SAVE ARTIFACTS ────────────────────────────────────────────────────────────
artifacts = {
    "launch_perf_df":          launch_perf_df,
    "feature_table":           feature_table,
    "auc":                     auc,
    "price_elasticity":        price_elasticity,
    "influencer_uplift_factor": influencer_uplift_factor,
    "hydration_momentum":      hydration_momentum,
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\n✅ model_artifacts.pkl saved successfully!")
print("   Run the app with:  python app.py")
