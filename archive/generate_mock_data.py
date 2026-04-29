"""
generate_mock_data.py
---------------------
Generates a realistic mock orders CSV that matches the schema expected by
build_artifacts.py, then saves it as mock_orders.csv.

Run:
    python generate_mock_data.py
"""

import random
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── Products (drawn from the real product catalogue) ──────────────────────────
PRODUCTS = [
    "Daily Gut Pulver",
    "Daily Gut Creamy",
    "Daily Gut + Immunity",
    "Daily Collagen Pulver",
    "Daily Glow Pulver",
    "Premium Greens",
    "Vegan Protein Pulver",
    "Vegan Glow + Clear Protein Pulver",
    "Hydrate Drink",
    "Recharge Drink",
    "Magnesium Drink",
    "Summer Collagen",
    "Daily Fiber Drink",
    "Magnesium Komplex Kapseln",
    "Gut Restore Kapseln",
    "Mood Kapseln",
    "Vegan Basics",
    "Immune Protect Drink",
    "Daily Gut + Collagen Pulver",
]

FLAVOURS = {
    "Daily Gut Pulver":               ["Raspberry Hibiscus", "Mango Maracuja", "Lemon"],
    "Daily Gut Creamy":               ["Creamy Matcha", "Creamy Pistachio"],
    "Daily Gut + Immunity":           ["Ginger Lemon"],
    "Daily Collagen Pulver":          ["Neutral", "Vanilla"],
    "Daily Glow Pulver":              ["Raspberry Lemon"],
    "Premium Greens":                 ["Apple Kiwi", "Mango Maracuja"],
    "Vegan Protein Pulver":           ["Cookie Dough", "Choco", "Vanilla Cinnamon", "Coffee", "Neutral"],
    "Vegan Glow + Clear Protein Pulver": ["Mango Maracuja"],
    "Hydrate Drink":                  ["Lemon"],
    "Recharge Drink":                 ["Tropical Fruits", "Lemon"],
    "Magnesium Drink":                ["Blueberry Lemon", "Lavender"],
    "Summer Collagen":                ["Mango Passionfruit"],
    "Daily Fiber Drink":              ["Lemon"],
    "Magnesium Komplex Kapseln":      [""],
    "Gut Restore Kapseln":            [""],
    "Mood Kapseln":                   [""],
    "Vegan Basics":                   [""],
    "Immune Protect Drink":           ["Ginger Lemon"],
    "Daily Gut + Collagen Pulver":    ["Hazelnut", "Neutral"],
}

INFLUENCERS = ["fit_laura", "healthyanna", "veganmia", "", "", "", "", ""]  # mostly empty
COUPON_CODES = ["SAVE10", "BLACKWEEK", "SUMMER20", "LAURA15", "", "", "", ""]
STATUSES = ["new", "returning", "returning", "returning"]  # ~25% new

# ── Date range: Jan 2024 – Apr 2026 ──────────────────────────────────────────
START = datetime(2024, 1, 1)
END   = datetime(2026, 4, 22)
DATE_RANGE = (END - START).days


def random_date():
    return START + timedelta(days=random.randint(0, DATE_RANGE))


def fake_email(i):
    return f"customer_{i:05d}@example.com"


# ── Generate customers ────────────────────────────────────────────────────────
N_CUSTOMERS = 800
N_ORDERS    = 3500

customers = {
    i: {
        "email":  fake_email(i),
        "status": random.choice(STATUSES),
    }
    for i in range(N_CUSTOMERS)
}

# ── Generate orders ───────────────────────────────────────────────────────────
rows = []
for order_idx in range(N_ORDERS):
    cust_id  = random.randint(0, N_CUSTOMERS - 1)
    cust     = customers[cust_id]
    n_items  = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

    selected = random.sample(PRODUCTS, min(n_items, len(PRODUCTS)))
    product_parts, qty_parts = [], []

    for prod in selected:
        flavours = FLAVOURS[prod]
        flavour  = random.choice(flavours)
        label    = f"{prod} - {flavour}" if flavour else prod
        product_parts.append(label)
        qty_parts.append(str(random.randint(1, 3)))

    influencer = random.choice(INFLUENCERS)
    coupon     = random.choice(COUPON_CODES)
    if influencer == "fit_laura" and not coupon:
        coupon = "LAURA15"

    rows.append({
        "customer_email":    cust["email"],
        "customer_status":   cust["status"],
        "order_date":        random_date().strftime("%Y-%m-%d"),
        "order_id":          f"ORD-{order_idx:05d}",
        "product_list":      ",".join(product_parts),
        "quantity_list":     ",".join(qty_parts),
        "net_revenue":       round(random.uniform(25, 120), 2),
        "coupon_code":       coupon,
        "coupon_influencer": influencer,
    })

df = pd.DataFrame(rows)
df.to_csv("mock_orders.csv", index=False)

print(f"✅  mock_orders.csv saved — {len(df)} orders, {N_CUSTOMERS} customers")
print(f"    Columns: {list(df.columns)}")
print(df.head(3).to_string())
