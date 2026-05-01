import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_DIR = "data/mock"

N_CUSTOMERS = 800
N_ORDERS = 3500
N_LAUNCHES = 62

PRODUCTS = [
    "Protein Bar",
    "Vitamin Drink",
    "Collagen Powder",
    "Energy Shot",
    "Omega Capsules",
    "Electrolyte Mix",
]

CATEGORIES = [
    "Nutrition",
    "Supplements",
    "Functional Drinks",
]

FLAVOURS = [
    "Chocolate",
    "Vanilla",
    "Berry",
    "Lemon",
    "Mango",
    "Neutral",
]

FLAVOUR_GROUPS = [
    "Sweet",
    "Fruity",
    "Citrus",
    "Neutral",
]

PRODUCT_FORMS = [
    "Bar",
    "Powder",
    "Drink",
    "Capsule",
    "Shot",
]

NEED_AREAS = [
    "Energy",
    "Beauty",
    "Immunity",
    "Fitness",
    "Hydration",
]

STRATEGIES = [
    "standard",
    "co_creation",
    "limited_edition",
]

TARGET_GROUPS = [
    "Fitness Enthusiasts",
    "Busy Professionals",
    "Beauty-Oriented Customers",
    "Health-Conscious Customers",
    "Price-Sensitive Buyers",
]


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def random_date(start, end):
    delta_days = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, delta_days + 1)))


def generate_launches():
    rows = []

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 1)

    for i in range(N_LAUNCHES):
        product = np.random.choice(PRODUCTS)
        flavour = np.random.choice(FLAVOURS)
        flavour_group = np.random.choice(FLAVOUR_GROUPS)
        product_form = np.random.choice(PRODUCT_FORMS)
        need_area = np.random.choice(NEED_AREAS)
        strategy = np.random.choice(STRATEGIES, p=[0.65, 0.2, 0.15])
        target_group = np.random.choice(TARGET_GROUPS)

        launch_date = random_date(start_date, end_date)
        uvp = round(float(np.random.uniform(8, 45)), 2)
        first_order_quantity = int(np.random.randint(300, 5000))

        # Synthetic target metrics with simple relationships.
        strategy_factor = {
            "standard": 1.0,
            "co_creation": 1.25,
            "limited_edition": 1.4,
        }[strategy]

        form_factor = {
            "Bar": 1.15,
            "Powder": 1.0,
            "Drink": 1.2,
            "Capsule": 0.85,
            "Shot": 0.9,
        }[product_form]

        price_factor = max(0.55, 1.4 - uvp / 60)
        base = first_order_quantity * 0.35 * strategy_factor * form_factor * price_factor

        first_week_quantity = max(10, int(np.random.normal(base, base * 0.25)))
        first_6_week_quantity = max(
            first_week_quantity + 10,
            int(first_week_quantity * np.random.uniform(2.0, 4.5)),
        )

        first_week_total_c = max(5, int(first_week_quantity / np.random.uniform(1.1, 2.2)))
        first_6_week_total_c = max(
            first_week_total_c + 5,
            int(first_6_week_quantity / np.random.uniform(1.2, 2.5)),
        )

        first_week_nc = int(first_week_total_c * np.random.uniform(0.25, 0.75))
        first_6_week_nc = int(first_6_week_total_c * np.random.uniform(0.20, 0.65))

        sku = f"MOCK-SKU-{i+1:04d}"

        rows.append(
            {
                "sku": sku,
                "artikel_name": f"{product} {flavour} {product_form}",
                "product": product,
                "product_need_area": need_area,
                "benefit_keywords": f"{need_area}, {product_form}, {flavour}",
                "flavour": flavour,
                "flavour_group": flavour_group,
                "product_form": product_form,
                "launch_date": launch_date.strftime("%Y-%m-%d"),
                "first_order_quantity": first_order_quantity,
                "uvp": uvp,
                "launch_strategy_type": strategy,
                "Product Use Case / What it is about": f"Mock product for {need_area.lower()} use case",
                "Target Group": target_group,
                "first_week_quantity": first_week_quantity,
                "first_6_week_quantity": first_6_week_quantity,
                "first_week_nc": first_week_nc,
                "first_6_week_nc": first_6_week_nc,
                "first_week_total_c": first_week_total_c,
                "first_6_week_total_c": first_6_week_total_c,
            }
        )

    return pd.DataFrame(rows)


def generate_sale_times():
    rows = [
        {
            "name": "Spring Sale",
            "start_d": "2023-03-15",
            "end_d": "2023-03-31",
        },
        {
            "name": "Summer Campaign",
            "start_d": "2024-06-01",
            "end_d": "2024-06-20",
        },
        {
            "name": "Black Friday",
            "start_d": "2024-11-20",
            "end_d": "2024-11-30",
        },
        {
            "name": "New Year Campaign",
            "start_d": "2025-01-01",
            "end_d": "2025-01-15",
        },
        {
            "name": "Autumn Sale",
            "start_d": "2025-09-10",
            "end_d": "2025-09-25",
        },
    ]

    return pd.DataFrame(rows)


def generate_orders(launches):
    rows = []

    customer_ids = [f"CUST-{i+1:05d}" for i in range(N_CUSTOMERS)]

    min_date = datetime(2022, 1, 1)
    max_date = datetime(2026, 1, 31)

    customer_first_order = {
        customer: random_date(min_date, datetime(2025, 12, 31))
        for customer in customer_ids
    }

    launch_skus = launches["sku"].tolist()
    launch_lookup = launches.set_index("sku").to_dict(orient="index")

    for order_i in range(N_ORDERS):
        customer = np.random.choice(customer_ids)
        order_date = random_date(customer_first_order[customer], max_date)

        # Some orders are for launch SKUs, some are generic mock SKUs.
        if np.random.rand() < 0.65:
            sku = np.random.choice(launch_skus)
            product_info = launch_lookup[sku]
            product = product_info["product"]
            artikel_name = product_info["artikel_name"]
            flavour = product_info["flavour"]
            category = np.random.choice(CATEGORIES)
            price = float(product_info["uvp"]) * np.random.uniform(0.85, 1.05)
        else:
            product = np.random.choice(PRODUCTS)
            flavour = np.random.choice(FLAVOURS)
            category = np.random.choice(CATEGORIES)
            sku = f"MOCK-REG-{np.random.randint(1, 250):04d}"
            artikel_name = f"{product} {flavour}"
            price = np.random.uniform(5, 50)

        quantity = int(np.random.choice([1, 1, 1, 2, 3]))
        net_revenue = round(price * quantity, 2)

        first_order_date = customer_first_order[customer]
        customer_status = "NEW" if abs((order_date - first_order_date).days) <= 7 else "EXISTING"

        rows.append(
            {
                "order_id": f"ORDER-{order_i+1:06d}",
                "customer_nr": customer,
                "sku": sku,
                "price": round(price, 2),
                "date": order_date.strftime("%Y-%m-%d"),
                "artikel_name": artikel_name,
                "net_revenue": net_revenue,
                "quantity": quantity,
                "flavour": flavour,
                "product_category": category,
                "product": product,
                "customer_status": customer_status,
                "first_order_date": first_order_date.strftime("%Y-%m-%d"),
                "last_order_date": order_date.strftime("%Y-%m-%d"),
                "months_since_first_order": max(0, int((order_date - first_order_date).days / 30)),
                "nr_of_purchase": int(np.random.randint(1, 20)),
            }
        )

    return pd.DataFrame(rows)


def generate_target_group_mapping():
    return pd.DataFrame(
        {
            "raw_target_group": TARGET_GROUPS,
            "canonical_target_group": TARGET_GROUPS,
        }
    )


def main():
    ensure_dirs()

    launches = generate_launches()
    sale_times = generate_sale_times()
    orders = generate_orders(launches)
    target_group_mapping = generate_target_group_mapping()

    orders.to_csv(os.path.join(RAW_DIR, "orders.csv"), index=False)
    sale_times.to_csv(os.path.join(RAW_DIR, "sale_times.csv"), index=False, sep=";")
    launches.to_csv(os.path.join(RAW_DIR, "launched_product_details.csv"), index=False)
    target_group_mapping.to_csv(os.path.join(RAW_DIR, "target_group_mapping.csv"), index=False)

    print("Mock data generated successfully:")
    print(f"- {os.path.join(RAW_DIR, 'orders.csv')} {orders.shape}")
    print(f"- {os.path.join(RAW_DIR, 'sale_times.csv')} {sale_times.shape}")
    print(f"- {os.path.join(RAW_DIR, 'launched_product_details.csv')} {launches.shape}")
    print(f"- {os.path.join(RAW_DIR, 'target_group_mapping.csv')} {target_group_mapping.shape}")


if __name__ == "__main__":
    main()
