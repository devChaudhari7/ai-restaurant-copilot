from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sample_pos_data(n_rows: int = 800) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    items = [
        {"item_name": "Margherita Pizza", "category": "Pizza", "subcategory": "Veg", "unit_price": 8.99, "food_cost": 3.0},
        {"item_name": "Pepperoni Pizza", "category": "Pizza", "subcategory": "Non-Veg", "unit_price": 10.99, "food_cost": 4.0},
        {"item_name": "Veggie Supreme Pizza", "category": "Pizza", "subcategory": "Veg", "unit_price": 11.49, "food_cost": 4.2},
        {"item_name": "Garlic Bread", "category": "Sides", "subcategory": "Bread", "unit_price": 4.49, "food_cost": 1.2},
        {"item_name": "French Fries", "category": "Sides", "subcategory": "Fried", "unit_price": 3.99, "food_cost": 1.0},
        {"item_name": "Coke", "category": "Beverages", "subcategory": "Soft Drink", "unit_price": 2.49, "food_cost": 0.4},
        {"item_name": "Diet Coke", "category": "Beverages", "subcategory": "Soft Drink", "unit_price": 2.49, "food_cost": 0.4},
        {"item_name": "Lemonade", "category": "Beverages", "subcategory": "Juice", "unit_price": 3.49, "food_cost": 0.7},
        {"item_name": "Chicken Wings", "category": "Sides", "subcategory": "Chicken", "unit_price": 7.99, "food_cost": 3.2},
        {"item_name": "Chocolate Brownie", "category": "Dessert", "subcategory": "Cake", "unit_price": 5.49, "food_cost": 1.5},
    ]

    start_date = datetime.now() - timedelta(days=30)

    rows = []
    current_order_id = 1000

    for _ in range(n_rows):
        # New order every 1-3 rows
        if random.random() < 0.4 or not rows:
            current_order_id += 1

        item = random.choice(items)
        quantity = np.random.choice([1, 1, 1, 2, 3], p=[0.4, 0.3, 0.15, 0.1, 0.05])

        # Time distribution: busier evenings and weekends
        days_offset = random.randint(0, 29)
        base_time = start_date + timedelta(days=days_offset)
        hour = np.random.choice(
            [11, 12, 13, 18, 19, 20, 21],
            p=[0.05, 0.1, 0.05, 0.25, 0.25, 0.2, 0.1],
        )
        minute = random.randint(0, 59)
        tx_time = base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

        rows.append(
            {
                "order_id": current_order_id,
                "item_name": item["item_name"],
                "quantity": quantity,
                "unit_price": item["unit_price"],
                "food_cost_per_unit": item["food_cost"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "transaction_date": tx_time.isoformat(),
                "day_of_week": tx_time.strftime("%A"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    # Always write into the data/ directory alongside this script
    from pathlib import Path

    df = generate_sample_pos_data()
    out_path = Path(__file__).resolve().parent / "sample_pos_data.csv"
    df.to_csv(out_path, index=False)
    print("Generated", out_path, "with", len(df), "rows")


if __name__ == "__main__":
    main()

