"""
Data Cleaning
-------------
Loads raw warehouse order data and prepares it for analysis:
- removes duplicate orders
- standardizes timestamps
- fills missing delivery status
- drops rows without delivery coordinates (needed for clustering later)
"""

import pandas as pd


def clean_orders(input_path: str = "warehouse_orders.csv") -> pd.DataFrame:
    orders = pd.read_csv(input_path)

    # Drop duplicate order records
    orders = orders.drop_duplicates(subset="order_id")

    # Standardize timestamps and fill missing delivery status
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["delivery_status"] = orders["delivery_status"].fillna("pending")

    # Remove rows with missing coordinates (needed for clustering later)
    orders = orders.dropna(subset=["delivery_lat", "delivery_lon"])

    return orders


if __name__ == "__main__":
    df = clean_orders()
    df.to_csv("cleaned_orders.csv", index=False)
    print(f"Cleaned {len(df)} order records.")
