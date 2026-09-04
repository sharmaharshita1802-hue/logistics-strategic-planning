"""
Handling Missing Values
------------------------
Imputes numeric columns with the median, categorical columns with a
placeholder, and drops rows missing the primary key.
"""

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Order Item Quantity", "Sales", "Days for shipping (real)"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df["Customer Segment"] = df["Customer Segment"].fillna("Unknown")

    df = df.dropna(subset=["Order Id"])
    return df


if __name__ == "__main__":
    from importlib import import_module
    load_data = import_module("01_load_and_inspect").load_data

    df = load_data()
    df = handle_missing_values(df)
    df.to_csv("cleaned_step1.csv", index=False)
    print(f"Remaining rows: {len(df)}")
