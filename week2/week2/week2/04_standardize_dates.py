
"""
Standardizing Dates
---------------------
Parses order and shipping dates into a consistent datetime format,
flagging rows that fail to parse for manual review.
"""

import pandas as pd


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"], errors="coerce"
    )
    df["shipping date (DateOrders)"] = pd.to_datetime(
        df["shipping date (DateOrders)"], errors="coerce"
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("cleaned_step2.csv")
    df = standardize_dates(df)

    bad_dates = df[df["order date (DateOrders)"].isna()]
    print(f"Rows with unparseable order dates: {len(bad_dates)}")

    df.to_csv("cleaned_step3.csv", index=False)

