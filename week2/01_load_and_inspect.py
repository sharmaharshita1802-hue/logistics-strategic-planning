"""
Load and Initial Inspection
----------------------------
Loads the raw logistics dataset and reports its shape and missing values.
"""

import pandas as pd


def load_data(path: str = "supply_chain_orders.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.isnull().sum().sort_values(ascending=False).head(10))
