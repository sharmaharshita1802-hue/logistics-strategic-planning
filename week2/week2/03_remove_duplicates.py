"""
Removing Duplicates
--------------------
Drops duplicate orders based on the unique Order Id, keeping the
first occurrence.
"""

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="Order Id", keep="first")
    print(f"Removed {before - len(df)} duplicate order records")
    return df


if __name__ == "__main__":
    df = pd.read_csv("cleaned_step1.csv")
    df = remove_duplicates(df)
    df.to_csv("cleaned_step2.csv", index=False)


