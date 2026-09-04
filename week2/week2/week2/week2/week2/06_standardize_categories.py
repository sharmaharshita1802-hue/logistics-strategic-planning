
"""
Standardizing Categorical Values
-----------------------------------
Normalizes inconsistent country/region labels using a mapping
dictionary after trimming whitespace and lowercasing.
"""

import pandas as pd

REGION_MAP = {
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
}


def standardize_regions(df: pd.DataFrame) -> pd.DataFrame:
    df["Order Country"] = (
        df["Order Country"].str.strip().str.lower().replace(REGION_MAP)
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("cleaned_step4.csv")
    df = standardize_regions(df)
    df.to_csv("cleaned_final.csv", index=False)
    print("Preprocessing pipeline complete.")
