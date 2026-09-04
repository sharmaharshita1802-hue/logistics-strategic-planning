"""
Outlier Detection (IQR method)
--------------------------------
Flags and caps outliers in shipping duration using the interquartile
range (IQR) method, preserving row count by capping rather than
dropping.
"""

import pandas as pd


def flag_outliers_iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def cap_outliers(df: pd.DataFrame, column: str = "Days for shipping (real)") -> pd.DataFrame:
    df[f"{column}_outlier"] = flag_outliers_iqr(df[column])
    median_value = df[column].median()
    df.loc[df[f"{column}_outlier"], column] = median_value
    return df


if __name__ == "__main__":
    df = pd.read_csv("cleaned_step3.csv")
    df = cap_outliers(df)
    df.to_csv("cleaned_step4.csv", index=False)
    print(df["Days for shipping (real)_outlier"].value_counts())
