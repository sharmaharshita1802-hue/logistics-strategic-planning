"""
Exploratory Data Analysis
-------------------------
Visualizes daily order volume to understand demand trends over time.
"""

import pandas as pd
import matplotlib.pyplot as plt

from importlib import import_module

clean_orders = import_module("01_data_cleaning").clean_orders


def plot_daily_orders(orders: pd.DataFrame) -> None:
    daily_orders = orders.groupby(orders["order_date"].dt.date).size()

    plt.figure(figsize=(10, 4))
    daily_orders.plot(title="Daily Order Volume by Warehouse")
    plt.xlabel("Date")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig("daily_order_volume.png")
    plt.show()


if __name__ == "__main__":
    df = clean_orders()
    plot_daily_orders(df)
