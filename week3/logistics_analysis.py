================================================================================
FILE: week3/logistics_analysis.py
================================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
np.random.seed(42)

# ---------------------------------------------------------------
# 1. Data Simulation: hypothetical logistics dataset
# ---------------------------------------------------------------
n = 500
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Rail", "Air", "Sea"]

region = np.random.choice(regions, n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
mode = np.random.choice(modes, n, p=[0.5, 0.2, 0.15, 0.15])

mode_base_cost = {"Road": 8, "Rail": 5, "Air": 25, "Sea": 3}
mode_base_days = {"Road": 3, "Rail": 5, "Air": 1, "Sea": 12}

shipment_volume = np.random.gamma(shape=4, scale=50, size=n).round(0)  # units
distance_km = np.random.gamma(shape=3, scale=200, size=n).round(0)

transport_cost = np.array([
    mode_base_cost[m] * d / 100 + np.random.normal(0, 15)
    for m, d in zip(mode, distance_km)
])
transport_cost = np.clip(transport_cost, 20, None).round(2)

delivery_time_days = np.array([
    mode_base_days[m] + d / 800 + np.random.normal(0, 0.8)
    for m, d in zip(mode, distance_km)
])
delivery_time_days = np.clip(delivery_time_days, 0.5, None).round(1)

on_time = (delivery_time_days <= (
    np.array([mode_base_days[m] for m in mode]) + 1.5
)).astype(int)

df = pd.DataFrame({
    "region": region,
    "transport_mode": mode,
    "shipment_volume_units": shipment_volume,
    "distance_km": distance_km,
    "transport_cost_usd": transport_cost,
    "delivery_time_days": delivery_time_days,
    "on_time_delivery": on_time,
})

df.to_csv("simulated_logistics_data.csv", index=False)
print(df.head())
print(df.shape)

# ---------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------
desc = df[["shipment_volume_units", "distance_km", "transport_cost_usd", "delivery_time_days"]].describe()
desc.to_csv("descriptive_stats.csv")
print(desc)

corr = df[["shipment_volume_units", "distance_km", "transport_cost_usd", "delivery_time_days"]].corr()
corr.to_csv("correlation_matrix.csv")
print(corr)

otd_rate_by_region = df.groupby("region")["on_time_delivery"].mean().sort_values(ascending=False)
otd_rate_by_mode = df.groupby("transport_mode")["on_time_delivery"].mean().sort_values(ascending=False)
avg_cost_by_mode = df.groupby("transport_mode")["transport_cost_usd"].mean().sort_values(ascending=False)

print(otd_rate_by_region)
print(otd_rate_by_mode)
print(avg_cost_by_mode)

# ---------------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------------

# Chart 1: On-time delivery rate by region (bar chart)
plt.figure(figsize=(7, 4.2))
otd_rate_by_region.plot(kind="bar", color="#2F5496")
plt.title("On-Time Delivery Rate by Region")
plt.ylabel("On-Time Delivery Rate")
plt.xlabel("Region")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("chart1_otd_by_region.png", dpi=150)
plt.close()

# Chart 2: Distribution of delivery time (histogram)
plt.figure(figsize=(7, 4.2))
sns.histplot(df["delivery_time_days"], bins=25, color="#4472C4", kde=True)
plt.title("Distribution of Delivery Time (Days)")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart2_delivery_time_dist.png", dpi=150)
plt.close()

# Chart 3: Transport cost vs distance, colored by mode (scatter)
plt.figure(figsize=(7, 4.2))
sns.scatterplot(data=df, x="distance_km", y="transport_cost_usd", hue="transport_mode", palette="Set2", alpha=0.7)
plt.title("Transport Cost vs. Distance by Mode")
plt.xlabel("Distance (km)")
plt.ylabel("Transport Cost (USD)")
plt.legend(title="Mode", loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("chart3_cost_vs_distance.png", dpi=150)
plt.close()

# Chart 4: Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png", dpi=150)
plt.close()

# Chart 5: Average cost by transport mode (bar chart)
plt.figure(figsize=(7, 4.2))
avg_cost_by_mode.plot(kind="bar", color="#ED7D31")
plt.title("Average Transport Cost by Mode")
plt.ylabel("Average Cost (USD)")
plt.xlabel("Transport Mode")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart5_avg_cost_by_mode.png", dpi=150)
plt.close()

print("All charts saved.")
