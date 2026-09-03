================================================================================
FILE: README.md
================================================================================

# Logistics Strategic Planning & Data Exploration

Week 1 project for the NSDC / Yuva Intern data science track: a strategic
planning exercise for a regional e-commerce distribution network, covering
inventory management, route optimization, and supply chain integration.

## Scenario

A mid-sized e-commerce retailer ships orders from three regional warehouses
to customers across a metropolitan area. The goal is to design a data-driven
strategy that improves supply chain efficiency and resource allocation.

## Key Performance Indicators (KPIs)

| KPI | Definition |
|---|---|
| On-Time Delivery Rate (OTD) | % of orders delivered within the promised window |
| Inventory Turnover Ratio | COGS / average inventory value |
| Cost Per Delivery | Total distribution cost / number of deliveries |
| Order Fill Rate | % of orders fulfilled completely from available stock |

## Roadmap

1. **Data Collection** — order history, inventory logs, delivery timestamps, coordinates
2. **Data Cleaning** — `scripts/01_data_cleaning.py`
3. **Exploratory Analysis** — `scripts/02_eda.py`
4. **Demand Forecasting (Regression)** — `scripts/03_demand_forecast.py`
5. **Delivery Zone Clustering** — `scripts/04_zone_clustering.py`
6. **Route Optimization** — `scripts/05_route_optimization.py`

## Project Structure

```
logistics-strategic-planning/
├── README.md
├── requirements.txt
├── report/
│   └── Week1_Strategic_Planning_Report.docx
└── scripts/
    ├── 01_data_cleaning.py
    ├── 02_eda.py
    ├── 03_demand_forecast.py
    ├── 04_zone_clustering.py
    └── 05_route_optimization.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Expected Outcomes

More accurate warehouse-level demand forecasts, shorter/optimized delivery
routes, and a repeatable weekly analytics pipeline — raising On-Time Delivery
toward the 95%+ industry benchmark, improving Inventory Turnover, and
lowering Cost Per Delivery.


================================================================================
FILE: requirements.txt
================================================================================

pandas
matplotlib
scikit-learn
ortools


================================================================================
FILE: scripts/01_data_cleaning.py
================================================================================

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


================================================================================
FILE: scripts/02_eda.py
================================================================================

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


================================================================================
FILE: scripts/03_demand_forecast.py
================================================================================

"""
Demand Forecasting with Regression
-----------------------------------
Predicts daily demand per warehouse using a simple linear regression
model, so inventory can be planned ahead of demand spikes.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def forecast_demand(orders: pd.DataFrame):
    features = orders[["day_of_week", "promo_flag", "past_7day_avg"]]
    target = orders["daily_demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_demand = model.predict(X_test)

    return model, predicted_demand, y_test


if __name__ == "__main__":
    orders = pd.read_csv("cleaned_orders.csv")
    model, predictions, actuals = forecast_demand(orders)
    print("Sample predictions:", predictions[:5])


================================================================================
FILE: scripts/04_zone_clustering.py
================================================================================

"""
Delivery Zone Clustering
------------------------
Groups delivery addresses into geographically coherent zones using
k-means, forming the basis for route assignment in the next step.
"""

import pandas as pd
from sklearn.cluster import KMeans


def assign_delivery_zones(orders: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    coords = orders[["delivery_lat", "delivery_lon"]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    orders["delivery_zone"] = kmeans.fit_predict(coords)

    return orders


if __name__ == "__main__":
    orders = pd.read_csv("cleaned_orders.csv")
    zoned = assign_delivery_zones(orders)
    zoned.to_csv("orders_with_zones.csv", index=False)
    print(zoned["delivery_zone"].value_counts())


================================================================================
FILE: scripts/05_route_optimization.py
================================================================================

"""
Route Optimization (simplified)
--------------------------------
Solves a vehicle routing problem (VRP) for a delivery zone using
Google OR-Tools, minimizing total route distance/cost.
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def solve_routes(distance_matrix, num_vehicles: int, depot: int):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    return routing.SolveWithParameters(params)


if __name__ == "__main__":
    # Example toy distance matrix (5 stops incl. depot at index 0)
    distance_matrix = [
        [0, 9, 8, 7, 6],
        [9, 0, 5, 4, 3],
        [8, 5, 0, 3, 2],
        [7, 4, 3, 0, 1],
        [6, 3, 2, 1, 0],
    ]
    solution = solve_routes(distance_matrix, num_vehicles=2, depot=0)
    print("Solved:", solution is not None)
