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
