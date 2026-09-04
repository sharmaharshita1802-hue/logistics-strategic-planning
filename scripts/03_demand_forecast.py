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
