================================================================================
FILE: week4/predictive_modeling.py
================================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Problem Definition and Data Simulation
# ---------------------------------------------------------------
# Target: predict delivery_time_days from shipment features
n = 800
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Rail", "Air", "Sea"]

region = np.random.choice(regions, n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
mode = np.random.choice(modes, n, p=[0.5, 0.2, 0.15, 0.15])
mode_base_days = {"Road": 3, "Rail": 5, "Air": 1, "Sea": 12}

shipment_volume = np.random.gamma(shape=4, scale=50, size=n).round(0)
distance_km = np.random.gamma(shape=3, scale=200, size=n).round(0)
warehouse_load_pct = np.clip(np.random.normal(65, 15, n), 10, 100).round(1)

delivery_time_days = np.array([
    mode_base_days[m] + d / 800 + wl / 200 + np.random.normal(0, 0.7)
    for m, d, wl in zip(mode, distance_km, warehouse_load_pct)
])
delivery_time_days = np.clip(delivery_time_days, 0.5, None).round(2)

df = pd.DataFrame({
    "region": region,
    "transport_mode": mode,
    "shipment_volume_units": shipment_volume,
    "distance_km": distance_km,
    "warehouse_load_pct": warehouse_load_pct,
    "delivery_time_days": delivery_time_days,
})
df.to_csv("model_dataset.csv", index=False)
print(df.head())

# ---------------------------------------------------------------
# 2. Model Selection and Implementation
# ---------------------------------------------------------------
X = df.drop(columns=["delivery_time_days"])
y = df["delivery_time_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ["region", "transport_mode"]
numeric = ["shipment_volume_units", "distance_km", "warehouse_load_pct"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

    results[name] = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R2": round(r2, 3),
        "CV_R2_mean": round(cv_scores.mean(), 3),
        "CV_R2_std": round(cv_scores.std(), 3),
    }
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")
print(results_df)

# ---------------------------------------------------------------
# 3. Hyperparameter Tuning (Random Forest)
# ---------------------------------------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 8, 12],
}
rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(random_state=42))])
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV R2:", grid.best_score_)

best_pipe = grid.best_estimator_
best_preds = best_pipe.predict(X_test)
best_rmse = mean_squared_error(y_test, best_preds) ** 0.5
best_r2 = r2_score(y_test, best_preds)
print(f"Tuned RF -> RMSE: {best_rmse:.3f}, R2: {best_r2:.3f}")

# ---------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------

# Chart 1: Actual vs Predicted (best model = tuned Random Forest)
plt.figure(figsize=(6, 5.5))
plt.scatter(y_test, best_preds, alpha=0.5, color="#4472C4", edgecolor="none")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, color="#ED7D31", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Actual vs. Predicted Delivery Time (Tuned Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_actual_vs_predicted.png", dpi=150)
plt.close()

# Chart 2: Model comparison (R2 bar chart)
plt.figure(figsize=(7, 4.2))
r2_values = [results[m]["R2"] for m in models.keys()]
plt.bar(list(models.keys()), r2_values, color=["#4472C4", "#ED7D31", "#70AD47"])
plt.ylabel("R\u00b2 Score (Test Set)")
plt.title("Model Comparison: R\u00b2 on Test Set")
plt.tight_layout()
plt.savefig("chart2_model_comparison.png", dpi=150)
plt.close()

# Chart 3: Feature importance (tuned Random Forest)
feature_names = (
    list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical))
    + numeric
)
importances = best_pipe.named_steps["model"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(10)

plt.figure(figsize=(7, 5))
imp_series.plot(kind="barh", color="#70AD47")
plt.title("Top Feature Importances (Tuned Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("chart3_feature_importance.png", dpi=150)
plt.close()

print("Charts saved.")
print(imp_series)
