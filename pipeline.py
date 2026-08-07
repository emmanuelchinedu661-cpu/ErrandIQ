"""
ErrandIQ data pipeline
- Cleans raw ErrandMan order export
- Loads into SQLite (drop-before-create)
- Engineers features
- Builds insight tables via queries.build_insight_tables() (all SQL lives in queries.py)
- Trains a rider accept-time model (small-N: cross-validated, no holdout)
"""
import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score

import queries

RAW_PATH = "errandman-orders-anon.csv" if os.path.exists("errandman-orders-anon.csv") else "errandman-orders (1).csv"
DB_PATH = "errandiq.db"
MODEL_PATH = "accept_time_model.pkl"
ASSIGN_MODEL_PATH = "assignment_model.pkl"
COMPLETED_LIKE = ["Completed", "Delivered"]

#---1. Load & clean
df =  pd.read_csv(RAW_PATH) # type: ignore
df["Created"] = pd.to_datetime(df["Created"])
df["Assigned"] = pd.to_datetime(df["Assigned"])

df["Rider"] = df["Rider"].fillna("Unassigned")
df["Customer"] = df["Customer"].fillna("Guest")

df["created_date"] = df["Created"].dt.date.astype(str)
df["created_hour"] = df["Created"].dt.hour
df["created_dow"] = df["Created"].dt.day_name()
df["created_month"] = df["Created"].dt.to_period("M").astype(str)
df["is_weekend"] = (df["Created"].dt.dayofweek >= 5).astype(int)
df["accept_min"] = (df["Assigned"] - df["Created"]).dt.total_seconds() / 60
df["has_from"] = df["From"].notna().astype(int)
df["has_to"] = df["To"].notna().astype(int)
df["is_completed"] = df["Status"].isin(COMPLETED_LIKE).astype(int)
df["is_refunded"] = (df["Status"] == "Refunded").astype(int)
df["is_cancelled"] = (df["Status"] == "Cancelled").astype(int)
df["was_assigned"] = df["Assigned"].notna().astype(int)

df_clean = df.drop(columns=["Customer Phone", "Rider Phone"], errors="ignore")
df_clean["Created"] = df_clean["Created"].astype(str)
df_clean["Assigned"] = df_clean["Assigned"].astype(str)

connection = sqlite3.connect(DB_PATH)
cur = connection.cursor()
cur.execute("DROP TABLE IF EXISTS orders")
connection.commit()
df_clean.to_sql("orders", connection, if_exists="replace", index=False)
queries.build_insight_tables(connection)

#  3. Accept-time model

labeled = df[df["accept_min"].notna()].copy()
labeled = labeled[labeled["accept_min"] < 60 * 24 * 3]
labeled["log_accept_min"] = np.log1p(labeled["accept_min"])

features = ["Type", "Title", "Rider", "Price", "created_hour", "created_dow", "is_weekend", "has_from", "has_to"]
x = labeled[features]
y = labeled["log_accept_min"]

cat_cols = ["Type", "Title", "Rider", "created_dow"]
preprocess = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
model = Pipeline([("prep", preprocess), ("ridge", Ridge(alpha=5.0))])

n = len(labeled)
kf = KFold(n_splits=min(5, n), shuffle=True, random_state=42)
cv_pred_log = cross_val_predict(model, x, y, cv=kf)
cv_pred_min = np.expm1(cv_pred_log)
actual_min = labeled["accept_min"].values

mae = mean_absolute_error(actual_min, cv_pred_min)
r2 = r2_score(y, cv_pred_log)
baseline_pred = np.full_like(actual_min, np.median(actual_min))
baseline_mae = mean_absolute_error(actual_min, baseline_pred)

print(f"[Accept-time] rows used: {n} (of {df['accept_min'].notna().sum()} total labeled, after outlier cap)")
print(f"[Accept-time] Cross-validated MAE (minutes): {mae:.1f}  |  Naive median-baseline MAE: {baseline_mae:.1f}")
print(f"[Accept-time] Cross-validated R2 (log space): {r2:.3f}")

model.fit(x, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": model, "features": features,
        "mae_minutes": mae, "baseline_mae_minutes": baseline_mae,
        "r2_log": r2, "n_train": n,
    }, f)

labeled_out = labeled[features + ["accept_min"]].copy()
labeled_out["cv_pred_min"] = cv_pred_min
cur.execute("DROP TABLE IF EXISTS accept_time_training")
connection.commit()
labeled_out.to_sql("accept_time_training", connection, if_exists="replace", index=False)
connection.commit()
# ---------- 6. Assignment-likelihood model (classification — uses ALL 100 orders) ----------
# Rider and Status are deliberately excluded: Rider = "Unassigned" and canceled/refunded
# Status values are only known AFTER assignment happens, so including them would be leakage —
# the model would be "cheating" by looking at the outcome to predict the outcome.
assign_features = ["Type", "Title", "Price", "created_hour", "created_dow", "is_weekend", "has_from", "has_to"]
x_assign = df[assign_features]
y_assign = df["was_assigned"]

assign_cat_cols = ["Type", "Title", "created_dow"]
assign_preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), assign_cat_cols),
    ("num", StandardScaler(), ["Price"]),
], remainder="passthrough")
assign_model = Pipeline([("prep", assign_preprocess), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_proba = cross_val_predict(assign_model, x_assign, y_assign, cv=skf, method="predict_proba")[:, 1]
assign_auc = roc_auc_score(y_assign, cv_proba)
assign_acc = accuracy_score(y_assign, (cv_proba >= 0.5).astype(int))
assign_baseline_acc = max(y_assign.mean(), 1 - y_assign.mean())

print(f"[Assignment] AUC: {assign_auc:.3f} (0.5 = random)  |  Accuracy: {assign_acc:.3f} vs majority-class baseline: {assign_baseline_acc:.3f}")

assign_model.fit(x_assign, y_assign)

with open(ASSIGN_MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": assign_model, "features": assign_features,
        "auc": assign_auc, "accuracy": assign_acc, "baseline_accuracy": assign_baseline_acc,
        "n_total": len(df), "n_assigned": int(y_assign.sum()),
    }, f)

assign_training = df[assign_features + ["was_assigned"]].copy()
assign_training["predicted_proba"] = cv_proba
cur.execute("DROP TABLE IF EXISTS assignment_training")
connection.commit()
assign_training.to_sql("assignment_training", connection, if_exists="replace", index=False)
connection.commit()

connection.close()
print("Pipeline complete. DB:", DB_PATH, "| Models:", MODEL_PATH, "&", ASSIGN_MODEL_PATH)
print("Tables: orders, rider_stats, customer_stats, route_stats, monthly_summary, service_reliability,")
print("demand_heatmap, accept_time_training, assignment_by_service, assignment_by_hour, assignment_training")