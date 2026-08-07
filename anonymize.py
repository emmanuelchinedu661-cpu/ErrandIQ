"""
anonymize.py — replaces real customer names with generic labels ("Customer 1", "Customer 2", ...)
while preserving which orders belong to the same customer, so repeat-customer analytics
(customer_stats, repeat rate, etc.) still work correctly on the anonymized data.

Run once: python anonymize.py
Produces: errandman-orders-anon.csv  (safe to commit — no real names)
"""
import pandas as pd

RAW_PATH = "errandman-orders (1).csv"   # match your actual filename
OUT_PATH = "errandman-orders-anon.csv"

df = pd.read_csv(RAW_PATH)

# Missing values become "Guest" (matches pipeline.py's own fillna logic)
df["Customer"] = df["Customer"].fillna("Guest")

# "Guest" stays as-is (already generic); everything else gets a consistent Customer N label
real_customers = df.loc[df["Customer"] != "Guest", "Customer"].unique()
name_map = {name: f"Customer {i+1}" for i, name in enumerate(sorted(real_customers))}

df["Customer"] = df["Customer"].map(lambda x: name_map.get(x, x))

# Phone columns already get dropped downstream in pipeline.py, but strip them here too
# in case anyone opens the anonymized CSV directly
df = df.drop(columns=["Customer Phone"], errors="ignore")

df.to_csv(OUT_PATH, index=False)
print(f"Anonymized {len(name_map)} unique customer names.")
print(f"Saved to {OUT_PATH}")