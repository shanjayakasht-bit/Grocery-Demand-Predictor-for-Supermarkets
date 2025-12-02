# src/predict.py
import os
import pickle
import pandas as pd

model_dir = os.path.join("..","model")
data_dir = os.path.join("..","data")
os.makedirs(data_dir, exist_ok=True)

products = ["Rice","Milk","Bread"]

# Example: dynamic 7-day inputs (edit these arrays as needed)
future_data = {
    "temperature": [30,31,29,28,30,32,31],
    "is_holiday":  [0,0,0,1,0,0,0],
    "is_offer":    [0,1,0,0,1,0,0]
}

future_df = pd.DataFrame(future_data)

# Current stock can be updated here
current_stock = {"Rice":150, "Milk":180, "Bread":160}

results = {}
alerts = {}

for p in products:
    model_path = os.path.join(model_dir, f"{p.lower()}_model.pkl")
    model = pickle.load(open(model_path, "rb"))
    preds = model.predict(future_df)
    results[p] = preds.astype(int)
    alerts[p] = ["REORDER" if val > current_stock[p] else "OK" for val in preds]

df_forecast = pd.DataFrame(results)
df_alerts = pd.DataFrame(alerts)

# Save outputs
df_forecast.to_csv(os.path.join(data_dir, "7day_forecast.csv"), index=False)
df_alerts.to_csv(os.path.join(data_dir, "7day_alerts.csv"), index=False)

print("7-day forecast:")
print(df_forecast)
print("\nReorder alerts:")
print(df_alerts)
print("\n✅ Saved to data/7day_forecast.csv and data/7day_alerts.csv")
