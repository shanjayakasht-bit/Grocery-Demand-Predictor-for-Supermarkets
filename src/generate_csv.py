# src/generate_csv.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.makedirs("../data", exist_ok=True)  # when running from src/, data is one level up

start_date = datetime(2024, 1, 1)
products = ["Rice", "Milk", "Bread"]
rows = []

for i in range(365):
    current_date = start_date + timedelta(days=i)
    temperature = np.random.randint(20, 36)
    is_holiday = 1 if current_date.weekday() == 6 or i % 30 == 0 else 0
    is_offer = np.random.choice([0, 1], p=[0.7, 0.3])

    for product in products:
        base_sales = {"Rice": 110, "Milk": 90, "Bread": 70}
        sales = base_sales[product] + (20 if is_holiday else 0) + (25 if is_offer else 0) + np.random.randint(-10, 15)
        rows.append([
            current_date.strftime("%Y-%m-%d"),
            product,
            temperature,
            is_holiday,
            is_offer,
            sales
        ])

df = pd.DataFrame(rows, columns=["date","product","temperature","is_holiday","is_offer","sales"])

output_path = os.path.join("..","data","train_model.csv")
df.to_csv(output_path, index=False)
print("✅ Created:", output_path)
