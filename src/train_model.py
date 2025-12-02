# src/train_model.py
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

os.makedirs("../model", exist_ok=True)

csv_path = os.path.join("..","data","train_model.csv")
df = pd.read_csv(csv_path)

products = df['product'].unique().tolist()
print("Products found:", products)

for product in products:
    data = df[df['product'] == product].copy()
    X = data[['temperature','is_holiday','is_offer']]
    y = data['sales']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"{product}: MSE = {mse:.2f}")

    pickle.dump(model, open(os.path.join("..","model", f"{product.lower()}_model.pkl"), "wb"))

print("✅ All models saved to ../model/")
