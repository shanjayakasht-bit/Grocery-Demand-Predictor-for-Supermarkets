import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os


df = pd.read_csv('data/grocery_1year_3products.csv')
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
products = ["Rice", "Milk", "Bread"]
data = []
product_list = df['product'].unique().tolist()

for date in dates:
    for product in products:
        temperature = np.random.randint(20,36)
        is_holiday = 1 if date.weekday()==6 else 0
        is_offer = np.random.choice([0,1], p=[0.75,0.25])
        base_sales = {"Rice":100,"Milk":150,"Bread":120}
        units_sold = base_sales[product] + (10 if is_holiday else 0) + (5 if is_offer else 0) + np.random.randint(-10,15)
        data.append([date.strftime("%Y-%m-%d"), product, temperature, is_holiday, is_offer, units_sold])

df = pd.DataFrame(data, columns=["date","product","temperature","is_holiday","is_offer","units_sold"])

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

df.to_csv("data/grocery_1year_3products.csv", index=False)
print("✅ Generated grocery_1year_3products.csv")


for product in products:
    data_prod = df[df["product"]==product]
    X = data_prod[["temperature","is_holiday","is_offer"]]
    y = data_prod["units_sold"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"{product} model trained | MSE: {mse}")
    pickle.dump(model, open(f"model/{product.lower()}_model.pkl","wb"))

print("✅ All models trained!")


st.set_page_config(page_title="Grocery Demand Predictor", layout="wide")
st.title("Grocery Demand Predictor - 7-Day Forecast & Trends")

# Load models
models = {p: pickle.load(open(f"model/{p.lower()}_model.pkl","rb")) for p in products}

# Sidebar: Dynamic 7-day inputs
st.sidebar.header("7-Day Forecast Inputs")
forecast_days = 7
future_data = {"temperature":[],"is_holiday":[],"is_offer":[]}
current_stock = {}

for i in range(forecast_days):
    st.sidebar.subheader(f"Day {i+1}")
    temp = st.sidebar.slider(f"Temperature Day {i+1}", 20, 40, 30)
    holiday = st.sidebar.selectbox(f"Is Holiday Day {i+1}?", [0,1], index=0, key=f"holiday{i}")
    offer = st.sidebar.selectbox(f"Is Offer Day {i+1}?", [0,1], index=0, key=f"offer{i}")
    future_data["temperature"].append(temp)
    future_data["is_holiday"].append(holiday)
    future_data["is_offer"].append(offer)

for p in products:
    current_stock[p] = st.sidebar.number_input(f"Current stock {p}", 0, 1000, 100)

future_df = pd.DataFrame(future_data)

# Forecast
forecast_results = {}
for p in product_list:
    if p in models:
        forecast_results[p] = models[p].predict(future_df)
    else:
        print(f"Skipping product{p}: No trained model found.") 
        forecast_results[p] = [0] * len(future_df)  
    

df_forecast = pd.DataFrame(forecast_results)
st.subheader("7-Day Forecast (Units)")
st.dataframe(df_forecast.astype(int))

# Reorder alert
st.subheader("Reorder Alert")
alerts = {}
for p in products:
    alerts[p] = ["REORDER" if val > current_stock[p] else "OK" for val in df_forecast[p]]
st.dataframe(pd.DataFrame(alerts))

# Historical trends
st.subheader("Historical Sales Trend")
for p in products:
    prod_data = df[df["product"]==p]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(pd.to_datetime(prod_data["date"]), prod_data["units_sold"], label=f"{p} Historical Sales")
    ax.set_title(f"{p} Sales Trend Over the Year")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    st.pyplot(fig)

# Forecast vs Historical Avg
st.subheader("7-Day Forecast vs Historical Avg")
for p in products:
    hist_avg = df[df["product"]==p]["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(range(1,forecast_days+1), df_forecast[p], marker="o", label="Forecast")
    ax.hlines(hist_avg, 1, forecast_days, colors="red", linestyles="dashed", label="Historical Avg")
    ax.set_title(f"{p} Forecast vs Historical Avg")
    ax.set_xlabel("Day")
    ax.set_ylabel("Units Sold")
    ax.legend()
    st.pyplot(fig)
