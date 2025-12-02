# src/app_v2.py
import os
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Grocery Demand Predictor v2", layout="wide")
st.title("Grocery Demand Predictor v2 — Forecast, Trends & Reorder")

# Paths
DATA_PATH = os.path.join("..","data","train_model.csv")
MODEL_PATH = os.path.join("..","model")

# Load historical data
df = pd.read_csv(DATA_PATH)
products = sorted(df['product'].unique().tolist())

# Load models
models = {}
for p in products:
    models[p] = pickle.load(open(os.path.join(MODEL_PATH, f"{p.lower()}_model.pkl"), "rb"))

# Sidebar: dynamic inputs for 7 days
st.sidebar.header("7-Day Forecast Inputs")
forecast_days = 7
future_rows = {"temperature": [], "is_holiday": [], "is_offer": []}

st.sidebar.subheader("Set current stock (per product)")
current_stock = {}
for p in products:
    current_stock[p] = st.sidebar.number_input(f"Current stock: {p}", min_value=0, max_value=10000, value=100)

st.sidebar.markdown("---")
for d in range(forecast_days):
    st.sidebar.markdown(f"**Day {d+1} inputs**")
    temp = st.sidebar.slider(f"Temperature Day {d+1}", 15, 45, 30, key=f"temp{d}")
    hol = st.sidebar.selectbox(f"Is Holiday Day {d+1}?", [0,1], index=0, key=f"hol{d}")
    off = st.sidebar.selectbox(f"Is Offer Day {d+1}?", [0,1], index=0, key=f"off{d}")
    future_rows["temperature"].append(temp)
    future_rows["is_holiday"].append(hol)
    future_rows["is_offer"].append(off)

future_df = pd.DataFrame(future_rows)

# Predict
forecast_results = {}
alerts = {}
for p in products:
    preds = models[p].predict(future_df).astype(int)
    forecast_results[p] = preds
    alerts[p] = ["REORDER" if v > current_stock[p] else "OK" for v in preds]

df_forecast = pd.DataFrame(forecast_results)
df_alerts = pd.DataFrame(alerts)

# Show forecast table
st.subheader("7-Day Forecast (Units)")
st.dataframe(df_forecast)

# Download forecast CSV button
st.download_button("Download 7-day forecast CSV", df_forecast.to_csv(index=False).encode('utf-8'),
                   file_name="7day_forecast.csv", mime="text/csv")

# Show reorder alerts
st.subheader("Reorder Alerts")
st.dataframe(df_alerts)

# Historical trends (per product)
st.subheader("Historical Sales Trends (Year)")
for p in products:
    prod = df[df['product']==p].copy()
    prod['date'] = pd.to_datetime(prod['date'])
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(prod['date'], prod['sales'], linewidth=0.8)
    ax.set_title(f"{p} - Historical Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    st.pyplot(fig)

# Forecast vs historical average
st.subheader("Forecast vs Historical Average")
for p in products:
    hist_avg = int(df[df['product']==p]['sales'].mean())
    fig, ax = plt.subplots(figsize=(8,2.5))
    ax.plot(range(1,forecast_days+1), df_forecast[p], marker='o', label='Forecast')
    ax.hlines(hist_avg, 1, forecast_days, colors='red', linestyles='dashed', label=f'Historical Avg ({hist_avg})')
    ax.set_xticks(range(1,forecast_days+1))
    ax.set_xlabel("Day")
    ax.set_ylabel("Units Sold")
    ax.set_title(f"{p}: Forecast vs Historical Avg")
    ax.legend()
    st.pyplot(fig)
