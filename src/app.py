import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Grocery Demand Predictor", layout="wide")
st.title("Grocery Demand Predictor - 7-Day Forecast & Reorder Alert")

# Load models
products = ["Rice","Milk","Bread"]
models = {p: pickle.load(open(f"model/{p.lower()}_model.pkl","rb")) for p in products}

# Sidebar inputs
st.sidebar.header("Future Forecast Inputs")
temp_input = st.sidebar.slider("Temperature", 20, 40, 30, 1)
holiday_input = st.sidebar.selectbox("Is Holiday?", [0,1])
offer_input = st.sidebar.selectbox("Is Offer?", [0,1])
current_stock = {p: st.sidebar.number_input(f"Current stock {p}", 0, 1000, 100) for p in products}

future_data = pd.DataFrame({
    "temperature":[temp_input]*7,
    "is_holiday":[holiday_input]*7,
    "is_offer":[offer_input]*7
})

# Forecast
forecast_results = {}
for product in products:
    pred = models[product].predict(future_data)
    forecast_results[product] = pred

df_forecast = pd.DataFrame(forecast_results)
st.subheader("7-Day Forecast (Units)")
st.dataframe(df_forecast.astype(int))

# Reorder alert
st.subheader("Reorder Alert")
alert = {}
for product in products:
    alerts[product] = ["REORDER" if val > current_stock[product] else "OK" for val in df_forecast[product]]

df_alerts = pd.DataFrame(alerts)
st.dataframe(df_alerts)

