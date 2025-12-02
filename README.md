# Grocery Demand Predictor for Supermarkets

A machine learning project that predicts 7-day grocery demand for multiple products and provides reorder recommendations to prevent stock-outs using historical patterns, weather conditions, holidays, and offers.

Tech Stack:
Python
Pandas
Scikit-learn
Matplotlib
Streamlit

Products Included:

* Rice
* Milk
* Bread

Features:

* 1 year of auto-generated training data (365 days × 3 products)
* Trains separate models for each product (Random Forest)
* Predicts 7-day future demand
* Adds reorder alerts based on current stock
* Interactive Streamlit dashboard
* Historical trends per product
* Forecast vs average comparison

Project Structure:

Grocery_Demand_Predictor/
|
|-- data/
|   |-- train_model.csv
|
|-- model/
|   |-- rice_model.pkl
|   |-- milk_model.pkl
|   |-- bread_model.pkl
|
|-- src/
|   |-- generate_csv.py
|   |-- train_model.py
|   |-- predict.py
|   |-- app_v2.py
|
|-- requirements.txt
|-- README.md

How to Run Locally

1. Install Libraries

pip install -r requirements.txt

2. Generate data

python src/generate_csv.py

3. Train model

python src/train_model.py

4. Run prediction (optional)

python src/predict.py

5. Launch dashboard

streamlit run src/app_v2.py

Dashboard Features:
✅ Set temperature, holiday, and offer status for next 7 days
✅ Shows demand forecast for each product
✅ Displays “OK” or “REORDER” alerts
✅ Shows past 1-year sales trends
✅ Compares forecast with historical average

Use Case:
This system can help supermarket owners make data-driven inventory decisions, reduce waste, and prevent product shortages.

Future Improvements

* Add more products
* Include supplier lead time
* Deep Learning (LSTM/GRU) models
* REST API for POS systems
* Auto email/WhatsApp alerts

Created by: [Your Name]
Project: Grocery Demand Predictor
Year: 2025
