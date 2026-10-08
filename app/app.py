import streamlit as st
import sys
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

from development.feature_selection import feature_selection
from development.model_monitering import save_prediction

model = joblib.load(
    os.path.join(BASE_DIR,"model1.pkl")
)

preprocessing_pipeline = joblib.load(
    os.path.join(BASE_DIR,"preprocessing_pipeline.pkl")
)

#stramlit page configuration
st.set_page_config(
    page_title = "Credit Card Fraud Detection"
)

st.title("fraud detection ml application")

st.write("enetr details to view prediction")

#customer information

st.subheader("customer information")

card_type = st.selectbox(
    "card_type",
    ["Visa","Mastercard","Amex","RuPay","Discover"]
)

auth_method = st.selectbox(
    "auth_method",
    ["OTP","3D Secure","No Authentication","Biometric","PIN"]
)
channel = st.selectbox(
    "channel",
    ["Online","POS","In-App","Contactless","ATM"] 
)
device_type = st.selectbox(
    "device_type",
    ["Android Phone","Mac","iPhone","POS Terminal",
     "ATM Machine","Tablet","Windows PC","Smart Watch"]
)

is_foreign_transaction = st.selectbox(
    "is_foreign_transaction",
    ["False","True"]
)

txn_count_last_24h = st.number_input(
    "txn_count_last_24h",
    min_value=1,
    value = 50
)

customer_age = st.number_input(
    "customer_age",
    min_value=1,
    value = 50
)

is_new_merchant = st.selectbox(
    "is_new_merchant",
    ["False","True"]
)

used_vpn = st.selectbox(
    "used_vpn",
    ["False","True"]
)

ip_country_mismatch = st.selectbox(
    "ip_country_mismatch",
    ["False","True"]
)

billing_shipping_mismatch = st.selectbox(
    "billing_shipping_mismatch",
    ["False","True"]
)

cvv_retry_count = st.number_input(
    "cvv_retry_count",
    min_value=1,
    value = 10
)


velocity_score = st.number_input(
    "velocity_score",
    min_value=1,
    value = 100
)

time_of_day_hour = st.number_input(
    "time_of_day_hour",
    min_value=1,
    value = 24
)

day_of_week = st.number_input(
    "day_of_week",
    min_value=1,
    value = 7
)

is_ai_generated_scam_attempt = st.selectbox(
    "is_ai_generated_scam_attempt",
    ["False","True"]
)

merchant_risk_score = st.number_input(
    "merchant_risk_score",
    min_value=1,
    value = 100
)

prior_disputes = st.number_input(
    "prior_disputes",
    min_value=1,
    value = 10
)

#prediction

if st.button("predict"):

#create input DataFrame

    input_data = pd.DataFrame({

"card_type" : [card_type],
"auth_method" : [auth_method],
"channel" : [channel],
"device_type" : [device_type],
"is_foreign_transaction" : [is_foreign_transaction],
"txn_count_last_24h" : [txn_count_last_24h],
"customer_age" : [customer_age],
"is_new_merchant" : [is_new_merchant],
"used_vpn" : [used_vpn],
"ip_country_mismatch" : [ip_country_mismatch],
"billing_shipping_mismatch" : [billing_shipping_mismatch],
"cvv_retry_count" : [cvv_retry_count],
"velocity_score" : [velocity_score],
"time_of_day_hour" : [time_of_day_hour],
"day_of_week" : [day_of_week],
"is_ai_generated_scam_attempt" : [is_ai_generated_scam_attempt],
"merchant_risk_score" : [merchant_risk_score],
"prior_disputes" : [prior_disputes]
})

#feature engineering
    
#feature_selection

    input_data = feature_selection(input_data)

#preprocessing
    input_processed = preprocessing_pipeline.transform(input_data)

#prediction

    prediction = model.predict(input_processed)

    probability = model.predict_proba(input_processed)

    fraud_probability = probability[0][1]*100

#display prediction

    if prediction [0]==1:
        st.error("fraud transaction")
    else:
        st.success("not fraud transaction")

    st.write(
    "fraud probability: ",
    round(fraud_probability,2),
    "%"
)

#save prediction log

    log_file  = os.path.join(
    BASE_DIR,
    "prediction_logs.csv"
)

    save_prediction(
    input_data,
    prediction [0],
    fraud_probability,
    "v1",
    log_file    
)
    
    st.success("prediction logged successfully.")