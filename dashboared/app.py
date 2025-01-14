import streamlit as st
import json
import requests

st.title('predicting sales based on given value')
# Streamlit app

# Streamlit app


st.title("User Input Form")

st.header("Fill in the details below:")
# Create a form
with st.form("user_input_form"):
    # Input fields
    store = st.number_input("Store ID", min_value=1, step=1)
    day_of_week = st.selectbox(
        "Day of Week (1=Monday, 7=Sunday)", range(1, 8))
    sales = st.number_input("Sales", min_value=0, step=1)
    customers = st.number_input("Customers", min_value=0, step=1)
    open_status = st.checkbox("Store Open", value=True)
    promo = st.number_input("Promo", min_value=0, step=1)
    state_holiday = st.checkbox("State Holiday", value=False)
    school_holiday = st.checkbox("School Holiday", value=False)
    store_type = st.selectbox("Store Type", options=["a", "b", "c", "d"])
    assortment = st.selectbox("Assortment", options=["a", "b", "c"])
    competition_distance = st.number_input(
        "Competition Distance (meters)", min_value=0.0, step=0.1)
    competition_open_month = st.number_input(
        "Competition Open Since Month", min_value=1.0, max_value=12.0, step=1.0)
    competition_open_year = st.number_input(
        "Competition Open Since Year", min_value=1900.0, step=1.0)
    promo2 = st.number_input("Promo2", min_value=0, max_value=1, step=1)
    promo2_since_week = st.number_input(
        "Promo2 Since Week", min_value=1.0, max_value=52.0, step=1.0)
    promo2_since_year = st.number_input(
        "Promo2 Since Year", min_value=1900.0, step=1.0)
    promo_interval = st.selectbox("Promo Interval", options=[0, 1, 2, 3])

    # Submit button
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Form submitted successfully!")
    user_input = {
        "Store": store,
        "DayOfWeek": day_of_week,
        "Sales": sales,
        "Customers": customers,
        "Open": open_status,
        "Promo": promo,
        "StateHoliday": state_holiday,
        "SchoolHoliday": school_holiday,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": competition_distance,
        "CompetitionOpenSinceMonth": competition_open_month,
        "CompetitionOpenSinceYear": competition_open_year,
        "Promo2": promo2,
        "Promo2SinceWeek": promo2_since_week,
        "Promo2SinceYear": promo2_since_year,
        "PromoInterval": promo_interval
    }
    inputs = st.json(user_input)  # Display the input in JSON format
    st.session_state.inputs = inputs

if st.button('Predict'):
    inputs = st.session_state.inputs
    res = requests.post(
        url='http://127.0.0.1:8000/predict', data=json.dumps(inputs))
    st.subheader(
        f'Response from API call (Predicted result) is = {res.text}')
