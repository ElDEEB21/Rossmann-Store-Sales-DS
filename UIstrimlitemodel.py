import streamlit as st
from model_pipeline import Predict
import pandas as pd
def UI_Prediction():
        # Streamlit App
        st.title("Sales Prediction App")

        # Collect input values from the user
        store = st.number_input("Store", min_value=1, step=1, value=2)
        day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, step=1, value=5)
        date = st.date_input("Date", value=pd.to_datetime("2015-07-31"))
        # sales = st.number_input("Sales", min_value=0.0, value=6064.0)
        customers = st.number_input("Customers", min_value=0, value=625)
        open_store = st.number_input("Open (1 = Open, 0 = Closed)", min_value=0, max_value=1, step=1, value=1)
        promo = st.number_input("Promo (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1, value=1)
        state_holiday = st.selectbox("State Holiday", options=[0, 'a', 'b', 'c'])
        school_holiday = st.number_input("School Holiday (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1, value=1)
        store_type = st.selectbox("Store Type", options=['a', 'b', 'c', 'd'])
        assortment = st.selectbox("Assortment", options=['a', 'b', 'c'])
        competition_distance = st.number_input("Competition Distance", min_value=0.0, value=570.0)
        competition_open_month = st.number_input("Competition Open Since Month", min_value=1, max_value=12, step=1, value=11)
        competition_open_year = st.number_input("Competition Open Since Year", min_value=1900, value=2007, step=1)
        promo2 = st.number_input("Promo2 (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1, value=1)
        promo2_since_week = st.number_input("Promo2 Since Week", min_value=1, max_value=52, step=1, value=13)
        promo2_since_year = st.number_input("Promo2 Since Year", min_value=1900, value=2010, step=1)
        promo_interval = st.text_input("Promo Interval", value="Jan,Apr,Jul,Oct")
        # Prepare input data
        values = [
            store,day_of_week, date,customers, open_store, promo, state_holiday, school_holiday, store_type,
            assortment, competition_distance, competition_open_month, competition_open_year, promo2, promo2_since_week,
            promo2_since_year, promo_interval
        ]

        if st.button("Predict"):
            # Instantiate the prediction class and run the model
                model_pipline=Predict(values)
                data=model_pipline.preporcessing()
                data=model_pipline.feature_enginnering(data)
                data=model_pipline.pipline(data)
                        # print(data.shape,data)
                prectict_model=model_pipline.prectict_model(data)
                print(prectict_model)
            # Display the prediction
                st.success(f"Predicted Sales: {prectict_model[0]:,.2f}")
if __name__ =="__main__":
    UI_Prediction()