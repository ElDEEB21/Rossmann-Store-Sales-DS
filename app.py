import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
import UIstrimlitemodel
from model_pipeline import Predict
df = pd.read_csv(r"D:\model\data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', drop=False, inplace=True)

def EDA_Dashboard():
    st.title("📊 Exploratory Data Analysis")
    
    plots = [
        "Overall Sales Trend Over Time (Monthly)",
        "Average Sales by Store Type",
        "Average Sales With and Without Promotion",
        "Competition Distance vs Sales (Sampled)",
        "Average Number of Customers by Day of the Week",
        "Correlation of Factors with Number of Customers",
        "Average Number of Customers with and without Promotions",
        "Impact of Holidays on Sales",
        "Effect of Promo2 on Sales",
        "Sales Based on Assortment Type",
        "Sales by Year of Competition Opening",
        "Sales Data Distribution",
        "Average Sales by Month (Seasonal Analysis)",
        "Yearly Sales Trend",
        "Average Sales During Promotional Intervals",
        "Customer Count vs Total Sales by Store Type",
        "Average Competition Distance by Store Type",
    ]
    
    choice = st.selectbox("Choose a Plot:", plots)

    if choice == "Overall Sales Trend Over Time (Monthly)":
        monthly_sales = df['Sales'].resample('M').sum()
        fig = px.line(monthly_sales, labels={'value':'Total Sales', 'Date':'Date'}, 
                      title='Overall Sales Trend Over Time (Monthly Aggregated)')
        fig.update_layout(autosize=True, height=600, template='plotly_white')
        st.plotly_chart(fig)

    elif choice == "Average Sales by Store Type":
        store_type_sales = df.groupby('StoreType')['Sales'].mean()
        fig = px.bar(store_type_sales, labels={'value':'Average Sales', 'StoreType':'Store Type'}, 
                     title='Average Sales by Store Type', color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig)

    elif choice == "Average Sales With and Without Promotion":
        promo_sales = df.groupby('Promo')['Sales'].mean()
        fig = px.bar(promo_sales, labels={'value':'Average Sales', 'Promo':'Promotion (1=Yes, 0=No)'}, 
                     title='Average Sales With and Without Promotion', color_discrete_sequence=['orange', 'blue'])
        st.plotly_chart(fig)

    elif choice == "Competition Distance vs Sales (Sampled)":
        sampled_df = df.sample(5000)
        fig = px.scatter(sampled_df, x='CompetitionDistance', y='Sales', opacity=0.3, 
                         labels={'CompetitionDistance':'Competition Distance', 'Sales':'Sales'}, 
                         title='Competition Distance vs Sales')
        st.plotly_chart(fig)

    elif choice == "Average Number of Customers by Day of the Week":
        customers_per_day = df.groupby('DayOfWeek')['Customers'].mean()
        fig = px.bar(customers_per_day, labels={'value':'Average Customers', 'DayOfWeek':'Day of the Week'}, 
                     title='Average Number of Customers by Day of the Week', color_discrete_sequence=['red'])
        st.plotly_chart(fig)

    elif choice == "Correlation of Factors with Number of Customers":
        correlation = df[['Sales', 'Promo', 'SchoolHoliday', 'Customers']].corr()['Customers'].sort_values(ascending=False)
        fig = px.bar(correlation, labels={'value':'Correlation with Customers'}, title='Correlation of Factors with Number of Customers', color_discrete_sequence=['teal'])
        st.plotly_chart(fig)

    elif choice == "Average Number of Customers with and without Promotions":
        promo_customer = df.groupby('Promo')['Customers'].mean()
        fig = px.bar(promo_customer, labels={'value':'Average Customers', 'Promo':'Promotion (1=Yes, 0=No)'}, 
                     title='Average Number of Customers with and without Promotions', color_discrete_sequence=['cyan', 'magenta'])
        st.plotly_chart(fig)

    elif choice == "Impact of Holidays on Sales":
        holiday_sales = df.groupby('SchoolHoliday')['Sales'].mean()
        fig = px.bar(holiday_sales, labels={'value':'Average Sales', 'SchoolHoliday':'Holiday (1=Yes, 0=No)'}, 
                     title='Impact of Holidays on Sales', color_discrete_sequence=['lime'])
        st.plotly_chart(fig)

    elif choice == "Effect of Promo2 on Sales":
        promo2_sales = df.groupby('Promo2')['Sales'].mean()
        fig = px.bar(promo2_sales, labels={'value':'Average Sales', 'Promo2':'Promo2 (1=Yes, 0=No)'}, 
                     title='Effect of Promo2 on Sales', color_discrete_sequence=['yellow'])
        st.plotly_chart(fig)

    elif choice == "Sales Based on Assortment Type":
        assortment_sales = df.groupby('Assortment')['Sales'].mean()
        fig = px.bar(assortment_sales, labels={'value':'Average Sales', 'Assortment':'Assortment Type'}, 
                     title='Sales Based on Assortment Type', color_discrete_sequence=['pink'])
        st.plotly_chart(fig)

    elif choice == "Sales by Year of Competition Opening":
        competition_year_sales = df.groupby('CompetitionOpenSinceYear')['Sales'].mean().dropna()
        fig = px.bar(competition_year_sales, labels={'value':'Average Sales', 'CompetitionOpenSinceYear':'Year of Competition Opening'}, 
                     title='Sales by Year of Competition Opening', color_discrete_sequence=['orange'])
        st.plotly_chart(fig)

    elif choice == "Sales Data Distribution":
        fig = px.histogram(df, x='Sales', nbins=50, title='Sales Data Distribution', labels={'Sales':'Sales'}, color_discrete_sequence=['blue'])
        st.plotly_chart(fig)

    elif choice == "Average Sales by Month (Seasonal Analysis)":
        df['Month'] = df.index.month
        monthly_sales = df.groupby('Month')['Sales'].mean()
        fig = px.bar(monthly_sales, labels={'value':'Average Sales', 'Month':'Month'}, 
                     title='Average Sales by Month (Seasonal Analysis)', color_discrete_sequence=['purple'])
        st.plotly_chart(fig)

    elif choice == "Yearly Sales Trend":
        df['Year'] = df.index.year
        yearly_sales = df.groupby('Year')['Sales'].sum()
        fig = px.line(yearly_sales, labels={'value':'Total Sales', 'Year':'Year'}, 
                      title='Yearly Sales Trend', color_discrete_sequence=['blue'])
        st.plotly_chart(fig)

    elif choice == "Average Sales During Promotional Intervals":
        promo_sales = df.groupby(['PromoInterval'])['Sales'].mean().dropna()
        fig = px.bar(promo_sales, labels={'value':'Average Sales', 'PromoInterval':'Promotion Interval'}, 
                     title='Average Sales During Promotional Intervals', color_discrete_sequence=['cyan'])
        st.plotly_chart(fig)

    elif choice == "Customer Count vs Total Sales by Store Type":
        store_type_sales_customers = df.groupby('StoreType').agg({'Sales':'sum', 'Customers':'mean'})
        fig = go.Figure()
        fig.add_trace(go.Bar(x=store_type_sales_customers.index, y=store_type_sales_customers['Sales'], name='Total Sales', marker_color='blue'))
        fig.add_trace(go.Bar(x=store_type_sales_customers.index, y=store_type_sales_customers['Customers'], name='Average Customers', marker_color='red'))
        fig.update_layout(barmode='group', title='Customer Count vs Total Sales by Store Type')
        st.plotly_chart(fig)

    elif choice == "Average Competition Distance by Store Type":
        competition_distance = df.groupby('StoreType')['CompetitionDistance'].mean()
        fig = px.bar(competition_distance, labels={'value':'Average Competition Distance', 'StoreType':'Store Type'}, 
                     title='Average Competition Distance by Store Type', color_discrete_sequence=['green'])
        st.plotly_chart(fig)
    
    elif choice == "Monthly Sales Trends Over the Years":
        df['Month'] = df.index.month
        monthly_sales_trend = df.groupby(['Year', 'Month'])['Sales'].sum().unstack()
        fig = px.imshow(monthly_sales_trend, aspect="auto", 
                        labels={'color':'Total Sales'}, 
                        title='Monthly Sales Trends Over the Years')
        st.plotly_chart(fig)

    elif choice == "Sales Performance: Weekdays vs Weekends":
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)
        sales_weekday_vs_weekend = df.groupby('IsWeekend')['Sales'].mean()
        fig = px.bar(sales_weekday_vs_weekend, labels={'value':'Average Sales', 'IsWeekend':'Day (0=Weekday, 1=Weekend)'}, 
                     title='Sales Performance: Weekdays vs Weekends', color_discrete_sequence=['lightblue', 'orange'])
        st.plotly_chart(fig)

    elif choice == "Customer Count Trend Over the Years":
        yearly_customers = df.groupby('Year')['Customers'].mean()
        fig = px.line(yearly_customers, labels={'value':'Average Customers', 'Year':'Year'}, 
                      title='Customer Count Trend Over the Years', color_discrete_sequence=['darkgreen'])
        st.plotly_chart(fig)

    elif choice == "Total Sales by Day of the Week":
        sales_by_day = df.groupby('DayOfWeek')['Sales'].sum()
        fig = px.bar(sales_by_day, labels={'value':'Total Sales', 'DayOfWeek':'Day of the Week'}, 
                     title='Total Sales by Day of the Week', color_discrete_sequence=['purple'])
        st.plotly_chart(fig)

    elif choice == "Average Sales Based on Promotion Combinations (Promo & Promo2)":
        promo_combination_sales = df.groupby(['Promo', 'Promo2'])['Sales'].mean().unstack()
        fig = px.imshow(promo_combination_sales, labels={'color':'Average Sales'}, 
                        title='Average Sales Based on Promotion Combinations (Promo & Promo2)')
        st.plotly_chart(fig)

    elif choice == "Customer Count Trend Before and During Promotions":
        df['PromoLag'] = df.groupby('Store')['Promo'].shift(1)
        customer_count_trend = df.groupby('PromoLag')['Customers'].mean().dropna()
        fig = px.bar(customer_count_trend, labels={'value':'Average Customers', 'PromoLag':'Promo Previous Day (1=Yes, 0=No)'}, 
                     title='Customer Count Trend Before and During Promotions', color_discrete_sequence=['pink'])
        st.plotly_chart(fig)

    elif choice == "Average Sales During School Holidays":
        holiday_sales = df.groupby('SchoolHoliday')['Sales'].mean()
        fig = px.bar(holiday_sales, labels={'value':'Average Sales', 'SchoolHoliday':'Holiday (1=Yes, 0=No)'}, 
                     title='Average Sales During School Holidays', color_discrete_sequence=['blue', 'red'])
        st.plotly_chart(fig)

def Model_Usage():
    # # Streamlit App
    UIstrimlitemodel.UI_Prediction()

def main():
    st.title("Sales Analysis and Prediction Dashboard")

    st.markdown("""
        <style>
        .stRadio > div {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stRadio label {
            margin-right: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    page = st.radio("Choose a Page", ["EDA Dashboard", "Model Usage"], index=0, horizontal=True)

    if page == "EDA Dashboard":
        print(1)
        EDA_Dashboard()
    elif page == "Model Usage":
        print(2)

        Model_Usage()

if __name__ == "__main__":
    main()
