import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === MySQL connection ===
db_user = "root"
db_password = "vicky2004"
db_host = "localhost"
db_port = "3306"
db_name = "adventureworks"

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

st.title("AdventureWorks Analytics Dashboard")

# === Sidebar ===
option = st.sidebar.selectbox(
    "Select Section",
    ("Data Analysis", "Anomaly Detection", "Territory Performance", "Churn Prediction")
)

# === Helper: fetch tables ===
@st.cache_data
def fetch_table(table_name):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)

# --- Load Data ---
sales_2015 = fetch_table("sales_2015")
sales_2016 = fetch_table("sales_2016")
sales_2017 = fetch_table("sales_2017")
products = fetch_table("products")
territories = fetch_table("territories")
customers = fetch_table("customers")

# --- Combine and preprocess ---
sales = pd.concat([sales_2015, sales_2016, sales_2017], ignore_index=True)
sales = sales.merge(products[['ProductKey','ProductPrice']], on='ProductKey', how='left')

# Clean numeric fields
sales['OrderQuantity'] = pd.to_numeric(sales['OrderQuantity'], errors='coerce')
sales['ProductPrice'] = pd.to_numeric(sales['ProductPrice'].replace(r'[\$,]', '', regex=True), errors='coerce')
sales = sales.dropna(subset=['OrderQuantity','ProductPrice'])
sales['SalesAmount'] = sales['OrderQuantity'] * sales['ProductPrice']

# --- Data Analysis Section ---
if option == "Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Sales Summary")
    st.dataframe(sales.describe())
    
    st.subheader("Top 10 Products by Sales")
    top_products = sales.groupby('ProductKey')['SalesAmount'].sum().sort_values(ascending=False).head(10).reset_index()
    top_products = top_products.merge(products[['ProductKey','ProductName']], on='ProductKey', how='left')
    st.dataframe(top_products[['ProductName','SalesAmount']])
    
    st.subheader("Sales Over Time")
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])
    sales_time = sales.groupby(sales['OrderDate'].dt.to_period('M'))['SalesAmount'].sum()
    sales_time.plot(kind='line', figsize=(10,4))
    st.pyplot(plt)
    plt.clf()
    
    st.subheader("Territory Performance")
    territory_sales = sales.groupby('TerritoryKey')['SalesAmount'].sum().reset_index()
    territory_sales = territory_sales.merge(territories, left_on='TerritoryKey', right_on='SalesTerritoryKey', how='left')
    fig, ax = plt.subplots()
    sns.barplot(x='Region', y='SalesAmount', data=territory_sales, ax=ax)
    st.pyplot(fig)
    plt.clf()
    
    st.subheader("Customer Demographics")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', data=customers, ax=ax)
    st.pyplot(fig)
    plt.clf()

# --- Anomaly Detection Section ---
elif option == "Anomaly Detection":
    st.header("Anomaly Detection")
    
    iso_model = IsolationForest(contamination=0.01, random_state=42)
    iso_model.fit(sales[['OrderQuantity','SalesAmount']])
    sales['anomaly'] = iso_model.predict(sales[['OrderQuantity','SalesAmount']])
    
    st.write("### Sales with Anomalies")
    st.dataframe(sales.head(50))
    
    joblib.dump(iso_model, "anomaly_detection_model.pkl")
    st.success("Anomaly Detection Model saved as 'anomaly_detection_model.pkl'")

# --- Territory Performance Section ---
elif option == "Territory Performance":
    st.header("Territory Performance")
    
    territory_sales = sales.groupby('TerritoryKey').agg({
        'SalesAmount':'sum',
        'OrderQuantity':'sum'
    }).reset_index()
    territory_sales = territory_sales.merge(territories, left_on='TerritoryKey', right_on='SalesTerritoryKey', how='left')
    
    X_terr = pd.get_dummies(territory_sales[['Region','Country','Continent']])
    y_terr = territory_sales['SalesAmount']
    
    X_train, X_test, y_train, y_test = train_test_split(X_terr, y_terr, test_size=0.2, random_state=42)
    rf_terr = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_terr.fit(X_train, y_train)
    y_pred = rf_terr.predict(X_test)
    
    st.write("### Territory Predictions vs Actual")
    st.dataframe(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}).head(50))
    
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    
    joblib.dump(rf_terr, "territory_performance_model.pkl")
    st.success("Territory Performance Model saved as 'territory_performance_model.pkl'")

# --- Churn Prediction Section ---
elif option == "Churn Prediction":
    st.header("Churn Prediction")
    
    last_year_orders = sales_2017['CustomerKey'].unique()
    customers['churn'] = customers['CustomerKey'].apply(lambda x: 0 if x in last_year_orders else 1)
    customers['AnnualIncome'] = pd.to_numeric(customers['AnnualIncome'].replace(r'[\$,]', '', regex=True), errors='coerce')
    customers = customers.dropna(subset=['AnnualIncome'])
    
    for col in ['Gender','MaritalStatus','HomeOwner','Occupation','EducationLevel']:
        customers[col] = LabelEncoder().fit_transform(customers[col])
    
    X_churn = customers[['AnnualIncome','TotalChildren','Gender','MaritalStatus','HomeOwner','Occupation','EducationLevel']]
    y_churn = customers['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
    rf_churn = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_churn.fit(X_train, y_train)
    y_pred = np.round(rf_churn.predict(X_test))
    
    st.write("### Churn Prediction")
    st.dataframe(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}).head(50))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    
    joblib.dump(rf_churn, "churn_prediction_model.pkl")
    st.success("Churn Prediction Model saved as 'churn_prediction_model.pkl'")
