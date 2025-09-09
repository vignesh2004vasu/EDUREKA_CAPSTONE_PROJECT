import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import joblib

# --- MySQL connection ---
engine = create_engine("mysql+pymysql://root:password@localhost:3306/adventureworks")

# --- Fetch sales tables ---
sales_2015 = pd.read_sql("SELECT * FROM sales_2015", engine)
sales_2016 = pd.read_sql("SELECT * FROM sales_2016", engine)
sales_2017 = pd.read_sql("SELECT * FROM sales_2017", engine)
products = pd.read_sql("SELECT * FROM products", engine)

# --- Combine and preprocess ---
sales = pd.concat([sales_2015, sales_2016, sales_2017], ignore_index=True)
sales = sales.merge(products[['ProductKey', 'ProductPrice']], on='ProductKey', how='left')
sales['ProductPrice'] = sales['ProductPrice'].replace('[\$,]', '', regex=True).astype(float)
sales['SalesAmount'] = sales['OrderQuantity'] * sales['ProductPrice']

# --- Train IsolationForest ---
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(sales[['OrderQuantity', 'SalesAmount']])

# --- Save model ---
joblib.dump(iso, "anomaly_detection_model.pkl")
print("Anomaly Detection Model saved as 'anomaly_detection_model.pkl'")
