import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --- MySQL connection ---
engine = create_engine("mysql+pymysql://root:password@localhost:3306/adventureworks")

# --- Fetch data ---
sales_2015 = pd.read_sql("SELECT * FROM sales_2015", engine)
sales_2016 = pd.read_sql("SELECT * FROM sales_2016", engine)
sales_2017 = pd.read_sql("SELECT * FROM sales_2017", engine)
products = pd.read_sql("SELECT * FROM products", engine)
territories = pd.read_sql("SELECT * FROM territories", engine)

# --- Combine and preprocess ---
sales = pd.concat([sales_2015, sales_2016, sales_2017], ignore_index=True)
sales = sales.merge(products[['ProductKey', 'ProductPrice']], on='ProductKey', how='left')
sales['ProductPrice'] = sales['ProductPrice'].replace('[\$,]', '', regex=True).astype(float)
sales['SalesAmount'] = sales['OrderQuantity'] * sales['ProductPrice']

# --- Aggregate per territory ---
territory_sales = sales.groupby('TerritoryKey').agg({
    'SalesAmount': 'sum',
    'OrderQuantity': 'sum'
}).reset_index()
territory_sales = territory_sales.merge(territories, left_on='TerritoryKey', right_on='SalesTerritoryKey', how='left')

# --- Features and target ---
X_terr = pd.get_dummies(territory_sales[['Region','Country','Continent']])
y_terr = territory_sales['SalesAmount']

X_train, X_test, y_train, y_test = train_test_split(X_terr, y_terr, test_size=0.2, random_state=42)
rf_terr = RandomForestRegressor(n_estimators=100, random_state=42)
rf_terr.fit(X_train, y_train)
y_pred_terr = rf_terr.predict(X_test)
print("Territory Performance MAE:", mean_absolute_error(y_test, y_pred_terr))

# --- Save model ---
joblib.dump(rf_terr, "territory_performance_model.pkl")
print("Territory Performance Model saved as 'territory_performance_model.pkl'")
