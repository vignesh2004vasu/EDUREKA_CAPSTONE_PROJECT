# === Step 1: Setup ===
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import joblib

# === Step 2: Connect to MySQL ===
# Update user, password, host, port, database accordingly
db_user = "root"
db_password = "vicky2004"
db_host = "localhost"
db_port = "3306"
db_name = "adventureworks"

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# === Step 3: Fetch Tables ===
sales_2015 = pd.read_sql("SELECT * FROM sales_2015", engine)
sales_2016 = pd.read_sql("SELECT * FROM sales_2016", engine)
sales_2017 = pd.read_sql("SELECT * FROM sales_2017", engine)
products = pd.read_sql("SELECT * FROM products", engine)

# === Step 4: Combine sales and merge with products ===
sales = pd.concat([sales_2015, sales_2016, sales_2017], ignore_index=True)
sales = sales.merge(products, on="ProductKey", how="left")

# === Step 5: Feature Engineering ===
sales["SalesAmount"] = sales["OrderQuantity"] * sales["ProductPrice"]

X = sales[["OrderQuantity", "ProductPrice", "ProductCost"]]
y = sales["SalesAmount"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 6: Linear Regression ===
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)
lin_mae = mean_absolute_error(y_test, lin_preds)
print("Linear Regression MAE:", lin_mae)

# === Step 7: Random Forest with GridSearch ===
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    rf,
    param_grid,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# Best RF Model
best_rf = grid.best_estimator_
rf_preds = best_rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
print("Tuned Random Forest MAE:", rf_mae)

# === Step 8: Choose Best Model ===
if rf_mae < lin_mae:
    best_model = best_rf
    print("✅ Tuned Random Forest selected")
else:
    best_model = lin_model
    print("✅ Linear Regression selected")

# === Step 9: Save Best Model ===
joblib.dump(best_model, "sales_forecast_model.pkl")
print("Model saved as sales_forecast_model.pkl")
