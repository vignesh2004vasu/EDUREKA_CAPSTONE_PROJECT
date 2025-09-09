import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- MySQL connection ---
engine = create_engine("mysql+pymysql://root:password@localhost:3306/adventureworks")

# --- Fetch data ---
sales_2017 = pd.read_sql("SELECT * FROM sales_2017", engine)
customers = pd.read_sql("SELECT * FROM customers", engine)

# --- Define churn ---
last_year_orders = sales_2017['CustomerKey'].unique()
customers['churn'] = customers['CustomerKey'].apply(lambda x: 0 if x in last_year_orders else 1)

# --- Clean and encode ---
customers['AnnualIncome'] = customers['AnnualIncome'].replace('[\$,]', '', regex=True).astype(float)
for col in ['Gender','MaritalStatus','HomeOwner','Occupation','EducationLevel']:
    customers[col] = LabelEncoder().fit_transform(customers[col])

# --- Features and target ---
X_churn = customers[['AnnualIncome','TotalChildren','Gender','MaritalStatus','HomeOwner','Occupation','EducationLevel']]
y_churn = customers['churn']

X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
rf_churn = RandomForestRegressor(n_estimators=100, random_state=42)
rf_churn.fit(X_train, y_train)
y_pred_churn = round(rf_churn.predict(X_test))
print("Churn Prediction Accuracy:", accuracy_score(y_test, y_pred_churn))

# --- Save model ---
joblib.dump(rf_churn, "churn_prediction_model.pkl")
print("Churn Prediction Model saved as 'churn_prediction_model.pkl'")
