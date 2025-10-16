import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Dataset
file_path = r"C:\Users\hemal\OneDrive\Documents\task3.xlsx"
df = pd.read_excel(file_path)

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 2. Data Preprocessing
df = df.drop_duplicates()
df = df.dropna()  # Drop missing values

# Encode categorical columns safely
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nAfter Encoding Categorical Columns:")
print(df.head())

# 3. Split into features & target
X = df.drop("Selling_Price", axis=1)  # Assuming Selling_Price is target
y = df["Selling_Price"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train Models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 5. Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n{model_name} Evaluation:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# 6a. Visualization - Line Graph
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual Price", color="blue", marker='o')
plt.plot(y_pred_rf, label="Predicted Price", color="red", marker='x')
plt.xlabel("Car Index")
plt.ylabel("Selling Price")
plt.title("Actual vs Predicted Car Price (Random Forest) - Line Graph")
plt.legend()
plt.tight_layout()
plt.savefig("car_price_line_plot.png")
plt.show()

