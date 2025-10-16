import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel(r"C:\Users\hemal\OneDrive\Documents\unemployment.xlsm")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Convert Date column to datetime (force errors to NaT)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where Date is missing
df = df.dropna(subset=['Date'])

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Line graph of Unemployment Rate over Time
axes[0].plot(df['Date'], df['Estimated Unemployment Rate (%)'], color='blue', marker='o')
axes[0].set_title("Unemployment Rate Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Unemployment Rate (%)")
axes[0].tick_params(axis='x', rotation=45)

# 2. Bar chart of average Unemployment Rate by Region
avg_region = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
avg_region.plot(kind='bar', color='orange', ax=axes[1])
axes[1].set_title("Average Unemployment Rate by Region")
axes[1].set_ylabel("Unemployment Rate (%)")

plt.tight_layout()
plt.show()
