import pandas as pd            # For loading and handling data
import seaborn as sns           # For creating beautiful graphs
import matplotlib.pyplot as plt # For showing the graphs

# Step 1 — Load the Iris CSV file
file_path = r"C:\Users\hemal\OneDrive\Pictures\Desktop\vs code\Iris.csv"
data = pd.read_csv(file_path)

# Step 2 — Show the first few rows (just to confirm it's loaded)
print(data.head())

# Step 3 — Create a scatter plot of Sepal Length vs Sepal Width
sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=data)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Step 4 — Create a pairplot (plots all features vs each other)
sns.pairplot(data, hue="Species")
plt.show()

# Step 5 — Create a boxplot of Petal Length by Species
sns.boxplot(x="Species", y="PetalLengthCm", data=data)
plt.title("Petal Length by Species")
plt.show()
