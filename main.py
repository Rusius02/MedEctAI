import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes


df = pd.read_csv("diabetes.csv")

# Display the first rows
print("🔹 First rows :")
print(df.head())

# Information about the dataset
print("\n🔹 Info from dataset :")
print(df.info())

# Statistics
print("\n🔹 Statistics :")
print(df.describe())

# Check missed values
print("\n🔹 Missed values :")
print(df.isnull().sum())

# Visualisation : Ages plot
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Distribution de l'âge des patients")
plt.show()