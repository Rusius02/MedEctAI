import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

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


z_scores = df.drop(columns=["Outcome"]).apply(zscore)

# Trouver les valeurs aberrantes (Z-score > 3)
outliers = (z_scores.abs() > 3).sum()
print("🔹 Nombre de valeurs aberrantes par colonne :")
print(outliers)

# Visualisation : Ages plot
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Distribution de l'âge des patients")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="h", palette="coolwarm")
plt.title("Boxplots des variables du dataset")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

