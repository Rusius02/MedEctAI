import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
# plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="h", palette="coolwarm")
plt.title("Boxplots des variables du dataset")
# plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
#plt.show()

X = df.drop(columns=["Outcome"])

# Initialisation du modèle Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% d'anomalies
df["Anomaly"] = iso_forest.fit_predict(X)

# Le modèle marque les anomalies avec -1, on les filtre
anomalies = df[df["Anomaly"] == -1]

print(f"🔹 Nombre total d'anomalies détectées : {len(anomalies)}")
print(anomalies.head())  # Aperçu des anomalies


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Glucose", y="Insulin", hue="Anomaly", palette={1: "blue", -1: "red"})
plt.title("Détection d'anomalies (Isolation Forest)")
plt.legend(["Normaux", "Anomalies"])
# plt.show()

print(df.groupby(["Anomaly", "Outcome"]).size())


# Séparer les features (X) et la cible (y)
X = df.drop(columns=["Outcome", "Anomaly"])  # On enlève "Anomaly" car ce n'est pas une feature d'origine
y = df["Outcome"]

# Séparer en données d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Taille du dataset d'entraînement : {X_train.shape}")
print(f"Taille du dataset de test : {X_test.shape}")

# Initialiser et entraîner le modèle
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prédictions
y_pred = rf.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Précision du modèle : {accuracy:.2f}")

# Rapport détaillé des performances
print("\n🔹 Rapport de classification :")
print(classification_report(y_test, y_pred))