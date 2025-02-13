import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Configuration de la page Streamlit ---
st.title("Analyse et Prédiction de Données avec IA")
st.sidebar.header("Paramètres")

# --- Chargement des données ---
uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Aperçu des données :")
    st.write(df.head())

    # --- Détection de valeurs aberrantes ---
    st.write("### Détection d'anomalies avec Isolation Forest")
    contamination = st.sidebar.slider("Taux d'anomalies (Isolation Forest)", 0.01, 0.2, 0.05, 0.01)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df["Anomaly"] = iso_forest.fit_predict(df.select_dtypes(include=['number']))
    st.write(f"Nombre d'anomalies détectées: {(df['Anomaly'] == -1).sum()}")

    # --- Visualisation ---
    feature_x = st.sidebar.selectbox("Axe X", df.columns, index=1)
    feature_y = st.sidebar.selectbox("Axe Y", df.columns, index=2)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=feature_x, y=feature_y, hue="Anomaly", palette={1: "blue", -1: "red"})
    plt.title("Détection d'anomalies")
    st.pyplot(plt)

    # --- Modèle de Machine Learning ---
    if "Outcome" in df.columns:
        st.write("### Prédiction du diabète avec Random Forest")
        X = df.drop(columns=["Outcome", "Anomaly"], errors='ignore')
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if st.sidebar.checkbox("Utiliser SMOTE pour équilibrer les classes ?"):
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle : {accuracy:.2f}")
        st.text(classification_report(y_test, y_pred))

# Pour lancer l'application : dans le terminal -> streamlit run mon_fichier.py
