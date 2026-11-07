import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

from feature_utils import normalize_model_input

st.set_page_config(page_title="ML Assignment 2 - Classification Dashboard", layout="wide")

st.title("📊 Machine Learning Classification & Evaluation Dashboard")
st.write("BITS Pilani - M.Tech AIML (Machine Learning Assignment 2)")

# Sidebar File Upload & Selection
st.sidebar.header("User Configurations")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Model Path Mapping
model_dict = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dict = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "logistic_regression.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "model", "decision_tree.pkl"),
    "KNN": os.path.join(BASE_DIR, "model", "knn.pkl"),
    "Naive Bayes": os.path.join(BASE_DIR, "model", "naive_bayes.pkl"),
    "Support Vector Machine": os.path.join(BASE_DIR, "model", "svm.pkl")
}

target_col = st.sidebar.text_input("Target Column Name", value="quality")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(df.head())

    if target_col in df.columns:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]

        # Load selected model
        try:
            model = joblib.load(model_dict[model_choice])
            scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

            # Normalize/align the uploaded feature matrix against the trained model schema.
            X_test_prepared = normalize_model_input(model, X_test, scaler=scaler)

            # Predict using the aligned matrix instead of failing on the feature-length mismatch.
            y_pred = model.predict(X_test_prepared)
            y_proba = model.predict_proba(X_test_prepared)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

            # Display Metrics
            st.subheader(f"Evaluation Metrics: {model_choice}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 3))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading or executing model: {e}")
    else:
        st.warning(f"Target column '{target_col}' not found in the uploaded CSV.")
else:
    st.info("Please upload a test dataset CSV file to calculate metrics.")