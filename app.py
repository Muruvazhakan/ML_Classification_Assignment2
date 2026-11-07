import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "bank.csv")

st.set_page_config(page_title="Universal ML Dashboard", layout="wide")

st.title("📊 Universal Machine Learning Classification Dashboard")
st.write("BITS Pilani - M.Tech AIML (Machine Learning Assignment 2)")
st.write("Student: Muruvazhakan Muthuvelu | Roll No: 2026ak05426")

# Sidebar Controls
st.sidebar.header("1. Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Optional CSV Dataset", type=["csv"])
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)
target_input = st.sidebar.text_input("Target Column Name (Optional)", value="")
dataset_name = st.sidebar.text_input("Dataset Name", value="")

@st.cache_data(show_spinner="Processing dataset and training 5 ML models...")
def process_and_train(df_raw, user_target_col):
    df = df_raw.copy()
    
    # 1. Identify Target Column
    if user_target_col and user_target_col in df.columns:
        target_col = user_target_col
    elif 'y' in df.columns:
        target_col = 'y'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    # 2. Encode Target if categorical/string
    y = df[target_col]
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    X = df.drop(columns=[target_col])

    # 3. Encode Categorical Feature Columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Fit the 5 ML Classifiers
    trained_models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_train, y_train), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train), True),
        "Naive Bayes": (GaussianNB().fit(X_train, y_train), False),
        "Support Vector Machine": (SVC(probability=True, random_state=42).fit(X_train_scaled, y_train), True)
    }

    test_data_dict = {
        "X_test_raw": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "target_col": target_col
    }

    return trained_models, test_data_dict

# Logic: Load browser upload if available, otherwise fallback to local bank.csv
df = None
data_source_label = ""

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) <= 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
        data_source_label = dataset_name.strip() if dataset_name and dataset_name.strip() else f"Uploaded CSV ({uploaded_file.name})"
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()
elif os.path.exists(DEFAULT_CSV):
    try:
        df = pd.read_csv(DEFAULT_CSV, sep=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(DEFAULT_CSV, sep=',')
        data_source_label = "Default Local Dataset (bank.csv)"
    except Exception as e:
        st.error(f"Error reading default bank.csv: {e}")
        st.stop()
else:
    st.warning("No default 'bank.csv' found in directory and no file uploaded. Please upload a dataset.")
    st.stop()

if df is not None:
    st.header(f"Active Data Source: **{data_source_label}**")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Process dataset and train models dynamically
    models, test_dict = process_and_train(df, target_input)
    st.success(f"Models evaluated against target column: **'{test_dict['target_col']}'**")

    # Extract chosen model & evaluate
    model, is_scaled = models[model_choice]
    X_eval = test_dict["X_test_scaled"] if is_scaled else test_dict["X_test_raw"]
    y_test = test_dict["y_test"]

    y_pred = model.predict(X_eval)

    is_multiclass = len(np.unique(y_test)) > 2
    avg_type = 'weighted' if is_multiclass else 'binary'

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_eval)
        except Exception:
            y_proba = None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc_display = "N/A"
    if y_proba is not None:
        try:
            if is_multiclass:
                auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                auc_val = roc_auc_score(y_test, y_proba[:, 1])
            auc_display = f"{auc_val:.4f}"
        except Exception:
            auc_display = "N/A"

    # Display Metrics Cards
    st.subheader(f"Evaluation Metrics: {model_choice}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", auc_display)
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC Score", f"{mcc:.4f}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)