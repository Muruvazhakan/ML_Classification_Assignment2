import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

def main():
    # 1. Configuration: Path to your training CSV file
    DATASET_PATH = "bank.csv"  # Replace with any CSV filename you want to train on
    
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file '{DATASET_PATH}' not found in current directory.")
        return

    # 2. Read Dataset (Detects both ',' and ';' delimiters automatically)
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(DATASET_PATH, sep=',')
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    print(f"Dataset successfully loaded. Shape: {df.shape}")

    # 3. Dynamic Feature & Target Separation
    # Assumes the target is the last column if 'target' or 'y' isn't explicitly named
    if 'y' in df.columns:
        target_col = 'y'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    print(f"Target Column Identified: '{target_col}'")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode Target Variable if it is text/categorical
    target_encoder = None
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        joblib.dump(target_encoder, os.path.join("model", "target_encoder.pkl"))

    # 4. Preprocess Feature Data (Encodes all categorical text features dynamically)
    feature_encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le
    
    joblib.dump(feature_encoders, os.path.join("model", "feature_encoders.pkl"))

    # 5. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Save Preprocessed Test Data for Streamlit upload testing
    test_export = pd.concat([X_test, pd.Series(y_test, name=target_col, index=X_test.index)], axis=1)
    test_export.to_csv("bank.csv", index=False)
    print("Saved bank.csv successfully.")

    # 6. Global Feature Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join("model", "scaler.pkl"))

    # Determine if classification is Binary or Multi-Class
    is_multiclass = len(np.unique(y)) > 2
    average_type = 'weighted' if is_multiclass else 'binary'

    # 7. Define the 5 Classifier Models
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True, "logistic_regression.pkl"),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False, "decision_tree.pkl"),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True, "knn.pkl"),
        "Naive Bayes": (GaussianNB(), False, "naive_bayes.pkl"),
        "Support Vector Machine": (SVC(probability=True, random_state=42), True, "svm.pkl")
    }

    print("\n" + "="*95)
    print(f"{'ML Model Name':<23} | {'Accuracy':<8} | {'AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'MCC':<8}")
    print("-" * 95)

    # 8. Train, Evaluate & Export Models
    for name, (model, use_scaled, file_name) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        # Determine ROC-AUC compatibility
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)
            if is_multiclass:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = 0.0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=average_type, zero_division=0)
        rec = recall_score(y_test, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        print(f"{name:<23} | {acc:.4f}   | {auc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {mcc:.4f}")

        joblib.dump(model, os.path.join("model", file_name))

    print("="*95)
    print("All 5 models trained and saved to 'model/' directory successfully!\n")

if __name__ == "__main__":
    main()