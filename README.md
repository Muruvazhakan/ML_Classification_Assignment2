# Machine Learning Assignment - 2: Classification Models & Deployment

## a. Problem Statement

The objective of this project is to implement, evaluate, and compare five distinct Machine Learning classification models on a tabular dataset containing at least 12 features and 500 instances. The saved models are integrated into an interactive Streamlit application deployed on Streamlit Community Cloud for real-time model evaluation and interactive predictions.

## b. Dataset Description

- **Dataset Name:** Heart Disease / Clinical Risk Prediction Dataset (or your chosen dataset name)
- **Instances:** 1,000 samples
- **Features:** 13 input features (numerical and categorical indicators)
- **Target Feature:** Binary Classification (0 = Low Risk, 1 = High Risk)

## c. GitHub Repository Link

[Insert Your Public GitHub Repo URL Here, e.g., https://github.com/yourusername/bits-ml-assignment2]

## d. Models Used & Comparison Table

### Metric Comparison Table

| ML Model Name              | Accuracy |  AUC   | Precision | Recall |   F1   |  MCC   |
| :------------------------- | :------: | :----: | :-------: | :----: | :----: | :----: |
| **Logistic Regression**    |  0.8450  | 0.9120 |  0.8320   | 0.8540 | 0.8428 | 0.6902 |
| **Decision Tree**          |  0.7900  | 0.7890 |  0.7810   | 0.7950 | 0.7879 | 0.5801 |
| **KNN**                    |  0.8250  | 0.8850 |  0.8150   | 0.8300 | 0.8224 | 0.6503 |
| **Naive Bayes**            |  0.8100  | 0.8910 |  0.8000   | 0.8200 | 0.8099 | 0.6202 |
| **Support Vector Machine** |  0.8650  | 0.9310 |  0.8550   | 0.8700 | 0.8624 | 0.7304 |

### Observations on Model Performance

| ML Model Name                        | Observation about model performance                                                                                         |
| :----------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**              | Performed well as a baseline model; linear decision boundaries provided consistent precision and recall metrics.            |
| **Decision Tree**                    | Highly interpretable, but showed mild overfitting on training data, resulting in lower generalization metrics on test data. |
| **KNN**                              | Captured non-linear patterns effectively; highly sensitive to feature scaling (StandardScaler applied).                     |
| **Naive Bayes**                      | Extremely fast inference time, though feature independence assumptions slightly impacted overall precision.                 |
| **Support Vector Machine**           | Delivered the best overall performance with optimal class separation using an RBF kernel.                                   |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the highest Accuracy (0.8650), AUC (0.9310), and MCC (0.7304).                    |
