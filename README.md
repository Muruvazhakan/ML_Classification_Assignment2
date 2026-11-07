# Machine Learning Assignment - 2: Bank Marketing Dataset Classification

## a. Problem Statement

The objective of this assignment is to predict whether a client will subscribe to a term deposit (binary variable 'y') based on direct marketing campaigns of a Portuguese banking institution using five different machine learning classifiers.

## b. Dataset Description

- **Dataset Name:** UCI Bank Marketing Dataset
- **Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Instances:** 10,000 bank-full subset)
- **Features:** 16 input features (age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
- **Target:** Binary (`y`: 1 = Subscribed, 0 = Not Subscribed)

## c. GitHub Repository Link

https://github.com/Muruvazhakan/ML_Classification_Assignment2

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name              | Accuracy |  AUC   | Precision | Recall |   F1   |  MCC   |
| :------------------------- | :------: | :----: | :-------: | :----: | :----: | :----: |
| **Decision Tree**          |  0.9645  | 0.9157 |  0.4925   | 0.4714 | 0.4818 | 0.4635 |
| **Logistic Regression**    |  0.9660  | 0.9726 |  0.5312   | 0.2429 | 0.3333 | 0.3443 |
| **KNN**                    |  0.9635  | 0.8473 |  0.4516   | 0.2000 | 0.2772 | 0.2844 |
| **Naive Bayes**            |  0.9685  | 0.9552 |  0.5443   | 0.6143 | 0.5772 | 0.5620 |
| **Support Vector Machine** |  0.9655  | 0.8720 |  0.5455   | 0.0857 | 0.1481 | 0.2066 |

### Observations on Model Performance

| ML Model Name                        | Observation about model performance                                                                                                    |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **Decision Tree**                    | Strong and balanced model performance with high accuracy and solid MCC, making it a reliable classifier for this dataset.              |
| **Logistic Regression**              | Best AUC among the models, showing excellent discrimination ability, though recall is comparatively low.                               |
| **KNN**                              | Good accuracy but weaker recall and F1, suggesting lower sensitivity to positive cases.                                                |
| **Naive Bayes**                      | Best overall accuracy and strongest recall/F1 balance, indicating the strongest practical classification trade-off.                    |
| **Support Vector Machine**           | High precision but very low recall, which limits its usefulness for detecting positive cases in this dataset.                          |
| **Overall Winner for your dataset?** | **Naive Bayes** performed best overall, achieving the highest Accuracy (0.9685), Recall (0.6143), F1 Score (0.5772), and MCC (0.5620). |
