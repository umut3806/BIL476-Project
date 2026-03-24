## Project Description

This project investigates the task of predicting whether a bank client will subscribe to a term deposit based on direct marketing campaign data. Using the UCI Bank Marketing dataset (41,188 records, 20 features), we implement and compare seven classification algorithms: Decision Tree, k-Nearest Neighbours (k-NN), Naive Bayes, Random Forest, XGBoost, LightGBM, and a Stacking Ensemble (Random Forest + XGBoost + LightGBM with Logistic Regression as the meta-learner). Consistent preprocessing is applied across all models, including ordinal encoding for education, one-hot encoding for nominal features, BorderlineSMOTE for class imbalance, and StandardScaler normalization. Hyperparameters are tuned via GridSearchCV or RandomizedSearchCV with 5-fold stratified cross-validation. Classification thresholds are further optimized using precision-recall curve analysis to maximize the F1-score. Experimental results show that ensemble methods, particularly the Stacking Ensemble and LightGBM, outperform simpler classifiers across F1-score and ROC-AUC metrics. The study highlights the importance of imbalance handling, feature encoding choices, and threshold tuning in real-world marketing classification tasks.

## Student Information

Name: Umut Bayram <br>
ID: 221101012 <br>
Department: Computer Engineering
