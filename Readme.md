# ML Assignment 2 - Telco Customer Churn

## 1. Problem Statement
The goal of this project is to predict customer churn in a telecommunications company. Our aim is to identify customers who are likely to cancel their subscription.

## 2. Dataset Description
**Dataset:** Telco Customer Churn  
**Source:** IBM / Kaggle  
**Features:** 20 features including 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', etc.  
**Instances:** 7,043 rows.  
**Target:** Binary classification (Churn: Yes/No).

## 3. Models Used & Comparison
Six models were implemented: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost.

### Comparison Table
| ML Model Name       | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| :------------------ | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.7875 | 0.8319 |    0.6206 | 0.5160 |   0.5635 | 0.4278 |
| Decision Tree       |   0.7214 | 0.6567 |    0.4776 | 0.5134 |   0.4948 | 0.3032 |
| KNN                 |   0.7527 | 0.7652 |    0.5367 | 0.5080 |   0.5220 | 0.3556 |
| Naive Bayes         |   0.6560 | 0.8107 |    0.4276 | 0.8690 |   0.5732 | 0.3970 |
| Random Forest       |   0.7854 | 0.8164 |    0.6295 | 0.4679 |   0.5368 | 0.4085 |
| XGBoost             |   0.7633 | 0.8097 |    0.5659 | 0.4706 |   0.5139 | 0.3619 |

### Observations on Model Performance

| ML Model Name | Observation about model performance                                                                                                                                                                                                                                                                    |
| :--- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | **Best Performer.** It achieved the highest Accuracy (78.75%) and AUC (0.8319). The high MCC (0.4278) confirms it has the best balance between precision and recall, suggesting the churn predictors have a strong linear relationship with the target.                                                |
| **Decision Tree** | **Lowest Stability.** It showed the lowest AUC (0.6567) and MCC (0.3032). This indicates the model likely overfitted to the training data and struggled to generalize to the test set compared to the ensemble methods.                                                                                |
| **KNN** | **Average Performance.** With 75.27% accuracy, it performed moderately well but failed to beat the linear models. This suggests that distance-based classification is less effective for this high-dimensional customer data than probability-based methods.                                           |
| **Naive Bayes** | It had the lowest overall Accuracy (65.60%), but achieved a massive **Recall of 86.90%**. It is the best model for ensuring we catch almost every potential churner, though at the cost of many false alarms (Low Precision).                                                                          |
| **Random Forest** | **High Precision.** It performed very similarly to Logistic Regression (78.54% Accuracy) and achieved the highest Precision (62.95%). However, its lower Recall (46.79%) means it is more "conservative" and misses some customers who actually churn.                                                 |
| **XGBoost** | **Good Generalization.** It performed well (76.33% Accuracy) but surprisingly slightly lower than Logistic Regression. This is likely due to the dataset size being on the smaller side for deep learning/boosting, or the default hyperparameters being slightly unoptimized for this specific split. |
