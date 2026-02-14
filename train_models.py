import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print(f"Downloading dataset from: {url}")
df = pd.read_csv(url)

df = df.drop(columns=['customerID'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Churn'])
y = df['Churn']

print(f"Final Dataset Shape: {X.shape} (Rows, Cols)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_df = X_test.copy()
test_df['Churn'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("Test data saved as test_data.csv")

