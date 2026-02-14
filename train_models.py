import os
import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print(f"Downloading dataset from: {url}")
df = pd.read_csv(url)
