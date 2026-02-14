import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Customer Churn Prediction")


st.sidebar.header("Model Selection")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Choose a Model", model_options)


@st.cache_resource
def load_model(model_name):
    filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error(f"Model file {filename} not found. Please train models first.")
        return None


# Helper: Load Scaler
@st.cache_resource
def load_scaler():
    if os.path.exists('model/scaler.pkl'):
        return joblib.load('model/scaler.pkl')
    return None


model = load_model(selected_model_name)
scaler = load_scaler()

# Main Area - File Upload
st.subheader("1. Upload Test Data (CSV)")
st.info("Please upload the 'test_data.csv'")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        target_col = 'Churn'

        if target_col in df.columns:
            X_test = df.drop(columns=[target_col])
            y_test = df[target_col]

            if selected_model_name in ["Logistic Regression", "KNN"]:
                if scaler:
                    try:
                        X_test = scaler.transform(X_test)
                    except ValueError:
                        st.error("columns mismatch!")
                        st.stop()
                else:
                    st.warning("Scaler not found.")

            # Prediction
            if st.button("Run Prediction"):
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                st.subheader("2. Evaluation Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.4f}")

                st.text("Classification Report:")
                st.code(classification_report(y_test, preds))

                # Confusion Matrix
                st.subheader("3. Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
        else:
            st.error(f"Dataset must contain a '{target_col}' column for evaluation.")

    except Exception as e:
        st.error(f"Error processing file: {e}")