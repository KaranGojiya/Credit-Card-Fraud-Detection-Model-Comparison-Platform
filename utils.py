import pandas as pd
import streamlit as st

required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]

def load_kaggle_dataset():
    return pd.read_csv('creditcard.csv')

def load_uploaded_dataset(uploaded_file):
    data = pd.read_csv(uploaded_file)
    if set(required_columns) != set(data.columns):
        st.error("Unsupported dataset. Please upload the Credit Card Fraud Detection dataset.")
        st.stop()
    return data

def show_dataset_summary(data):
    with st.expander("Dataset Summary", expanded = False):
        st.subheader("Dataset Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", data.shape[0])
            st.metric("Columns", data.shape[1])

        with col2:
            st.metric("Fraud Cases", int(data["Class"].sum()))
            st.metric("Normal Cases", int((data["Class"] == 0).sum()))

        missing_values = data.isnull().sum().sum()
        st.metric("Missing Values", int(missing_values))

