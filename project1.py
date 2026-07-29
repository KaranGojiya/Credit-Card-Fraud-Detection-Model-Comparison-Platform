import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import time
import joblib

st.title("Credit Card Fraud Detection Model Comparison Platform")
st.caption(
    "Train, evaluate, compare, and download machine learning models for credit card fraud detection."
)

dataset = st.radio(label="Select Dataset",options=("Kaggle Dataset", "Upload your own Dataset"))
if dataset == "Kaggle Dataset":
    data = pd.read_csv('creditcard.csv')
else:
    uploaded_file = st.file_uploader("Upload CSV file here", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()
    required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]

    if set(required_columns) != set(data.columns):
        st.error("Unsupported dataset. Please upload the Credit Card Fraud Detection dataset.")
        st.stop()

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

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if st.session_state.trained_models is None:
    progress = st.progress(0)
    status = st.empty()

    # Loading
    status.info("Loading Dataset...")
    progress.progress(10)

    # Split Dataset
    status.info("Splitting Dataset...")
    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    progress.progress(20)

    # Scaling
    status.info("Scaling Features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    progress.progress(30)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel="poly",degree=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=5)
        ,"Random Forest": RandomForestClassifier(n_estimators=200 , n_jobs= -1 ,random_state=42),
        "XGBoost" : XGBClassifier(learning_rate =  0.05, max_depth =  7, n_estimators = 200 , random_state=42 , eval_metric="logloss")
    }

    trained_models = {}
    results = []

    model_progress = {
        "Logistic Regression": 40,
        "Gaussian Naive Bayes": 50,
        "KNN": 60,
        "SVM": 70,
        "Decision Tree": 80,
        "Random Forest" : 90,
        "XGBoost" : 100
    }

    scaled_models = [
     "Logistic Regression",
     "KNN",
     "SVM"
    ]

    for name, model in models.items():

        status.info(f"🤖 Training {name}...")

        start_train = time.perf_counter()

        if name in scaled_models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        end_train = time.perf_counter()

        train_time = end_train - start_train

        trained_models[name] = model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Precision": precision_score(y_test, y_pred) * 100,
            "Recall": recall_score(y_test, y_pred) * 100,
            "F1 Score": f1_score(y_test, y_pred) * 100,
            "Training Time (s)": round(train_time, 4)
        })

        progress.progress(model_progress[name])

    st.session_state.trained_models = trained_models
    st.session_state.results_df = pd.DataFrame(results)
    st.session_state.results_df = st.session_state.results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)
    st.session_state.results_df.index = st.session_state.results_df.index + 1
    st.session_state.results_df.index.name = "Rank"

    csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

    status.success("✅ Training Completed!")

if st.session_state.results_df is not None:

    results_df = st.session_state.results_df
    trained_models = st.session_state.trained_models

    st.dataframe(results_df)

    # Compare model
    st.subheader("Comparison of Model")
    best_recall_model = results_df.loc[results_df["Recall"].idxmax()]
    best_precision_model = results_df.loc[results_df["Precision"].idxmax()]
    best_accuracy_model = results_df.loc[results_df["Accuracy"].idxmax()]
    best_f1_model = results_df.loc[results_df["F1 Score"].idxmax()]

    st.subheader("Best Models")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Best Accuracy",
            f"{best_accuracy_model['Accuracy']:.2f}%",
            best_accuracy_model["Model"]
        )

        st.metric(
            "Best Precision",
            f"{best_precision_model['Precision']:.2f}%",
            best_precision_model["Model"]
        )

    with col2:
        st.metric(
            "Best Recall",
            f"{best_recall_model['Recall']:.2f}%",
            best_recall_model["Model"]
        )

        st.metric(
            "Best F1 Score",
            f"{best_f1_model['F1 Score']:.2f}%",
            best_f1_model["Model"]
        )

    st.subheader("Best Overall Model")

    st.success(f"""
    **{best_f1_model['Model']}**

    Accuracy : {best_f1_model['Accuracy']:.2f}%

    Precision : {best_f1_model['Precision']:.2f}%

    Recall : {best_f1_model['Recall']:.2f}%

    F1 Score : {best_f1_model['F1 Score']:.2f}%
    """)

model_list = list(st.session_state.trained_models.keys())

st.subheader("Select Model for Download")

selected_model = st.selectbox(label="Choose a Model",options=model_list)
model = st.session_state.trained_models[selected_model]
joblib.dump(model, "model.pkl")

with open("model.pkl", "rb") as f:
    st.download_button(
        label="Download Model",
        data=f,
        file_name=f"{selected_model}.pkl",
        mime="application/octet-stream"
    )

st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | "
    "[GitHub](https://github.com/KaranGojiya) | "
    "[LinkedIn](https://linkedin.com/in/karan-gojiya)"
)