import streamlit as st
import joblib
from utils import load_kaggle_dataset, load_uploaded_dataset, show_dataset_summary
from model import split_and_scale, train_all_models

st.title("Credit Card Fraud Detection Model Comparison Platform")
st.caption(
    "Train, evaluate, compare, and download machine learning models for credit card fraud detection."
)
st.sidebar.header("Settings")

dataset = st.sidebar.radio(
    "Select Dataset",
    ("Kaggle Dataset", "Upload your own Dataset")
)
if dataset == "Kaggle Dataset":
    data = load_kaggle_dataset()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )
    if uploaded_file is not None:
        data = load_uploaded_dataset(uploaded_file)

        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("last_uploaded_file") != file_id:
            st.session_state.clear()
            st.session_state.last_uploaded_file = file_id

show_dataset_summary(data)

st.session_state.setdefault("trained_models", None)
st.session_state.setdefault("results_df", None)

if st.session_state.trained_models is None:
    progress = st.progress(0)
    status = st.empty()

    status.info("Splitting & scaling dataset...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = split_and_scale(data)

    def update_progress(name, i, total):
        status.info(f"Training {name}...")
        progress.progress(int((i + 1) / total * 100))

    trained_models, results_df = train_all_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled,
        progress_callback=update_progress
    )

    st.session_state.trained_models = trained_models
    st.session_state.results_df = results_df
    status.success("✅ Training Completed!")

# --- Display results ---
results_df = st.session_state.results_df
trained_models = st.session_state.trained_models
st.dataframe(results_df)

csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Results (CSV)", csv, "model_comparison_results.csv", "text/csv")

st.subheader("Comparison of Model")
best_recall = results_df.loc[results_df["Recall"].idxmax()]
best_precision = results_df.loc[results_df["Precision"].idxmax()]
best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
best_f1 = results_df.loc[results_df["F1 Score"].idxmax()]

col1, col2 = st.columns(2)
with col1:
    st.metric("Best Accuracy", f"{best_accuracy['Accuracy']:.2f}%", best_accuracy["Model"])
    st.metric("Best Precision", f"{best_precision['Precision']:.2f}%", best_precision["Model"])
with col2:
    st.metric("Best Recall", f"{best_recall['Recall']:.2f}%", best_recall["Model"])
    st.metric("Best F1 Score", f"{best_f1['F1 Score']:.2f}%", best_f1["Model"])

st.subheader("Best Overall Model")
st.success(f"""
    **{best_f1['Model']}**

    Accuracy : {best_f1['Accuracy']:.2f}%

    Precision : {best_f1['Precision']:.2f}%

    Recall : {best_f1['Recall']:.2f}%

    F1 Score : {best_f1['F1 Score']:.2f}%
    """)

st.sidebar.subheader("Download Model")
model_list = list(trained_models.keys())
selected_model = st.sidebar.selectbox("Choose a Model", model_list)
model = trained_models[selected_model]
filename = f"{selected_model.replace(' ', '_')}.pkl"
joblib.dump(model, filename)
with open(filename, "rb") as f:
    st.sidebar.download_button(
        "Download Model",
        f,
        filename,
        "application/octet-stream"
    )

st.divider()
st.markdown("Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | [LinkedIn](https://linkedin.com/in/karan-gojiya)")