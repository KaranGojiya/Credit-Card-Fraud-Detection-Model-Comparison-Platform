# 💳 Credit Card Fraud Detection Model Comparison Platform

An interactive Machine Learning application built with **Streamlit** that trains, evaluates, compares, and exports multiple classification models for credit card fraud detection.

The application supports both the original Kaggle Credit Card Fraud Detection dataset and user-uploaded datasets with the same structure.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small percentage of all transactions.

This project provides an easy-to-use platform to compare the performance of several machine learning algorithms using important evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time

The trained models can also be downloaded for later use.

---

# 🚀 Features

- ✅ Load the original Kaggle dataset
- ✅ Upload custom CSV datasets
- ✅ Automatic dataset validation
- ✅ Dataset summary dashboard
- ✅ Automatic train-test split
- ✅ Feature scaling when required
- ✅ Train multiple ML models automatically
- ✅ Compare model performance
- ✅ Display best model for each metric
- ✅ Download comparison results as CSV
- ✅ Download trained models (.pkl)
- ✅ Progress bar during training
- ✅ Modular project structure

---

# 🤖 Machine Learning Models

The following algorithms are included:

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

---

# 📊 Workflow

<img width="1719" height="2826" alt="app_flow_chart" src="https://github.com/user-attachments/assets/cb9c8410-d4d4-48af-a76c-3bfaa95255d5" />
<img width="1639" height="1993" alt="utils_flow_chart" src="https://github.com/user-attachments/assets/c1052a34-c2af-44ec-8f35-ee8e33e1a6bb" />
<img width="2261" height="1899" alt="model_flow_chart" src="https://github.com/user-attachments/assets/c3d07c02-be75-4b59-aef9-554601342339" />

---

# 🖥 Application Preview

### Home Page

<img width="759" height="327" alt="Screenshot 2026-07-29 174036" src="https://github.com/user-attachments/assets/ea1f05bd-a391-4d48-8e5d-e744dae566aa" />

---

### Dataset Summary

<img width="736" height="680" alt="Screenshot 2026-07-29 174050" src="https://github.com/user-attachments/assets/8553707e-f721-4320-a2da-66bdbc44d4ff" />

---

### Model Comparison

<img width="736" height="362" alt="Screenshot 2026-07-29 174300" src="https://github.com/user-attachments/assets/c97cb967-c66e-4592-876f-400bfda83146" />

<img width="533" height="300" alt="Screenshot 2026-07-29 174310" src="https://github.com/user-attachments/assets/7021ca38-915f-4943-b732-87d578151ad2" />

---

### Best Model Dashboard

<img width="540" height="293" alt="image" src="https://github.com/user-attachments/assets/e4a3371c-aceb-4a03-a679-7bcd560e2329" />

---

### Download Model

<img width="263" height="175" alt="Screenshot 2026-07-29 175748" src="https://github.com/user-attachments/assets/0428998f-c98b-4542-9fed-13c4c6f46a38" />
<img width="264" height="406" alt="Screenshot 2026-07-29 175740" src="https://github.com/user-attachments/assets/d9298a96-5c7b-4d1a-a876-27acc48cf06f" />

---

# 📈 Evaluation Metrics

The application compares every model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time

The model with the highest value for each metric is automatically highlighted.

---

# 📦 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

- 284,807 transactions
- 31 features
- Highly imbalanced dataset
- 492 fraudulent transactions

Dataset Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

# 📚 What I Learned

During this project I learned:

- Data preprocessing
- Feature scaling
- Handling imbalanced datasets
- Model evaluation
- Hyperparameter tuning
- Ensemble learning
- Streamlit application development
- Model serialization using Joblib
- Building modular Python projects

---

# 🔮 Future Improvements

Some features planned for future versions:

- ROC Curve
- Precision-Recall Curve
- Confusion Matrix Heatmap
- Feature Importance Visualization
- SHAP Explainability
- Hyperparameter tuning from UI
- Deep Learning model comparison
- Streamlit Cloud deployment

---

# 👨‍💻 Author

**Karan Gojiya**

GitHub:
https://github.com/KaranGojiya

LinkedIn:
https://linkedin.com/in/karan-gojiya

If you found this project useful, consider giving it a ⭐ on GitHub.
