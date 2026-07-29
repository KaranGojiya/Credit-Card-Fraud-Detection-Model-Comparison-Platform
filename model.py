import time
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_models():
    return{
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel="poly",degree=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=200 , n_jobs= -1 ,random_state=42),
        "XGBoost" : XGBClassifier(learning_rate =  0.05, max_depth =  7, n_estimators = 200 , random_state=42 , eval_metric="logloss")
    }

def split_and_scale(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled,
                      progress_callback=None):
    models = get_models()
    trained_models = {}
    results = []

    scaled_models = [
         "Logistic Regression",
         "KNN",
         "SVM"
    ]
    for i, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(name, i, len(models))

        start_train = time.perf_counter()
        if name in scaled_models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        train_time = time.perf_counter() - start_train

        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Precision": precision_score(y_test, y_pred) * 100,
            "Recall": recall_score(y_test, y_pred) * 100,
            "F1 Score": f1_score(y_test, y_pred) * 100,
            "Training Time (s)": round(train_time, 4),
        })

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score", ascending=False
    ).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "Rank"

    return trained_models, results_df
