import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import pandas as pd
import os, joblib
from dotenv import load_dotenv

# Load env file
load_dotenv()

# Argument parser for MLflow Project compatibility
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="diabetes_preprocessing.csv")
args = parser.parse_args()

# Set DagsHub URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Diabetes Prediction")

# Load data
df = pd.read_csv(args.data)
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define model and hyperparameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

with mlflow.start_run():
    # Log hyperparams and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("cv_best_score", grid_search.best_score_)

    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }

    for name, val in metrics.items():
        mlflow.log_metric(name, val)

    # Log model and optionally artifact to local file
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_artifact(args.data)

    # Save model locally for Docker build (Advanced)
    output_path = "output_model"
    os.makedirs(output_path, exist_ok=True)
    mlflow.sklearn.save_model(best_model, output_path)

    # Dump model:
    joblib.dump(best_model, "model_artifact.pkl")
    mlflow.log_artifact("model_artifact.pkl")


    print("Model training and logging completed.")
