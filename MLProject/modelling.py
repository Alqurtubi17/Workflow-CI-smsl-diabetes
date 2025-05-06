import mlflow
import mlflow.sklearn
import pandas as pd
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.base import clone

def main(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Base model
    base_model = RandomForestClassifier(random_state=42)

    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }

    # Grid Search hanya untuk referensi skor
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Simpan dataset sebagai artifact
    df.to_csv("diabetes_preprocessing.csv", index=False)

    # Loop setiap kombinasi parameter dan log ke MLflow
    for i, params in enumerate(grid_search.cv_results_['params']):
        model = clone(base_model)
        model.set_params(**params)
        model.fit(X_train, y_train)

        # Prediksi
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Hitung metrik
        metrics = {
            "accuracy_train": accuracy_score(y_train, y_train_pred),
            "f1_train": f1_score(y_train, y_train_pred),
            "roc_auc_train": roc_auc_score(y_train, y_train_pred),
            "recall_train": recall_score(y_train, y_train_pred),
            "precision_train": precision_score(y_train, y_train_pred),
            "accuracy_test": accuracy_score(y_test, y_test_pred),
            "f1_test": f1_score(y_test, y_test_pred),
            "roc_auc_test": roc_auc_score(y_test, y_test_pred),
            "recall_test": recall_score(y_test, y_test_pred),
            "precision_test": precision_score(y_test, y_test_pred),
            "cv_score": grid_search.cv_results_['mean_test_score'][i]
        }

        run_name = f"rf_n{params['n_estimators']}_d{params['max_depth']}_s{params['min_samples_split']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("diabetes_preprocessing.csv")

            with open("metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact("metrics.json")

    print("Semua kombinasi parameter berhasil dilogging ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path ke file diabetes_preprocessing.csv")
    args = parser.parse_args()
    main(args.data_path)
