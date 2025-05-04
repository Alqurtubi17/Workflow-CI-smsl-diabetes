import mlflow
import mlflow.sklearn
import argparse, joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

def main(data_path):
    # Load the preprocessed dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Log the best parameters and model
    with mlflow.start_run():
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)
        
        best_model = grid_search.best_estimator_
        
        mlflow.sklearn.log_model(best_model, "best_model")

        # Optionally, evaluate and log final test accuracy
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Save the model
        joblib.dump(best_model, "model.pkl")

        print(f"Best params: {best_model.get_params()}")
        print(f"Best model accuracy on test set: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        print("Model berhasil dilogging ke DagsHub MLflow Tracking dan di Dump.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to the preprocessed dataset")
    args = parser.parse_args()
    
    main(args.data_path)
