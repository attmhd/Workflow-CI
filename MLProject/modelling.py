import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "diabetes_dataset_processing.csv")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "test_confusion_matrix.png")

def setup_mlflow():
    """Initialize DagsHub and MLflow tracking"""
    dagshub.init(
        repo_owner="attmhd",
        repo_name="membangun-sistem-ml",
        mlflow=True
    )
    mlflow.set_experiment("Hyperparameter Tuning")

def load_data():
    """Load and preprocess dataset, return train and test split"""
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test

def create_grid_search():
    """Create grid search with hyperparameter grid"""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    return GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2
    )

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix plot"""
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def main():
    setup_mlflow()

    X_train_resampled, y_train_resampled, X_test, y_test = load_data()

    grid_search = create_grid_search()

    with mlflow.start_run(run_name="Random Forest Test 0.2"):
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        grid_search.fit(X_train_resampled, y_train_resampled)

        mlflow.log_param("best_params", grid_search.best_params_)
        mlflow.log_metric("best_accuracy", grid_search.best_score_)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Accuracy:", grid_search.best_score_)

        signature = mlflow.models.infer_signature(X_test, grid_search.best_estimator_.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_,
            artifact_path="tuned_model",
            signature=signature,
            input_example=X_test.head(5)
        )
        
        # Save model locally for Docker build in CI
        local_model_path = os.path.join(BASE_DIR, "tuned_model_local")
        if os.path.exists(local_model_path):
            import shutil
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(
            sk_model=grid_search.best_estimator_,
            path=local_model_path,
            signature=signature,
            input_example=X_test.head(5)
        )
        mlflow.log_artifact(DATASET_PATH, artifact_path="data")

        y_pred = grid_search.best_estimator_.predict(X_test)
        conf_matrix_path = plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
        mlflow.log_artifact(conf_matrix_path, artifact_path="confusion_matrix")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()       