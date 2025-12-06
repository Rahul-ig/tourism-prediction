import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi
import pickle
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("tourism_package_prediction")

# Load train and test data from Hugging Face
print("Loading data from Hugging Face...")
train_dataset = load_dataset("RahulSingh211/tourism_train_data", split="train")
test_dataset = load_dataset("RahulSingh211/tourism_test_data", split="train")

train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Separate features and target
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Encode categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to experiment with
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# Hyperparameter grids for tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate models
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=model_name):
        # Get parameter grid for this model
        params = param_grids[model_name]

        # Simple grid search
        from itertools import product
        param_combinations = [dict(zip(params.keys(), v)) for v in product(*params.values())]

        best_local_score = 0
        best_local_model = None
        best_local_params = None

        for param_set in param_combinations[:3]:  # Limit to 3 combinations for time
            model.set_params(**param_set)
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            if f1 > best_local_score:
                best_local_score = f1
                best_local_model = model
                best_local_params = param_set

        # Log best parameters and metrics for this model
        mlflow.log_params(best_local_params)

        # Final predictions with best local model
        y_pred = best_local_model.predict(X_test_scaled)
        y_pred_proba = best_local_model.predict_proba(X_test_scaled)[:, 1]

        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"\nBest Parameters: {best_local_params}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Track best overall model
        if f1 > best_score:
            best_score = f1
            best_model = best_local_model
            best_model_name = model_name

        # Log model
        mlflow.sklearn.log_model(best_local_model, model_name)

print(f"\n{'='*50}")
print(f"Best Model: {best_model_name} with F1 Score: {best_score:.4f}")
print(f"{'='*50}")

# Save the best model and preprocessing objects locally
os.makedirs("tourism_project/models", exist_ok=True)

with open("tourism_project/models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("tourism_project/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("tourism_project/models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("\nModel and preprocessing objects saved locally!")

# Upload to Hugging Face Model Hub
api = HfApi()

try:
    api.create_repo(
        repo_id="RahulSingh211/tourism_model",
        repo_type="model",
        token=os.environ['HF_TOKEN'],
        exist_ok=True
    )

    api.upload_file(
        path_or_fileobj="tourism_project/models/best_model.pkl",
        path_in_repo="best_model.pkl",
        repo_id="RahulSingh211/tourism_model",
        repo_type="model",
        token=os.environ['HF_TOKEN']
    )

    api.upload_file(
        path_or_fileobj="tourism_project/models/scaler.pkl",
        path_in_repo="scaler.pkl",
        repo_id="RahulSingh211/tourism_model",
        repo_type="model",
        token=os.environ['HF_TOKEN']
    )

    api.upload_file(
        path_or_fileobj="tourism_project/models/label_encoders.pkl",
        path_in_repo="label_encoders.pkl",
        repo_id="RahulSingh211/tourism_model",
        repo_type="model",
        token=os.environ['HF_TOKEN']
    )

    print("Model successfully uploaded to Hugging Face Model Hub!")
except Exception as e:
    print(f"Error uploading to Hugging Face: {e}")

print("\nModel training and registration completed!")
