"""
Model training script for tourism prediction
"""
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

def load_processed_data():
    """Load preprocessed data"""
    data_dir = os.path.join(os.path.dirname(__file__), "processed_data")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Train tourism prediction model"""
    
    print("\nTraining Random Forest Classifier...")
    
    # Check if MLflow server is available
    mlflow_enabled = False
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=2)
        if response.status_code == 200:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("tourism-prediction")
            mlflow_enabled = True
            print("MLflow tracking enabled")
    except Exception as e:
        print(f"MLflow server not available, proceeding without tracking: {e}")
        mlflow_enabled = False
    
    if mlflow_enabled:
        run_context = mlflow.start_run()
    else:
        from contextlib import nullcontext
        run_context = nullcontext()
    
    with run_context:
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Log parameters
        if mlflow_enabled:
            mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Log metrics
        if mlflow_enabled:
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, 
                                    target_names=['Low', 'Medium', 'High']))
        
        # Print confusion matrix
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_test, y_test_pred))
        
        # Log model
        if mlflow_enabled:
            mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "tourism_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Model saved to {model_path}")
        
        # Save feature importance
        feature_importance = model.feature_importances_
        with open(os.path.join(model_dir, "feature_importance.pkl"), "wb") as f:
            pickle.dump(feature_importance, f)
        
        return model, test_accuracy

def main():
    """Main function"""
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    model, accuracy = train_model(X_train, X_test, y_train, y_test)
    
    print("\n✓ Model training pipeline completed successfully")
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
