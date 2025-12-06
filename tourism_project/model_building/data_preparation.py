"""
Data preparation script for tourism prediction model
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def load_data():
    """Load dataset from Hugging Face Hub or local file"""
    try:
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN")
        
        # Try to load from Hugging Face Hub
        if token:
            try:
                dataset = load_dataset("tourism-prediction-dataset", split="train", token=token)
                df = pd.DataFrame(dataset)
                print("Data loaded from Hugging Face Hub")
            except Exception as e:
                print(f"Could not load from Hub: {e}")
                # Fallback to local file
                data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tourism_data.csv")
                df = pd.read_csv(data_path)
                print("Data loaded from local file")
        else:
            # Load from local file
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tourism_data.csv")
            df = pd.read_csv(data_path)
            print("Data loaded from local file")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to local file
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tourism_data.csv")
        df = pd.read_csv(data_path)
        print("Data loaded from local file")
    
    return df

def prepare_features(df):
    """Prepare features for model training"""
    
    print("\nPreparing features...")
    print(f"Original shape: {df.shape}")
    
    # Create a copy
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['season', 'country']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_le = LabelEncoder()
    df_processed['target_encoded'] = target_le.fit_transform(df_processed['target'])
    label_encoders['target'] = target_le
    
    # Select features for model
    feature_cols = ['month', 'year', 'visitors', 'avg_temp', 'hotel_price', 
                    'event_count', 'season_encoded', 'country_encoded']
    
    X = df_processed[feature_cols]
    y = df_processed['target_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Class distribution in train: {np.bincount(y_train)}")
    print(f"Class distribution in test: {np.bincount(y_test)}")
    
    # Save preprocessing objects
    prep_dir = os.path.join(os.path.dirname(__file__), "preprocessing")
    os.makedirs(prep_dir, exist_ok=True)
    
    with open(os.path.join(prep_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(prep_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    
    with open(os.path.join(prep_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    
    # Save processed data
    data_dir = os.path.join(os.path.dirname(__file__), "processed_data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test.values)
    
    print("\nData preparation completed successfully!")
    print(f"Preprocessed data saved to {data_dir}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    """Main function"""
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_features(df)
    print("\n✓ Data preparation pipeline completed successfully")

if __name__ == "__main__":
    main()
