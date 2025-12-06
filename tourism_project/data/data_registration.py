"""
Script to register tourism dataset to Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, create_repo
import pandas as pd

def upload_dataset_to_hub():
    """Upload tourism dataset to Hugging Face Hub"""
    
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found. Skipping upload.")
        return
    
    # Initialize API
    api = HfApi()
    
    # Repository details
    repo_id = "tourism-prediction-dataset"
    
    try:
        # Create repository if it doesn't exist
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=token
        )
        print(f"Repository {repo_id} created/verified successfully")
        
        # Upload the dataset file
        data_file = os.path.join(os.path.dirname(__file__), "tourism_data.csv")
        
        if os.path.exists(data_file):
            api.upload_file(
                path_or_fileobj=data_file,
                path_in_repo="tourism_data.csv",
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            print(f"Dataset uploaded to {repo_id} successfully")
            
            # Verify the upload
            df = pd.read_csv(data_file)
            print(f"\nDataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst few rows:")
            print(df.head())
        else:
            print(f"Error: Data file not found at {data_file}")
            
    except Exception as e:
        print(f"Error uploading dataset: {e}")

if __name__ == "__main__":
    upload_dataset_to_hub()
