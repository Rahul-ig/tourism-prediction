from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import os

# Initialize HuggingFace API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define repository details
repo_id = "RahulSingh211/tourism_dataset"
repo_type = "dataset"

# Create repository if it doesn't exist
api.create_repo(
    repo_id=repo_id, 
    repo_type=repo_type, 
    private=False,
    exist_ok=True
)
print(f"Repository '{repo_id}' is ready.")

# Upload the tourism.csv file to the repository
print("Uploading tourism.csv to Hugging Face Hub...")
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)
