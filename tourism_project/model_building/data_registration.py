from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import pandas as pd
import os

repo_type = "dataset"
repo_id = "RahulSingh211/tourism_dataset"

api = HfApi(token=os.getenv('HF_TOKEN'))


# Push to Hugging Face Hub
api.create_repo(
    repo_id=repo_id, repo_type=repo_type, exist_ok=True
)
print(f"Repository '{repo_id}' is ready.")

api.upload_folder(folder_path="tourism_project/data", repo_id=repo_id, repo_type=repo_type)
