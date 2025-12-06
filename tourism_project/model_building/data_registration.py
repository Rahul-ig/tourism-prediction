from datasets.load import RepositoryNotFoundError

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo
import pandas as pd
import os

repo_type = "dataset"
repo_id = "RahulSingh211/tourism_dataset"

api = HfApi( token=os.getenv['HF_TOKEN'])


# Push to Hugging Face Hub
try:
  api.create_repo(
      repo_id=repo_id, repo_type=repo_type, exist_ok=True
  )
  print(f"Space '{repo_id}' already exists. using it.")
except RepositoryNotFoundError as e:
  print(f"Repository not found: {e}")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}'Created")

api.upload_file(folder_path="tourism_project/data", repo_id=repo_id, repo_type=repo_type)
