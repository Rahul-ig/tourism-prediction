from huggingface_hub import HfApi
import os

# Initialize Hugging Face API
api = HfApi()

# Define repository details
SPACE_REPO_ID = "RahulSingh211/tourism-prediction-app"
HF_TOKEN = os.environ.get('HF_TOKEN')

# Create Hugging Face Space (if it doesn't exist)
try:
    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="streamlit",
        token=HF_TOKEN,
        exist_ok=True
    )
    print(f"Space '{SPACE_REPO_ID}' created or already exists!")
except Exception as e:
    print(f"Error creating space: {e}")

# Upload deployment files to the Space
deployment_folder = "tourism_project/deployment"
files_to_upload = ["app.py", "requirements.txt", "Dockerfile"]

for file_name in files_to_upload:
    file_path = os.path.join(deployment_folder, file_name)

    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=SPACE_REPO_ID,
                repo_type="space",
                token=HF_TOKEN
            )
            print(f"Uploaded {file_name} successfully!")
        except Exception as e:
            print(f"Error uploading {file_name}: {e}")
    else:
        print(f"File {file_name} not found at {file_path}")

print(f"\n Deployment completed! Your app should be available at:")
print(f"https://huggingface.co/spaces/{SPACE_REPO_ID}")
