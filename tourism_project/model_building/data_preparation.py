from sklearn.model_selection import train_test_split
import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Define the dataset path using hf:// format
DATASET_PATH = "hf://datasets/RahulSingh211/TourismPackagePrediction/tourism.csv"

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
df = pd.read_csv(DATASET_PATH)

# Data Cleaning
print("Original shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Remove CustomerID as it's not needed for modeling
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# Handle missing values
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nAfter cleaning shape:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum().sum())

# Split the data
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save split data as CSV files
os.makedirs("tourism_project/data/processed", exist_ok=True)

X_train.to_csv("tourism_project/data/processed/Xtrain.csv", index=False)
X_test.to_csv("tourism_project/data/processed/Xtest.csv", index=False)
y_train.to_csv("tourism_project/data/processed/ytrain.csv", index=False)
y_test.to_csv("tourism_project/data/processed/ytest.csv", index=False)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print("Data split and saved locally!")

# Initialize HuggingFace API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define repository details for train data
train_repo_id = "RahulSingh211/TourismPackagePrediction"
test_repo_id = "RahulSingh211/TourismPackagePrediction"

# Create repositories if they don't exist
for repo_id in [train_repo_id, test_repo_id]:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository {repo_id} already exists.")
    except RepositoryNotFoundError:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id=repo_id, repo_type="dataset", token=os.getenv("HF_TOKEN"))

# Upload train data files
print("\nUploading train data to Hugging Face...")
api.upload_file(
    path_or_fileobj="tourism_project/data/processed/Xtrain.csv",
    path_in_repo="Xtrain.csv",
    repo_id=train_repo_id,
    repo_type="dataset"
)
api.upload_file(
    path_or_fileobj="tourism_project/data/processed/ytrain.csv",
    path_in_repo="ytrain.csv",
    repo_id=train_repo_id,
    repo_type="dataset"
)
print("Train data uploaded successfully!")

# Upload test data files
print("Uploading test data to Hugging Face...")
api.upload_file(
    path_or_fileobj="tourism_project/data/processed/Xtest.csv",
    path_in_repo="Xtest.csv",
    repo_id=test_repo_id,
    repo_type="dataset"
)
api.upload_file(
    path_or_fileobj="tourism_project/data/processed/ytest.csv",
    path_in_repo="ytest.csv",
    repo_id=test_repo_id,
    repo_type="dataset"
)
print("Test data uploaded successfully!")

print("\n Data preparation completed!")
