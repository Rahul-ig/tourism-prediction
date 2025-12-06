from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from huggingface_hub import HfApi
import os

DATASET_PATH = "hf://datasets/RahulSingh211/tourism_dataset/tourism.csv"

# Load dataset from Hugging Face
dataset = pd.read_csv(DATASET_PATH)
df = dataset.to_pandas()

# Data Cleaning
print("Original shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Remove CustomerID as it's not needed for modeling
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

# Combine features and target for train and test
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save locally
os.makedirs("tourism_project/data/processed", exist_ok=True)
train_df.to_csv("tourism_project/data/processed/train.csv", index=False)
test_df.to_csv("tourism_project/data/processed/test.csv", index=False)

print(f"\nTrain set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print("\nData preparation completed and saved locally!")

# Upload to Hugging Face
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Push train dataset
train_dataset.push_to_hub("RahulSingh211/tourism_train_data", token=os.environ['HF_TOKEN'])
print("Train data uploaded to Hugging Face!")

# Push test dataset
test_dataset.push_to_hub("RahulSingh211/tourism_test_data", token=os.environ['HF_TOKEN'])
print("Test data uploaded to Hugging Face!")
